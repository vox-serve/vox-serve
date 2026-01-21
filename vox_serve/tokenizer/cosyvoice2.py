from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, OrderedDict, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio
import torchaudio as ta
import torchaudio.compliance.kaldi as Kaldi
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..utils import download_github_file
from .base import DecoderCache
from .cosyvoice_flow import (
    CausalConditionalCFM,
    CausalConditionalDecoder,
    CausalMaskedDiffWithXvec,
    UpsampleConformerEncoder,
)
from .cosyvoice_flow import FlowEncoderCache, FlowDecoderCache
from .hifigan import ConvRNNF0Predictor, HiFTGenerator, HiFTGeneratorCache
from .s3 import ModelConfig as S3ModelConfig
from .s3 import S3TokenizerV2


@dataclass
class CosyVoice2DecoderCache(DecoderCache):
    """Cache tensors used by CosyVoice2Decoder.

    Simplified structure using FlowEncoderCache, FlowDecoderCache, and HiFTGeneratorCache.
    """
    # Flow encoder and decoder caches
    flow_encoder_cache: Optional[FlowEncoderCache] = None
    flow_decoder_cache: Optional[FlowDecoderCache] = None

    # HiFT (vocoder) cache - now stores the complete HiFTGeneratorCache object
    hift_cache: Optional[HiFTGeneratorCache] = None


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    """perform fade_in_out in tensor style"""
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: S3ModelConfig = S3ModelConfig()):
        super().__init__(name, init_from_onnx=False)

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=config.n_mels)
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length
        is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        # accelerator: 'Accelerator'=None,
        max_len: int = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor`
        and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., : max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        # if accelerator is None:
        #     tokenizer = self
        # else:
        #     tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = self.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(audio, self.n_fft, S3_HOP, window=self.window.to(self.device), return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    # features = torch.cat(features)
    return features_padded, feature_lengths, feature_times


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(torch.nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = torch.nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(torch.nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(torch.nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


# @tables.register("model_classes", "CAMPPlus")
class CAMPPlus(torch.nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=False)):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))
        else:
            assert self.output_level == "frame", "`output_level` should be set to 'segment' or 'frame'. "

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))
        return results


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A torch.Tensor containing the audio waveform, in float32 dtype.
    """
    audio, sample_rate = torchaudio.load(file)
    if sample_rate != sr:
        audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
    audio = audio[0]  # get the first channel
    return audio


@lru_cache(maxsize=None)
def _mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    # filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    filters_path = download_github_file(
        "xingchensong",
        "S3Tokenizer",
        "s3tokenizer/assets/mel_filters.npz",
    )

    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the
        audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (128, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = _mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def padding(data: List[torch.Tensor]):
    """Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (128, T)

    Returns:
    -------
        feats [B, 128, T_max], feats lengths [B]
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample], dtype=torch.int32)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False
):
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class CosyVoice2Decoder(nn.Module):
    """
    The decoder of CosyVoice2 is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.
    """

    S3GEN_SR = 24000
    MAX_CACHE_LEN = 128  # Maximum attention cache length for sliding window
    PREFIX_LEN = 16  # Number of prefix tokens to keep for attention sink approach

    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        audio_tokenizer_path = hf_hub_download(
            repo_id=model_path,
            filename="speech_tokenizer_v2.onnx",
            revision=None,
        )
        spk_encoder_path = hf_hub_download(
            repo_id=model_path,
            filename="campplus.onnx",
            revision=None,
        )
        flow_path = hf_hub_download(
            repo_id=model_path,
            filename="flow.pt",
            revision=None,
        )
        hift_path = hf_hub_download(
            repo_id=model_path,
            filename="hift.pt",
            revision=None,
        )
        self.tokenizer = S3Tokenizer(audio_tokenizer_path)
        self.mel_extractor = mel_spectrogram  # TODO: make it a torch module?
        # self.speaker_encoder = CAMPPlus()  # use default args

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            # dropout_rate=0.1,
            # positional_dropout_rate=0.1,
            # attention_dropout_rate=0.1,
            # normalize_before=True,
            # input_layer='linear',
            # pos_enc_layer_type='rel_pos_espnet',
            # selfattention_layer_type='rel_selfattn',
            input_size=512,
            # use_cnn_module=False,
            # macaron_style=False,
        )

        estimator = CausalConditionalDecoder(
            in_channels=320,
            out_channels=80,
            # causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
        )
        # estimator.forward = torch.compile(estimator.forward, mode="max-autotune-no-cudagraphs")
        decoder = CausalConditionalCFM(
            in_channels=240,
            spk_emb_dim=80,
            # cfm_params=CFM_PARAMS,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)
        self.flow.load_state_dict(torch.load(flow_path, map_location="cpu", weights_only=True), strict=True)
        self.flow.to(self.device).to(torch.bfloat16).eval()

        self.resamplers = {}

        f0_predictor = ConvRNNF0Predictor()
        self.hift = HiFTGenerator(
            sampling_rate=self.S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(hift_path, map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = self.S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)  # (buffers get automatic device casting)

        # Cache configuration for streaming
        self.mel_cache_len = 6 # cache length for mel spectrograms
        self.source_cache_len = int(self.mel_cache_len * 480)  # 50hz mel -> 24kHz wave conversion
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(self.device)

    def init_cache(self, ref_dict: dict) -> CosyVoice2DecoderCache:
        """Initialize cache for streaming inference by processing speech tokens."""
        # Get prompt speech tokens for cache initialization
        prompt_speech_tokens = torch.cat([
            ref_dict["prompt_speech_token"], 
            ref_dict["prompt_speech_token"][:, :3],
        ], dim=1).to(self.device)
        prompt_speech_token_len = torch.ones(
            1, device=self.device, dtype=torch.long
        ) * (ref_dict["prompt_speech_token_len"] + 3)
        
        # Initialize empty caches
        empty_encoder_cache = FlowEncoderCache()
        empty_decoder_cache = FlowDecoderCache()
        
        # Process speech tokens through flow.forward_chunk to initialize caches
        with torch.no_grad():
            prompt_mels, flow_encoder_cache, flow_decoder_cache = self.flow.forward_chunk(
                token=prompt_speech_tokens,
                token_len=prompt_speech_token_len,
                prompt_feat=ref_dict["prompt_feat"],
                prompt_feat_len=ref_dict["prompt_feat_len"],
                embedding=ref_dict["embedding"],
                encoder_cache=empty_encoder_cache,
                decoder_cache=empty_decoder_cache,
                last_chunk=False,
            )

        if flow_encoder_cache.conformer_att_cache is not None:
            # flow_encoder_cache.conformer_att_cache = flow_encoder_cache.conformer_att_cache[:, :, :, :self.PREFIX_LEN // 2, :]
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN // 2 - PREFIX_LEN tokens
            cache_len = flow_encoder_cache.conformer_att_cache.shape[3]
            max_len = self.MAX_CACHE_LEN // 2
            if cache_len > max_len:
                prefix = flow_encoder_cache.conformer_att_cache[:, :, :, :self.PREFIX_LEN // 2, :]
                suffix = flow_encoder_cache.conformer_att_cache[:, :, :, -(max_len - self.PREFIX_LEN // 2):, :]
                flow_encoder_cache.conformer_att_cache = torch.cat([prefix, suffix], dim=3)

        if flow_encoder_cache.up_conformer_att_cache is not None:
            # flow_encoder_cache.up_conformer_att_cache = flow_encoder_cache.up_conformer_att_cache[:, :, :, :self.PREFIX_LEN, :]
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN - PREFIX_LEN tokens
            cache_len = flow_encoder_cache.up_conformer_att_cache.shape[3]
            if cache_len > self.MAX_CACHE_LEN:
                prefix = flow_encoder_cache.up_conformer_att_cache[:, :, :, :self.PREFIX_LEN, :]
                suffix = flow_encoder_cache.up_conformer_att_cache[:, :, :, -(self.MAX_CACHE_LEN - self.PREFIX_LEN):, :]
                flow_encoder_cache.up_conformer_att_cache = torch.cat([prefix, suffix], dim=3)

        if flow_decoder_cache.att_cache is not None:
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN - PREFIX_LEN tokens
            cache_len = flow_decoder_cache.att_cache.shape[5]
            if cache_len > self.MAX_CACHE_LEN:
                prefix = flow_decoder_cache.att_cache[:, :, :, :, :, :self.PREFIX_LEN, :]
                suffix = flow_decoder_cache.att_cache[:, :, :, :, :, -(self.MAX_CACHE_LEN - self.PREFIX_LEN):, :]
                flow_decoder_cache.att_cache = torch.cat([prefix, suffix], dim=5)

        # Initialize HiFT cache with real mel values from prompt processing
        initial_mel_cache = prompt_mels[:, :, -self.mel_cache_len:]
        
        hift_cache = HiFTGeneratorCache(
            mel_cache=initial_mel_cache,
            source_cache=torch.zeros(1, 1, self.source_cache_len, device=self.device, dtype=torch.float32),
            speech_cache=torch.zeros(1, self.source_cache_len, device=self.device, dtype=torch.bfloat16),
        )

        return CosyVoice2DecoderCache(
            flow_encoder_cache=flow_encoder_cache,
            flow_decoder_cache=flow_decoder_cache,
            hift_cache=hift_cache,
        )

    @torch.inference_mode()
    def decode_chunk(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lens: int,
        decoder_cache: CosyVoice2DecoderCache,
        ref_dict: Optional[dict] = None,
        last_chunk: bool = False,
    ) -> Tuple[torch.Tensor, CosyVoice2DecoderCache]:
        """Forward pass with caching for streaming inference.
        
        Args:
            speech_tokens: Token sequence to decode
            speech_token_lens: Length of token sequence
            decoder_cache: Cache state from previous chunk
            ref_dict: Reference audio features
            last_chunk: Whether this is the last chunk
            
        Returns:
            Tuple of (audio_output, updated_cache)
        """

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # Convert speech_token_lens to tensor if it's an int
        if isinstance(speech_token_lens, int):
            speech_token_lens = torch.ones(speech_tokens.size(0), device=self.device, dtype=torch.long) * speech_token_lens

        output_mels, new_flow_encoder_cache, new_flow_decoder_cache = self.flow.forward_chunk(
            token=speech_tokens,
            token_len=speech_token_lens,
            prompt_feat=torch.zeros(1, 0, 80, device=self.device, dtype=torch.bfloat16),
            prompt_feat_len=0,
            embedding=ref_dict["embedding"],
            encoder_cache=decoder_cache.flow_encoder_cache,
            decoder_cache=decoder_cache.flow_decoder_cache,
            last_chunk=last_chunk,
        )

        if new_flow_encoder_cache.conformer_att_cache is not None:
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN // 2 - PREFIX_LEN tokens
            cache_len = new_flow_encoder_cache.conformer_att_cache.shape[3]
            max_len = self.MAX_CACHE_LEN // 2
            if cache_len > max_len:
                prefix = new_flow_encoder_cache.conformer_att_cache[:, :, :, :self.PREFIX_LEN // 2, :]
                suffix = new_flow_encoder_cache.conformer_att_cache[:, :, :, -(max_len - self.PREFIX_LEN // 2):, :]
                new_flow_encoder_cache.conformer_att_cache = torch.cat([prefix, suffix], dim=3)

        if new_flow_encoder_cache.up_conformer_att_cache is not None:
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN - PREFIX_LEN tokens
            cache_len = new_flow_encoder_cache.up_conformer_att_cache.shape[3]
            if cache_len > self.MAX_CACHE_LEN:
                prefix = new_flow_encoder_cache.up_conformer_att_cache[:, :, :, :self.PREFIX_LEN, :]
                suffix = new_flow_encoder_cache.up_conformer_att_cache[:, :, :, -(self.MAX_CACHE_LEN - self.PREFIX_LEN):, :]
                new_flow_encoder_cache.up_conformer_att_cache = torch.cat([prefix, suffix], dim=3)

        if new_flow_decoder_cache.att_cache is not None:
            # Attention sink approach: keep first PREFIX_LEN and last MAX_CACHE_LEN - PREFIX_LEN tokens
            cache_len = new_flow_decoder_cache.att_cache.shape[5]
            if cache_len > self.MAX_CACHE_LEN:
                prefix = new_flow_decoder_cache.att_cache[:, :, :, :, :, :self.PREFIX_LEN, :]
                suffix = new_flow_decoder_cache.att_cache[:, :, :, :, :, -(self.MAX_CACHE_LEN - self.PREFIX_LEN):, :]
                new_flow_decoder_cache.att_cache = torch.cat([prefix, suffix], dim=5)

        # HiFT forward pass with caching
        # output_mels = torch.concat(
        #     [decoder_cache.hift_cache.mel_cache, output_mels], 
        #     dim=2, 
        # ).to(torch.float32).to(self.device)

        output_mels = output_mels.to(torch.float32).to(self.device)
        output_wavs, source = self.hift.forward_chunk(
            output_mels,
            # decoder_cache.hift_cache.source_cache,
        )
        output_wavs = fade_in_out(
            output_wavs, 
            decoder_cache.hift_cache.speech_cache, 
            self.speech_window,
        )

        new_hift_cache = HiFTGeneratorCache(
            mel_cache=output_mels[..., -self.mel_cache_len:],
            source_cache=source[:, :, -self.source_cache_len:],
            speech_cache=output_wavs[:, -self.source_cache_len:],
        )
        output_wavs = output_wavs[:, :-self.source_cache_len]

        # Update caches
        updated_cache = CosyVoice2DecoderCache(
            flow_encoder_cache=new_flow_encoder_cache,
            flow_decoder_cache=new_flow_decoder_cache,
            hift_cache=new_hift_cache,
        )

        return output_wavs, updated_cache

    @torch.inference_mode()
    def decode(
        self,
        speech_tokens,
        speech_token_lens: int,
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        speech_tokens = torch.concat(
            [ref_dict["prompt_speech_token"].repeat(speech_tokens.size(0), 1), speech_tokens],
            dim=1,
        )
        speech_token_lens = torch.ones(speech_tokens.size(0), device=self.device, dtype=torch.long) * (
            ref_dict["prompt_speech_token_len"] + speech_token_lens
        )

        output_mels, _ = self.flow(
            token=speech_tokens,
            token_len=speech_token_lens,
            prompt_feat=ref_dict["prompt_feat"],
            prompt_feat_len=ref_dict["prompt_feat_len"],
            embedding=ref_dict["embedding"],
            streaming=True,
            finalize=finalize,
        )
        output_mels = output_mels[:, :, ref_dict["prompt_feat_len"] :]

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        cache_source = torch.zeros(1, 1, 0, dtype=torch.bfloat16, device=self.device)

        output_wavs, output_sources = self.hift.forward(output_mels.to(torch.bfloat16), cache_source)

        # # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        # output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs
