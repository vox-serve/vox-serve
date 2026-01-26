# Adopted from https://github.com/stepfun-ai/Step-Audio2

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torchaudio
from einops import pack, repeat
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance import kaldi

from ..utils import download_github_file
from .base import DecoderCache
from .hifigan import HiFTGenerator
from .s3 import S3TokenizerV2


@dataclass
class StepAudio2DecoderCache(DecoderCache):
    """Cache tensors used by StepAudio2Decoder.

    All tensors are batch-first.
    """
    spk_emb: torch.Tensor

    # Flow (encoder/estimator) caches
    conformer_cnn_cache: torch.Tensor
    conformer_att_cache: torch.Tensor
    estimator_cnn_cache: torch.Tensor
    estimator_att_cache: torch.Tensor

    # HiFT (vocoder) caches
    hift_mel_cache: torch.Tensor
    hift_source_cache: torch.Tensor
    hift_speech_cache: torch.Tensor


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

    filters_path = download_github_file(
        "xingchensong", "S3Tokenizer", "s3tokenizer/assets/mel_filters.npz",
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


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    """perform fade_in_out in tensor style"""
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class DiTMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTAttention(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(self.inner_dim, dim)

    def to_heads(self, ts: torch.Tensor):
        b, t, c = ts.shape
        ts = ts.reshape(b, t, self.num_heads, c // self.num_heads)
        ts = ts.transpose(1, 2)
        return ts

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Args:
        x(torch.Tensor): shape (b, t, c)
        attn_mask(torch.Tensor): shape (b, t, t)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)
        k = self.to_heads(k)
        v = self.to_heads(v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_chunk(self, x: torch.Tensor, att_cache: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, dt, c)
            att_cache: shape (b, nh, t, c*2)
        """
        b, t, c = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.to_heads(q)
        k = self.to_heads(k)
        v = self.to_heads(v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if att_cache is not None:
            if attn_mask is not None:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

            else:
                k_cache, v_cache = att_cache.chunk(2, dim=3)
                k = torch.cat([k, k_cache], dim=2)
                v = torch.cat([v, v_cache], dim=2)

        new_att_cache = torch.cat([k, v], dim=3)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, new_att_cache


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.scale = 1000

    @staticmethod
    def timestep_embedding(t, dim, max_period=torch.tensor(10000)):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -torch.log(max_period)
            * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTTranspose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class DiTCausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super(DiTCausalConv1d, self).__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super(DiTCausalConv1d, self).forward(x)
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        if cnn_cache is None:
            cnn_cache = x.new_zeros((x.shape[0], self.in_channels, self.causal_padding[0]))
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0] :]
        x = super(DiTCausalConv1d, self).forward(x)
        return x, new_cnn_cache


class DiTCausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.block = torch.nn.Sequential(
            DiTTranspose(1, 2),
            DiTCausalConv1d(in_channels, out_channels, kernel_size),
            DiTTranspose(1, 2),
            nn.LayerNorm(out_channels),
            nn.Mish(),
            DiTTranspose(1, 2),
            DiTCausalConv1d(out_channels, out_channels, kernel_size),
            DiTTranspose(1, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1)
        """
        if mask is not None:
            x = x * mask
        x = self.block(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        """
        Args:
            x: shape (b, dt, c)
            cnn_cache: shape (b, c1+c2, 2)
        """
        if cnn_cache is not None:
            cnn_cache1, cnn_cache2 = cnn_cache.split((self.in_channels, self.out_channels), dim=1)
        else:
            cnn_cache1, cnn_cache2 = None, None
        x = self.block[0](x)
        x, new_cnn_cache1 = self.block[1].forward_chunk(x, cnn_cache1)
        x = self.block[2:6](x)
        x, new_cnn_cache2 = self.block[6].forward_chunk(x, cnn_cache2)
        x = self.block[7](x)
        new_cnn_cache = torch.cat((new_cnn_cache1, new_cnn_cache2), dim=1)
        return x, new_cnn_cache


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, head_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(
            hidden_size, num_heads=num_heads, head_dim=head_dim, qkv_bias=True, qk_norm=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = DiTMLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv = DiTCausalConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor):
        """Args
        x: shape (b, t, c)
        c: shape (b, 1, c)
        attn_mask: shape (b, t, t), bool type attention mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_conv * self.conv(modulate(self.norm3(x), shift_conv, scale_conv))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            c: shape (b, 1, c)
            cnn_cache: shape (b, c1+c2, 2)
            att_cache: shape (b, nh, t, c * 2)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        x_att, new_att_cache = self.attn.forward_chunk(modulate(self.norm1(x), shift_msa, scale_msa), att_cache, mask)
        x = x + gate_msa * x_att
        x_conv, new_cnn_cache = self.conv.forward_chunk(modulate(self.norm3(x), shift_conv, scale_conv), cnn_cache)
        x = x + gate_conv * x_conv
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, new_cnn_cache, new_att_cache


class DiTFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        depth: int = 28,
        num_heads: int = 8,
        head_dim: int = 64,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_embedder = DiTTimestepEmbedder(hidden_size)

        self.in_proj = nn.Linear(in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, head_dim, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = DiTFinalLayer(hidden_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward_chunk(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            x: shape (b, dt, c)
            mu: shape (b, dt, c)
            t: shape (b,)
            spks: shape (b, c)
            cond: shape (b, dt, c)
            cnn_cache: shape (depth, b, c1+c2, 2)
            att_cache: shape (depth, b, nh, t, c * 2)
        """

        # time
        batch_size = x.shape[0]
        t = t.repeat(batch_size // 2)
        t = self.t_embedder(t).unsqueeze(1)  # (b, 1, c)
        x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        if cnn_cache is None:
            cnn_cache = [None] * len(self.blocks)
        if att_cache is None:
            att_cache = [None] * len(self.blocks)
        if att_cache[0] is not None:
            last_att_len = att_cache.shape[3]
        else:
            last_att_len = 0
        chunk_size = x.shape[2]
        mask = None
        x, new_cnn_cache, new_att_cache = self.blocks_forward_chunk(
            x,
            t,
            mask,
            cnn_cache,
            att_cache,
        )

        return x, new_cnn_cache, new_att_cache

    def blocks_forward_chunk(
        self, x, t, mask, cnn_cache=None, att_cache=None
    ):
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        new_cnn_caches = []
        new_att_caches = []
        for b_idx, block in enumerate(self.blocks):
            x, this_new_cnn_cache, this_new_att_cache = block.forward_chunk(
                x, t, cnn_cache[b_idx], att_cache[b_idx], mask
            )
            new_cnn_caches.append(this_new_cnn_cache)
            new_att_caches.append(this_new_att_cache)
        x = self.final_layer(x, t)
        x = x.transpose(1, 2)
        new_cnn_cache = torch.stack(new_cnn_caches, dim=0)
        new_att_cache = torch.stack(new_att_caches, dim=0)
        return x, new_cnn_cache, new_att_cache


class CausalConditionalCFM(torch.nn.Module):
    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels


    def solve_euler_chunk(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        assert self.inference_cfg_rate > 0, "cfg rate should be > 0"

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        if cnn_cache is None:
            cnn_cache = [None for _ in range(len(t_span) - 1)]
        if att_cache is None:
            att_cache = [None for _ in range(len(t_span) - 1)]

        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        new_cnn_caches = []
        new_att_caches = []

        for step in range(1, len(t_span)):
            this_att_cache = att_cache[step - 1]
            this_cnn_cache = cnn_cache[step - 1]

            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.estimator.forward_chunk(
                x=x.repeat(2, 1, 1),
                mu=mu_in,
                t=t.repeat(2),
                spks=spks_in,
                cond=cond_in,
                cnn_cache=this_cnn_cache,
                att_cache=this_att_cache,
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            new_cnn_caches.append(this_new_cnn_cache)
            new_att_caches.append(this_new_att_cache)

        cnn_cache = torch.stack(new_cnn_caches, dim=0)
        att_cache = torch.stack(new_att_caches, dim=0)
        return x, cnn_cache, att_cache

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            mu(torch.Tensor): shape (b, c, t)
            spks(torch.Tensor): shape (b, 192)
            cond(torch.Tensor): shape (b, c, t)
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        offset = att_cache.shape[4] if att_cache is not None else 0
        single_noise = torch.randn(1, mu.size(1), mu.size(2), device=mu.device, dtype=mu.dtype) * temperature
        z = single_noise.expand(mu.size(0), -1, -1)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x, new_cnn_cache, new_att_cache = self.solve_euler_chunk(
            x=z,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        return x, new_cnn_cache, new_att_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        return self._forward_impl(x, mask, pos_emb, mask_pad, att_cache, cnn_cache)

    def _forward_impl(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        """For getting encoding in a streaming fashion

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size,
        ]
        return pos_emb


class FlowEncoderMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                CosyVoice.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(FlowEncoderMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]  # only keep the positions from 0 to time2
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2, scale_factor: float = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)
        self.scale_factor = float(self.stride) if scale_factor is None else float(scale_factor)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride

    def forward_chunk(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ):
        """
        Args:
            inputs(torch.Tensor): shape (b, c, t)
            input_length(torch.Tensor): shape (b), can be None
            cache(torch.Tensor): shape (b, c, cache_t), where cache_t = stride * 2
        """
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")

        if cache is None:
            cache = inputs.new_zeros(inputs.shape[0], inputs.shape[1], self.stride * 2)
        outputs = torch.cat([cache, outputs], dim=2)
        new_cache = outputs[..., -self.stride * 2 :]
        outputs = self.conv(outputs)

        if input_lengths is not None:
            input_lengths = input_lengths * self.stride
        return outputs, input_lengths, new_cache


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        outputs = outputs + inputs
        return outputs

    def forward_chunk(self, inputs: torch.Tensor, cache: torch.Tensor = None):
        """
        Args:
            inputs(torch.Tensor): shape (b, t, c)
            cache(torch.Tensor): shape (b, c, cache_t=2), c = channels
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs[:, : -self.pre_lookahead_len]
        return outputs, new_cache


class UpsampleConformerEncoderV2(torch.nn.Module):
    def __init__(
        self,
        # input & output
        input_size: int,
        output_size: int = 256,
        input_layer: str = "linear",
        pre_lookahead_len: int = 3,
        # size
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        # upsampling
        up_stride: int = 2,
        up_scale_factor: float = 2,
        # attention
        attention_heads: int = 4,
        pos_enc_layer_type: str = "rel_pos_espnet",
        selfattention_layer_type: str = "rel_selfattn",
        key_bias: bool = True,
        # mlp
        linear_units: int = 2048,
        # dropouts
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        # other
        normalize_before: bool = True,
        activation_type: str = "swish",
        # **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        activation = torch.nn.SiLU()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=pre_lookahead_len)
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(
            channels=output_size, out_channels=output_size, stride=up_stride, scale_factor=up_scale_factor
        )
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_up_blocks)
            ]
        )


    def _forward_impl_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def _forward_impl_up_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.up_encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def output_size(self) -> int:
        return self._output_size

    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool = False,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            xs: shape (b, dt, c)
            last_chunk: bool. If last chunk, will pad input with lookaheads
            att_cache: shape (depth1+depth2, b, nh, 2*t1, c).
            cnn_cache: shape (b, c, t1+t2). Where t1=2 (pre_lookahead_layer), t2=4 (up_layer)
        """
        if att_cache is not None:
            assert att_cache.shape[3] % 2 == 0, att_cache.shape
        if cnn_cache is not None:
            assert cnn_cache.shape[2] == 2 + self.up_layer.stride * 2, cnn_cache.shape

        # unpack caches
        offset1 = att_cache.shape[3] // 2 if att_cache is not None else 0
        att_cache1 = (
            att_cache[: len(self.encoders), :, :, :offset1] if att_cache is not None else [None] * len(self.encoders)
        )
        att_cache2 = att_cache[len(self.encoders) :] if att_cache is not None else [None] * len(self.encoders)
        cnn_cache1 = cnn_cache[:, :, :2] if cnn_cache is not None else None
        cnn_cache2 = cnn_cache[:, :, 2:] if cnn_cache is not None else None
        xs, _, _ = self.embed(xs, None)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, self.pre_lookahead_layer.pre_lookahead_len))

        xs, new_cnn_cache1 = self.pre_lookahead_layer.forward_chunk(xs, cache=cnn_cache1)

        pos_emb = self.embed.position_encoding(offset=None, size=offset1 + xs.shape[1])

        chunk_masks = torch.zeros((0, 0, 0))
        new_att_cache1 = []

        for idx, layer in enumerate(self.encoders):
            xs, _, this_new_att_cache1, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache1[idx])
            new_att_cache1.append(this_new_att_cache1)
        new_att_cache1 = torch.stack(new_att_cache1, dim=0)

        xs = xs.transpose(1, 2).contiguous()
        xs, _, new_cnn_cache2 = self.up_layer.forward_chunk(xs, None, cache=cnn_cache2)
        xs = xs.transpose(1, 2).contiguous()

        xs, _, _ = self.up_embed(xs, None)

        pos_emb = self.embed.position_encoding(offset=None, size=offset1 * self.up_layer.stride + xs.shape[1])

        chunk_masks = torch.zeros((0, 0, 0), dtype=torch.bfloat16)
        new_att_cache2 = []

        for idx, layer in enumerate(self.up_encoders):
            xs, _, this_new_att_cache2, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache2[idx])
            new_att_cache2.append(this_new_att_cache2)
        new_att_cache2 = torch.stack(new_att_cache2, dim=0)

        if self.normalize_before:
            xs = self.after_norm(xs)

        new_att_cache = torch.cat([new_att_cache1.repeat(1, 1, 1, 2, 1), new_att_cache2], dim=0)
        new_cnn_cache = torch.cat([new_cnn_cache1, new_cnn_cache2], dim=2)

        return xs, new_cnn_cache, new_att_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 5121,
        encoder: UpsampleConformerEncoderV2 = None,
        decoder: CausalConditionalCFM = None,
        input_embedding: torch.nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        if input_embedding is None:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.input_embedding = input_embedding
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder


    @torch.inference_mode()
    def setup_cache(
        self,
        token: torch.Tensor,
        mel: torch.Tensor,
        spk: torch.Tensor,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            mel: shape (b, t, c), groundtruth mel
            spk: shape (b, 192), speaker embedding
        Returns:
            cache: dict {
                'conformer': {'cnn_cache': xxx, 'att_cache': xxx},
                'estimator': {'cnn_cache': xxx, 'att_cache': xxx}
            }
        """
        # check if look ahead token included
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[1], (token.shape, mel.shape)

        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=False,
            cnn_cache=None,
            att_cache=None,
        )
        h = self.encoder_proj(h)

        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=mel.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=None,
            att_cache=None,
        )

        if isinstance(conformer_att_cache, torch.Tensor):
            conformer_att_cache = conformer_att_cache.permute(1, 0, 2, 3, 4).contiguous()
        if isinstance(estimator_cnn_cache, torch.Tensor):
            estimator_cnn_cache = estimator_cnn_cache.permute(2, 0, 1, 3, 4).contiguous()
        if isinstance(estimator_att_cache, torch.Tensor):
            estimator_att_cache = estimator_att_cache.permute(2, 0, 1, 3, 4, 5).contiguous()

        cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        return cache

    @torch.inference_mode()
    def inference_chunk(
        self,
        token: torch.Tensor,
        spk: torch.Tensor,
        cache: dict,
        last_chunk: bool = False,
        n_timesteps: int = 10,
    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            spk: shape (b, 192), speaker embedding
            cache: dict {
                'conformer_cnn_cache': xxx,
                ...
            }
        """
        # unpack cache (batch-first at API), convert to internal layout
        conformer_cnn_cache = cache["conformer_cnn_cache"]  # (b, c, t) unchanged
        conformer_att_cache = cache["conformer_att_cache"]
        estimator_cnn_cache = cache["estimator_cnn_cache"]
        estimator_att_cache = cache["estimator_att_cache"]

        # Permute cache tensors for internal use
        # NOTE: Skip .contiguous() to avoid memory allocation during CUDA graph capture
        # PyTorch operations work fine with non-contiguous permuted tensors
        if isinstance(conformer_att_cache, torch.Tensor):
            conformer_att_cache = conformer_att_cache.permute(1, 0, 2, 3, 4)
        if isinstance(estimator_cnn_cache, torch.Tensor):
            estimator_cnn_cache = estimator_cnn_cache.permute(1, 2, 0, 3, 4)
        if isinstance(estimator_att_cache, torch.Tensor):
            estimator_att_cache = estimator_att_cache.permute(1, 2, 0, 3, 4, 5)

        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(torch.clamp(token, min=0))
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs=token,
            last_chunk=last_chunk,
            cnn_cache=conformer_cnn_cache,
            att_cache=conformer_att_cache,
        )
        h = self.encoder_proj(h)

        cond = torch.zeros_like(h)
        # For freshly created tensors (mu, cond), we need .contiguous() for downstream ops
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu=h.transpose(1, 2).contiguous(),
            spks=spk,
            cond=cond.transpose(1, 2).contiguous(),
            n_timesteps=n_timesteps,
            temperature=1.0,
            cnn_cache=estimator_cnn_cache,
            att_cache=estimator_att_cache,
        )

        # Permute back to batch-first format for output cache
        # NOTE: Skip .contiguous() to avoid memory allocation during CUDA graph capture
        if isinstance(conformer_att_cache, torch.Tensor):
            conformer_att_cache = conformer_att_cache.permute(1, 0, 2, 3, 4)
        if isinstance(estimator_cnn_cache, torch.Tensor):
            estimator_cnn_cache = estimator_cnn_cache.permute(2, 0, 1, 3, 4)
        if isinstance(estimator_att_cache, torch.Tensor):
            estimator_att_cache = estimator_att_cache.permute(2, 0, 1, 3, 4, 5)

        new_cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }

        return feat, new_cache


class StepAudio2Decoder(nn.Module):
    def __init__(self, model_path, device, float16=False):
        super().__init__()
        self.device = device
        self.float16 = float16

        audio_tokenizer_path = hf_hub_download(
            repo_id=model_path,
            filename="token2wav/speech_tokenizer_v2_25hz.onnx",
            revision=None,
        )
        spk_model_path = hf_hub_download(
            repo_id=model_path,
            filename="token2wav/campplus.onnx",
            revision=None,
        )
        flow_path = hf_hub_download(
            repo_id=model_path,
            filename="token2wav/flow.pt",
            revision=None,
        )
        hift_path = hf_hub_download(
            repo_id=model_path,
            filename="token2wav/hift.pt",
            revision=None,
        )

        self.audio_tokenizer = S3TokenizerV2(audio_tokenizer_path).to(device)

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            spk_model_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

        flow_encoder = UpsampleConformerEncoderV2(
            input_size=512,
            output_size=512,
            input_layer="linear",
            pre_lookahead_len=3,
            num_blocks=6,
            num_up_blocks=4,
            up_stride=2,
            up_scale_factor=2,
            attention_heads=8,
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            key_bias=True,
            linear_units=2048,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
        )
        flow_decoder = CausalConditionalCFM(
            inference_cfg_rate=0.7,
            estimator=DiT(
                in_channels=320,
                out_channels=80,
                mlp_ratio=4.0,
                depth=16,
                num_heads=8,
                head_dim=64,
                hidden_size=512,
            ),
        )
        self.flow = CausalMaskedDiffWithXvec(
            input_size=512,
            output_size=80,
            spk_embed_dim=192,
            output_type="mel",
            vocab_size=6561,
            encoder=flow_encoder,
            decoder=flow_decoder,
        )
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(flow_path, map_location="cpu", weights_only=True), strict=True)
        self.flow.to(self.device).eval()

        self.hift = HiFTGenerator(device=self.device)
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(hift_path, map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(self.device)

        self.preset_audio = download_github_file(
            "stepfun-ai", "Step-Audio2", "assets/default_female.wav"
        )

    def _prepare_prompt(self, prompt_wav):
        audio = load_audio(prompt_wav, sr=16000)  # [T]
        mels = log_mel_spectrogram(audio)
        mels, mels_lens = padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
            device=self.device,
            dtype=torch.float16,
        )

        audio, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device, dtype=torch.float16)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode="replicate",
        )
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def init_cache(self, prompt_wav = None) -> StepAudio2DecoderCache:
        if prompt_wav is None:
            prompt_wav = self.preset_audio
        self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache
        stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1).to(self.device),
            prompt_mels,
            spk_emb,
            n_timesteps=10,
        )
        stream_cache["conformer_att_cache"] = stream_cache["conformer_att_cache"][:, :, :, -128:, :]
        stream_cache["estimator_att_cache"] = stream_cache["estimator_att_cache"][:, :, :, :, -128:, :]

        # Normalize flow caches: unflatten CFG duplication (b' = 2 * b) into explicit CFG dim.
        estimator_cnn_cache = stream_cache.get("estimator_cnn_cache")
        if isinstance(estimator_cnn_cache, torch.Tensor):
            if estimator_cnn_cache.dim() == 5 and estimator_cnn_cache.shape[0] % 2 == 0:
                b = estimator_cnn_cache.shape[0] // 2
                estimator_cnn_cache = estimator_cnn_cache.view(b, 2, *estimator_cnn_cache.shape[1:])

        estimator_att_cache = stream_cache.get("estimator_att_cache")
        if isinstance(estimator_att_cache, torch.Tensor):
            # setup_cache returns (b', n_time, depth, nh, t, d) with b' = 2*b
            if estimator_att_cache.dim() == 6 and estimator_att_cache.shape[0] % 2 == 0:
                b = estimator_att_cache.shape[0] // 2
                estimator_att_cache = estimator_att_cache.view(b, 2, *estimator_att_cache.shape[1:])

        # Initialize HiFT cache with actual mel values from prompt (like CosyVoice does)
        # prompt_mels has shape (B, T, F) = (1, 452, 80), extract last mel_cache_len frames and transpose
        initial_mel_cache = prompt_mels[:, -self.mel_cache_len:, :].transpose(1, 2)

        new_cache = StepAudio2DecoderCache(
            spk_emb=spk_emb,
            conformer_cnn_cache=stream_cache.get("conformer_cnn_cache"),
            conformer_att_cache=stream_cache.get("conformer_att_cache"),
            estimator_cnn_cache=estimator_cnn_cache,
            estimator_att_cache=estimator_att_cache,
            hift_mel_cache=initial_mel_cache,
            hift_source_cache=torch.zeros(1, 1, self.source_cache_len, device=self.device),
            hift_speech_cache=torch.zeros(1, self.source_cache_len, device=self.device),
        )
        return new_cache

    @torch.inference_mode()
    def forward(
        self,
        generated_speech_tokens: torch.Tensor,
        decoder_cache: StepAudio2DecoderCache,
        last_chunk: bool = False,
    ) -> Tuple[torch.Tensor, StepAudio2DecoderCache]:
        estimator_cnn_cache_flat = decoder_cache.estimator_cnn_cache.view(
            -1,
            *decoder_cache.estimator_cnn_cache.shape[2:],
        ) if decoder_cache.estimator_cnn_cache.dim() > 5 else decoder_cache.estimator_cnn_cache

        estimator_att_cache_flat = decoder_cache.estimator_att_cache.view(
            -1,
            *decoder_cache.estimator_att_cache.shape[2:],
        ) if decoder_cache.estimator_att_cache.dim() > 6 else decoder_cache.estimator_att_cache

        stream_cache = {
            "conformer_cnn_cache": decoder_cache.conformer_cnn_cache,
            "conformer_att_cache": decoder_cache.conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache_flat,
            "estimator_att_cache": estimator_att_cache_flat,
        }

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, new_stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=decoder_cache.spk_emb,
                cache=stream_cache,
                last_chunk=last_chunk,
                n_timesteps=10,
            )

        mel = torch.concat([decoder_cache.hift_mel_cache, chunk_mel], dim=2)
        # HiFT runs in fp32, convert from autocast fp16
        speech, source = self.hift(mel.to(torch.float32), decoder_cache.hift_source_cache.to(torch.float32))

        if decoder_cache.hift_speech_cache.shape[-1] > 0:
            speech = fade_in_out(speech, decoder_cache.hift_speech_cache, self.speech_window)

        new_hift_mel_cache = mel[..., -self.mel_cache_len :]
        new_hift_source_cache = source[:, :, -self.source_cache_len :]
        new_hift_speech_cache = speech[:, -self.source_cache_len :]

        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]

        if isinstance(new_stream_cache.get("conformer_att_cache"), torch.Tensor):
            new_stream_cache["conformer_att_cache"] = new_stream_cache["conformer_att_cache"][:, :, :, -128:, :]
        if isinstance(new_stream_cache.get("estimator_att_cache"), torch.Tensor):
            new_stream_cache["estimator_att_cache"] = new_stream_cache["estimator_att_cache"][:, :, :, :, -128:, :]

        estimator_cnn_cache_flat = new_stream_cache.get("estimator_cnn_cache")
        batch_size = estimator_cnn_cache_flat.shape[0] // 2 if estimator_cnn_cache_flat is not None else -1

        estimator_cnn_cache_unflat = new_stream_cache.get("estimator_cnn_cache")
        if estimator_cnn_cache_unflat is not None and batch_size > 0:
            estimator_cnn_cache_unflat = estimator_cnn_cache_unflat.view(
                batch_size, 2, *estimator_cnn_cache_unflat.shape[1:])

        estimator_att_cache_unflat = new_stream_cache.get("estimator_att_cache")
        if estimator_att_cache_unflat is not None and batch_size > 0:
            estimator_att_cache_unflat = estimator_att_cache_unflat.view(
                batch_size, 2, *estimator_att_cache_unflat.shape[1:])

        updated_cache = StepAudio2DecoderCache(
            spk_emb=decoder_cache.spk_emb,
            conformer_cnn_cache=new_stream_cache.get("conformer_cnn_cache"),
            conformer_att_cache=new_stream_cache.get("conformer_att_cache"),
            estimator_cnn_cache=estimator_cnn_cache_unflat
            if batch_size > 0
            else new_stream_cache.get("estimator_cnn_cache"),
            estimator_att_cache=estimator_att_cache_unflat
                if batch_size > 0
                else new_stream_cache.get("estimator_att_cache"),
            hift_mel_cache=new_hift_mel_cache,
            hift_source_cache=new_hift_source_cache,
            hift_speech_cache=new_hift_speech_cache,
        )

        return speech, updated_cache
