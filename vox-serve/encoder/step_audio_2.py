from typing import Iterable, Optional, Tuple

import librosa
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


def _mel_filters(n_mels: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram."""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if n_mels == 128:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    else:
        return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))


def load_audio(file_path, target_rate=16000, max_length=None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    If max_length is provided, truncate the audio to that length
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
    audio = waveform[0]  # get the first channel

    # Truncate audio if it exceeds max_length
    if max_length is not None and audio.shape[0] > max_length:
        audio = audio[:max_length]

    return audio

def log_mel_spectrogram(audio, n_mels=128, padding=479, device=None):
    """
    Compute the log-Mel spectrogram with specific padding for StepAudio
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def compute_token_num(max_feature_len):
    # First, audio goes through encoder:
    # 1. conv1: kernel=3, stride=1, padding=1 -> size unchanged
    # 2. conv2: kernel=3, stride=2, padding=1 -> size/2
    # 3. avg_pooler: kernel=2, stride=2 -> size/2
    max_feature_len = max_feature_len - 2  # remove padding
    encoder_output_dim = (max_feature_len + 1) // 2 // 2  # after conv2 and avg_pooler

    # Then through adaptor (parameters from config file):
    padding = 1
    kernel_size = 3  # from config: audio_encoder_config.kernel_size
    stride = 2      # from config: audio_encoder_config.adapter_stride
    adapter_output_dim = (encoder_output_dim + 2 * padding - kernel_size) // stride + 1
    return adapter_output_dim

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.
    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.
    1 for non-padded part and 0 for padded part.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks = [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
                    [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
                    [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class StepAudio2EncoderAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class StepAudio2EncoderBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = StepAudio2EncoderAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x.contiguous()), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x.contiguous()))
        return x


class StepAudio2Encoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = nn.Embedding(n_ctx, n_state)
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[StepAudio2EncoderBlock] = nn.ModuleList(
            [StepAudio2EncoderBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = nn.LayerNorm(n_state)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2::2], x.dtype)  # (B, 1, T // 2)
        x = (x + self.positional_embedding.weight[:x.shape[1], :]).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x.contiguous())
        return x, x_len
