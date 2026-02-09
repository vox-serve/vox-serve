import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy import signal
from torch import Tensor, nn


@dataclass
class T3Cond:
    """
    Dataclass container for most / all conditioning info.
    TODO: serialization methods aren't used, keeping them around for convenience
    """

    speaker_emb: torch.Tensor
    clap_emb: Optional[torch.Tensor] = None
    cond_prompt_speech_tokens: Optional[torch.Tensor] = None
    cond_prompt_speech_emb: Optional[torch.Tensor] = None
    emotion_adv: Optional[torch.Tensor] = 0.5

    def to(self, *, device=None, dtype=None):
        "Cast to a device and dtype. Dtype casting is ignored for long/int tensors."
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                is_fp = type(v.view(-1)[0].item()) is not int
                setattr(self, k, v.to(device=device, dtype=dtype if is_fp else None))
        return self

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return T3Cond(**kwargs)


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim**-0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.flash_config = self.setup_flash_config() if flash else None

    def setup_flash_config(self):
        # Setup flash attention configuration
        flash_config = {"enable_flash": True, "enable_math": True, "enable_mem_efficient": True}
        return flash_config

    def forward(self, q, k, v, mask=None):
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        if self.flash:
            out = self.flash_attention(q, k, v, mask=mask)
        else:
            out = self.scaled_dot_product_attention(q, k, v, mask=mask)

        return self.combine_heads(out)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bhts,bhls->bhlt", attn, v)

    def flash_attention(self, q, k, v, mask=None):
        config = self.flash_config if self.flash_config else {}
        with torch.backends.cuda.sdp_kernel(**config):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                # dropout_p=self.dropout_rate if self.training else 0.
                dropout_p=0,
            )
        return out

    def split_heads(self, x):
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, length, -1)


class AttentionBlock2(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other,
    using AttentionQKV and separate linear transformations for Q, K, and V.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        relative_pos_embeddings=False,
        flash_attention=True,
        dropout_rate=0.2,
        scale=None,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, and V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(
            self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale
        )

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64,
            )
        else:
            self.relative_pos_embeddings = None

    def forward(self, x1, x2, mask=None):
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)


class ChatterboxPerceiver(nn.Module):
    def __init__(
        self, pre_attention_query_token=32, pre_attention_query_size=1024, embedding_dim=1024, num_attn_heads=4
    ):
        """
        Initialize the perceiver module.

        :param pre_attention_query_token: Number of query tokens for pre-attention
        :param pre_attention_query_size: Size of each query token
        :param embedding_dim: Dimension of the embedding space
        :param num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Initialize the pre-attention query parameter
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, pre_attention_query_token, pre_attention_query_size)
        )

        # Calculate the variance for uniform initialization
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))

        # Initialize the pre-attention query with uniform distribution
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)

        # Initialize the attention block
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def forward(self, h):
        """
        Forward pass of the perceiver module.
        :param h: Input tensor
        :return: Output after applying attention mechanisms
        """
        # Expand the pre-attention query to match the batch size of the input
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)
        # Apply the first attention mechanism (cross-attention)
        pre_att = self.attn(query_, h)
        # Apply the second attention mechanism (self-attention)
        attn = self.attn(pre_att, pre_att)
        return attn


class ChatterboxCondEnc(nn.Module):
    """
    Handle all non-text conditioning, like speaker embeddings / prompts, CLAP, emotion, etc.
    """

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))

        # emotion adv
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # perceiver resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = ChatterboxPerceiver()

    @torch.inference_mode()
    def forward(self, cond: T3Cond):
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), (
            "no embeddings for cond_prompt_speech_tokens"
        )

        # Speaker embedding projection
        cond_spkr = self.spkr_enc(cond.speaker_emb.view(-1, self.hp.speaker_embed_size))[:, None]  # (B, 1, dim)
        empty = torch.zeros_like(cond_spkr[:, :0])  # (B, 0, dim)

        # TODO CLAP
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))

        # Concat and return
        cond_embeds = torch.cat(
            (
                cond_spkr,
                cond_clap,
                cond_prompt_speech_emb,
                cond_emotion_adv,
            ),
            dim=1,
        )
        return cond_embeds


# =============================================================================
# Voice Encoder for speaker embedding extraction
# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License
# =============================================================================


@dataclass
class VoiceEncConfig:
    """Configuration for voice encoder."""

    num_mels: int = 40
    sample_rate: int = 16000
    speaker_embed_size: int = 256
    ve_hidden_size: int = 256
    flatten_lstm_params: bool = False
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
    fmax: int = 8000
    fmin: int = 0
    preemphasis: float = 0.0
    mel_power: float = 2.0
    mel_type: str = "amp"
    normalized_mels: bool = False
    ve_partial_frames: int = 160
    ve_final_relu: bool = True
    stft_magnitude_min: float = 1e-4


@lru_cache()
def _ve_mel_basis(
    sample_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: int,
    fmax: int,
):
    """Compute mel filterbank for voice encoder."""
    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )


def ve_melspectrogram(wav: np.ndarray, hp: VoiceEncConfig, pad: bool = True) -> np.ndarray:
    """Compute mel spectrogram from waveform for voice encoder.

    Args:
        wav: Audio waveform as numpy array
        hp: Voice encoder configuration
        pad: Whether to pad the STFT

    Returns:
        Mel spectrogram of shape (M, T) where M is num_mels
    """
    # Pre-emphasis if needed
    if hp.preemphasis > 0:
        wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
        wav = np.clip(wav, -1, 1)

    # STFT
    spec_complex = librosa.stft(
        wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )

    # Magnitude spectrogram
    spec_magnitudes = np.abs(spec_complex)
    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Apply mel filterbank
    mel_basis = _ve_mel_basis(hp.sample_rate, hp.n_fft, hp.num_mels, hp.fmin, hp.fmax)
    mel = np.dot(mel_basis, spec_magnitudes)

    # Convert to dB if needed
    if hp.mel_type == "db":
        mel = 20 * np.log10(np.maximum(hp.stft_magnitude_min, mel))

    # Normalize if needed
    if hp.normalized_mels:
        min_level_db = 20 * np.log10(hp.stft_magnitude_min)
        headroom_db = 15
        mel = (mel - min_level_db) / (-min_level_db + headroom_db)
        mel = mel.astype(np.float32)

    return mel  # (M, T)


def _ve_pack(arrays: List, seq_len: int = None, pad_value: float = 0) -> torch.Tensor:
    """Pack a list of arrays into a single tensor by padding."""
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # Convert lists to np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # Convert to tensor and handle device
    device = None
    if isinstance(arrays[0], torch.Tensor):
        tensors = arrays
        device = tensors[0].device
    else:
        tensors = [torch.as_tensor(array) for array in arrays]

    # Fill the packed tensor with the array data
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)

    for i, tensor in enumerate(tensors):
        packed_tensor[i, : tensor.size(0)] = tensor

    return packed_tensor


def _ve_get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    """Get number of windows that fit in the mel spectrogram."""
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def _ve_get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
) -> int:
    """Compute how many frames separate two partial utterances."""
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


class VoiceEncoder(nn.Module):
    """LSTM-based voice encoder for speaker embedding extraction.

    Takes mel spectrograms and produces L2-normalized speaker embeddings.
    """

    def __init__(self, hp: VoiceEncConfig = None):
        super().__init__()

        self.hp = hp if hp is not None else VoiceEncConfig()

        # Network definition
        self.lstm = nn.LSTM(
            self.hp.num_mels,
            self.hp.ve_hidden_size,
            num_layers=3,
            batch_first=True,
        )
        if self.hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Cosine similarity scaling (for training, not used in inference)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]), requires_grad=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, mels: torch.FloatTensor) -> torch.Tensor:
        """Compute embeddings from batch of mel spectrograms.

        Args:
            mels: Batch of mel spectrograms, shape (B, T, M)

        Returns:
            L2-normalized speaker embeddings, shape (B, E)
        """
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise ValueError(f"Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}")

        # Pass through LSTM
        _, (hidden, _) = self.lstm(mels)

        # Project the final hidden state
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2 normalize
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(
        self,
        mels: torch.Tensor,
        mel_lens: List[int],
        overlap: float = 0.5,
        rate: float = None,
        min_coverage: float = 0.8,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Compute embeddings from full utterances with partials.

        Args:
            mels: Mel spectrograms, shape (B, T, M)
            mel_lens: List of mel lengths
            overlap: Overlap ratio between partials
            rate: Partials rate
            min_coverage: Minimum coverage for last partial
            batch_size: Batch size for processing partials

        Returns:
            Speaker embeddings, shape (B, E)
        """
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens

        # Compute where to split the utterances into partials
        frame_step = _ve_get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(
            *(_ve_get_num_wins(mel_len, frame_step, min_coverage, self.hp) for mel_len in mel_lens),
            strict=False,
        )

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        # Group all partials together
        partials = [
            mel[i * frame_step : i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials, strict=False)
            for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = torch.stack(partials)

        # Forward the partials
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # Reduce the partial embeds into full embeds and L2-normalize them
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [
            torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:], strict=False)
        ]
        raw_embeds = torch.stack(raw_embeds)
        embeds = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray) -> np.ndarray:
        """Reduce utterance embeddings to a single speaker embedding."""
        assert utt_embeds.ndim == 2
        utt_embeds = np.mean(utt_embeds, axis=0)
        return utt_embeds / np.linalg.norm(utt_embeds, 2)

    def embeds_from_mels(
        self,
        mels: Union[Tensor, List[np.ndarray]],
        mel_lens: List[int] = None,
        as_spk: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Compute embeddings from mel spectrograms."""
        # Load mels in memory and pack them
        if isinstance(mels, list):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels), "Mels must have same feature dim"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = _ve_pack(mels)

        # Embed them
        with torch.inference_mode():
            utt_embeds = self.inference(mels.to(self.device), mel_lens, batch_size=batch_size, **kwargs).numpy()

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: Optional[float] = 20,
        **kwargs,
    ) -> np.ndarray:
        """Compute embeddings from audio waveforms.

        Args:
            wavs: List of audio waveforms as numpy arrays
            sample_rate: Sample rate of input audio
            as_spk: Whether to return a single speaker embedding
            batch_size: Batch size for processing partials
            trim_top_db: Silence trimming threshold in dB
            **kwargs: Additional args for embeds_from_mels()

        Returns:
            Embeddings as (B, E) array if as_spk=False, else (E,) array
        """
        # Resample if needed
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]

        # Trim silence
        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        # Set default rate
        if "rate" not in kwargs:
            kwargs["rate"] = 1.3

        # Extract mel spectrograms
        mels = [ve_melspectrogram(w, self.hp).T for w in wavs]

        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)
