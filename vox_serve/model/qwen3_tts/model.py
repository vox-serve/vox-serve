import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from librosa.filters import mel as librosa_mel_fn
from torch import nn

from ...flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ...sampling import SamplingConfig
from ...tokenizer.qwen3_codec import Qwen3TTSDecoder
from ...utils import get_logger
from ..base import BaseLMWithDepth


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank and uses Hann window for STFT.

    Args:
        y: Input signal.
        n_fft: FFT size.
        num_mels: Number of mel bins.
        sampling_rate: Sampling rate of the input signal.
        hop_size: Hop size for STFT.
        win_size: Window size for STFT.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank. If None, defaults to half the sampling rate.
        center: Whether to pad the input to center the frames. Default is False.

    Returns:
        Mel spectrogram tensor.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device

    mel = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )

    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


@dataclass
class Qwen3TTSSpeakerEncoderConfig:
    enc_dim: int = 2048
    sample_rate: int = 24000
    mel_dim: int = 80
    enc_channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    enc_attention_channels: int = 128


@dataclass
class Qwen3TTSRopeScalingConfig:
    interleaved: bool = True
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    rope_type: str = "default"
    type: str = "default"


@dataclass
class Qwen3TTSCodePredictorConfig:
    _name_or_path: str = ""
    add_cross_attention: bool = False
    architectures: Optional[List[str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bad_words_ids: Optional[Any] = None
    begin_suppress_tokens: Optional[Any] = None
    bos_token_id: Optional[int] = None
    chunk_size_feed_forward: int = 0
    cross_attention_hidden_size: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    diversity_penalty: float = 0.0
    do_sample: bool = False
    early_stopping: bool = False
    encoder_no_repeat_ngram_size: int = 0
    eos_token_id: Optional[int] = None
    exponential_decay_length_penalty: Optional[Any] = None
    finetuning_task: Optional[str] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    id2label: Dict[str, str] = field(default_factory=lambda: {
        "0": "LABEL_0",
        "1": "LABEL_1",
    })
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    is_decoder: bool = False
    is_encoder_decoder: bool = False
    label2id: Dict[str, int] = field(default_factory=lambda: {
        "LABEL_0": 0,
        "LABEL_1": 1,
    })
    layer_types: List[str] = field(default_factory=lambda: [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ])
    length_penalty: float = 1.0
    max_length: int = 20
    max_position_embeddings: int = 65536
    max_window_layers: int = 28
    min_length: int = 0
    model_type: str = "qwen3_tts_talker_code_predictor"
    no_repeat_ngram_size: int = 0
    num_attention_heads: int = 16
    num_beam_groups: int = 1
    num_beams: int = 1
    num_code_groups: int = 16
    num_hidden_layers: int = 5
    num_key_value_heads: int = 8
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    pad_token_id: Optional[int] = None
    prefix: Optional[str] = None
    problem_type: Optional[str] = None
    pruned_heads: Dict[str, Any] = field(default_factory=dict)
    remove_invalid_values: bool = False
    repetition_penalty: float = 1.0
    return_dict: bool = True
    return_dict_in_generate: bool = False
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[Any] = None
    rope_theta: int = 1_000_000
    sep_token_id: Optional[int] = None
    sliding_window: Optional[Any] = None
    suppress_tokens: Optional[Any] = None
    task_specific_params: Optional[Any] = None
    temperature: float = 1.0
    tf_legacy_loss: bool = False
    tie_encoder_decoder: bool = False
    tie_word_embeddings: bool = False
    tokenizer_class: Optional[str] = None
    top_k: int = 50
    top_p: float = 1.0
    dtype: Optional[str] = None
    torchscript: bool = False
    typical_p: float = 1.0
    use_bfloat16: bool = False
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 2048


@dataclass
class Qwen3TTSTalkerConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    code_predictor_config: Qwen3TTSCodePredictorConfig = field(default_factory=Qwen3TTSCodePredictorConfig)

    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_language_id: Dict[str, int] = field(default_factory=lambda: {
        "chinese": 2055,
        "english": 2050,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
    })
    codec_nothink_id: int = 2155
    codec_pad_id: int = 2148
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157

    spk_id: Dict[str, Any] = field(default_factory=dict)
    spk_is_dialect: Dict[str, Any] = field(default_factory=dict)

    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 32768
    model_type: str = "qwen3_tts_talker"
    num_attention_heads: int = 16
    num_code_groups: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    position_id_per_seconds: int = 13
    rms_norm_eps: float = 1e-6
    rope_scaling: Qwen3TTSRopeScalingConfig = field(default_factory=Qwen3TTSRopeScalingConfig)
    rope_theta: int = 1_000_000
    sliding_window: Optional[Any] = None
    text_hidden_size: int = 2048
    text_vocab_size: int = 151_936
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 3072


@dataclass
class Qwen3TTSConfig:
    assistant_token_id: int = 77091
    im_end_token_id: int = 151645
    im_start_token_id: int = 151644
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671

    model_type: str = "qwen3_tts"
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"
    tts_model_size: str = "1b7"
    tts_model_type: str = "base"

    speaker_encoder_config: Qwen3TTSSpeakerEncoderConfig = field(
        default_factory=Qwen3TTSSpeakerEncoderConfig
    )
    talker_config: Qwen3TTSTalkerConfig = field(default_factory=Qwen3TTSTalkerConfig)

    @classmethod
    def from_json(cls, json_path: str):
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        # Create nested configs
        # For base model, set mel_dim=128 if not specified (base model uses 128 mel channels)
        speaker_encoder_config_dict = config_dict.get('speaker_encoder_config', {})
        if config_dict.get('tts_model_type') == 'base' and 'mel_dim' not in speaker_encoder_config_dict:
            speaker_encoder_config_dict['mel_dim'] = 128
        speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(
            **speaker_encoder_config_dict
        )

        # Create talker config with nested code_predictor_config
        talker_config_dict = config_dict.get('talker_config', {})
        code_predictor_config = Qwen3TTSCodePredictorConfig(
            **talker_config_dict.get('code_predictor_config', {})
        )
        rope_scaling_config = Qwen3TTSRopeScalingConfig(
            **talker_config_dict.get('rope_scaling', {})
        )
        talker_config = Qwen3TTSTalkerConfig(
            **{k: v for k, v in talker_config_dict.items()
               if k not in ['code_predictor_config', 'rope_scaling']},
            code_predictor_config=code_predictor_config,
            rope_scaling=rope_scaling_config
        )

        # Create main config
        # Allow missing keys: only use those in the dataclass fields.
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items()
                    if k in fields and k not in ['speaker_encoder_config', 'talker_config']}
        return cls(
            **filtered,
            speaker_encoder_config=speaker_encoder_config,
            talker_config=talker_config
        )


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        # Apply layers
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class Qwen3TTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3TTSRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(
            hidden_states=hidden_states,
            weight=self.weight,
            eps=self.variance_epsilon,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3TTSMLP(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output


class Qwen3TTSAttention(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Initialize inverse frequencies for multimodal RoPE
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_norm(self.q_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_norm(self.k_proj(hidden_states).view(-1, self.head_dim)).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        if self.config.rope_scaling is None:
            # Depth transformer; no mRoPE.
            query_states, key_states = apply_rope_pos_ids(
                query_states=query_states,
                key_states=key_states,
                position_ids=position_ids,
                # rotary_dim=self.head_dim // 2,
                interleave=False,
                rope_theta=self.rope_theta,
            )
        else:
            # While the talker model uses mRoPE, it is essentially the same as standard RoPE.
            query_states, key_states = apply_rope_pos_ids(
                query_states=query_states,
                key_states=key_states,
                position_ids=position_ids,
                interleave=False,  # This is different from mRoPE's interleaving option.
                rope_theta=self.rope_theta,
            )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3TTSDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3TTSAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3TTSMLP(config)

        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3TTSTalkerModel(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.rope_deltas = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3TTSTalkerCodePredictorModel(nn.Module):
    def __init__(self, config: Qwen3TTSCodePredictorConfig, embedding_dim: int):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )
        self.rope_deltas = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3TTSCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        self.register_buffer(
            "lm_head_weight",
            torch.stack([head.weight for head in self.lm_head], dim=0),
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = torch.nn.Identity()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.small_to_mtp_projection(inputs_embeds)

        hidden_states = self.model(
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return hidden_states

class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()

        self.model = Qwen3TTSTalkerModel(config)
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            config.hidden_act,
            bias=True,
        )
        self.codec_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.code_predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config=config.code_predictor_config,
            talker_config=config
        )


class Qwen3TTSSpeakerEncoder(nn.Module):
    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


class Qwen3TTSForCausalLM(nn.Module):
    def __init__(self, config: Qwen3TTSConfig):
        super().__init__()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(
                config.speaker_encoder_config
            )
        else:
            self.speaker_encoder = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.talker.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        logits = self.talker.codec_head(hidden_states)

        return logits, hidden_states

    def forward_depth(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.talker.code_predictor(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        # Choose the codebook head based on the current depth position using tensor ops
        # so CUDA graph capture doesn't require Python-side indexing.
        num_heads = self.talker.code_predictor.lm_head_weight.shape[0]
        head_idx = position_ids.max()
        head_idx = head_idx.clamp(min=1, max=num_heads).sub(1).to(torch.long).view(1)
        head_weight = self.talker.code_predictor.lm_head_weight.index_select(0, head_idx).squeeze(0)
        logits = hidden_states @ head_weight.t()

        return logits

    def forward_depth_unrolled(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        kv_cache: torch.Tensor,
        cache_pos: int,
    ):
        """Depth forward using torch SDPA instead of FlashInfer.

        This avoids FlashInfer workspace buffers entirely, making it safe for
        multi-step CUDA graph capture without plan()/workspace interference.

        Args:
            inputs_embeds: (bs, seq_len, hidden_size) batched input embeddings
            position_ids: (bs, seq_len) position IDs
            kv_cache: (n_layers, bs, 2, max_seq_len, n_kv_heads, head_dim) dense KV cache
            cache_pos: write position in the cache (seq_len tokens starting here)

        Returns:
            logits: (bs, seq_len, vocab_size) — caller should extract last token if needed
        """
        cp = self.talker.code_predictor
        hidden_states = cp.small_to_mtp_projection(inputs_embeds)
        bs, seq_len, _ = hidden_states.shape
        n_kv_heads = cp.model.layers[0].self_attn.num_key_value_heads
        n_q_heads = cp.model.layers[0].self_attn.num_attention_heads
        head_dim = cp.model.layers[0].self_attn.head_dim
        n_groups = n_q_heads // n_kv_heads

        # Flatten position_ids for RoPE: (bs * seq_len,)
        flat_pos = position_ids.reshape(-1)

        for layer_idx, layer in enumerate(cp.model.layers):
            attn = layer.self_attn
            residual = hidden_states
            # RMS norm expects 2D (tokens, hidden), so flatten and restore
            hidden_states = layer.input_layernorm(hidden_states.view(-1, hidden_states.shape[-1])).view(bs, seq_len, -1)

            total_tokens = bs * seq_len
            q = attn.q_norm(attn.q_proj(hidden_states).view(-1, head_dim)).view(total_tokens, n_q_heads, head_dim)
            k = attn.k_norm(attn.k_proj(hidden_states).view(-1, head_dim)).view(total_tokens, n_kv_heads, head_dim)
            v = attn.v_proj(hidden_states).view(total_tokens, n_kv_heads, head_dim)

            q, k = apply_rope_pos_ids(q, k, flat_pos, rope_theta=attn.rope_theta, interleave=False)

            # Reshape to batched: (bs, seq_len, heads, head_dim)
            q = q.view(bs, seq_len, n_q_heads, head_dim)
            k = k.view(bs, seq_len, n_kv_heads, head_dim)
            v = v.view(bs, seq_len, n_kv_heads, head_dim)

            # Write K,V into cache
            kv_cache[layer_idx, :bs, 0, cache_pos:cache_pos + seq_len] = k
            kv_cache[layer_idx, :bs, 1, cache_pos:cache_pos + seq_len] = v

            # Read full K,V up to current position
            valid_len = cache_pos + seq_len
            k_full = kv_cache[layer_idx, :bs, 0, :valid_len]  # (bs, valid_len, n_kv_heads, head_dim)
            v_full = kv_cache[layer_idx, :bs, 1, :valid_len]

            # SDPA expects (bs, n_heads, seq_len, head_dim)
            q_sdpa = q.transpose(1, 2)      # (bs, n_q_heads, seq_len, head_dim)
            k_sdpa = k_full.transpose(1, 2)  # (bs, n_kv_heads, valid_len, head_dim)
            v_sdpa = v_full.transpose(1, 2)

            # GQA: expand KV heads to match Q heads
            if n_groups > 1:
                k_sdpa = k_sdpa.repeat_interleave(n_groups, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(n_groups, dim=1)

            attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=(seq_len > 1))
            # (bs, n_q_heads, seq_len, head_dim) -> (bs, seq_len, hidden_size)
            attn_out = attn_out.transpose(1, 2).reshape(bs, seq_len, -1).contiguous()
            hidden_states = residual + attn.o_proj(attn_out)

            residual = hidden_states
            hs_flat = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = layer.post_attention_layernorm(hs_flat).view(bs, seq_len, -1)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = cp.model.norm(hidden_states.view(-1, hidden_states.shape[-1])).view(bs, seq_len, -1)

        # Head selection based on max position ID
        num_heads = cp.lm_head_weight.shape[0]
        head_idx = position_ids.max().clamp(min=1, max=num_heads).sub(1).to(torch.long).view(1)
        head_weight = cp.lm_head_weight.index_select(0, head_idx).squeeze(0)
        logits = hidden_states @ head_weight.t()

        return logits


class Qwen3TTSModel(BaseLMWithDepth):
    """
    Qwen3 TTS model implementation for VoxServe.

    TODO: Update this implementation based on the actual Qwen3 TTS model architecture.
    """

    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        device="cuda:0",
        tokenizer_path=None,  # TODO: Update with actual tokenizer path
        enable_torch_compile=False,
        audio_decoder_device=None,
        detokenize_interval=None,
    ):
        # Store detokenize_interval (default to 10 if not provided)
        self._detokenize_interval = detokenize_interval if detokenize_interval is not None else 10

        if model_name == "qwen3-tts":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif model_name == "qwen3-tts-base":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        elif model_name == "qwen3-tts-voice-design":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.logger = get_logger(__name__)
        self.model_name = model_name

        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
        )
        config = Qwen3TTSConfig.from_json(config_path)
        self.config = config

        # Create model from config
        self.model = Qwen3TTSForCausalLM(config)

        # Load pretrained weights using transformers utilities
        from transformers.utils import cached_file

        # Get the model directory
        try:
            # Check if there's an index file (for sharded models)
            index_file = cached_file(model_name, "model.safetensors.index.json")
            resolved_archive_file = None
        except Exception:
            try:
                # Single safetensors file
                resolved_archive_file = cached_file(model_name, "model.safetensors")
            except Exception:
                # Fallback to pytorch_model.bin
                try:
                    resolved_archive_file = cached_file(model_name, "pytorch_model.bin")
                except Exception:
                    # Try sharded pytorch format
                    index_file = cached_file(model_name, "pytorch_model.bin.index.json")
                    resolved_archive_file = None

        # Load the state dict
        if resolved_archive_file:
            if resolved_archive_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(resolved_archive_file, device=str(device))
            else:
                state_dict = torch.load(resolved_archive_file, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Handle sharded weights - let transformers handle it
            # Handle sharded weights: download all shard files and load them
            # Download all shard files and load them
            import json
            with open(index_file, 'r') as f:
                index_data = json.load(f)

            state_dict = {}
            for shard_file in set(index_data["weight_map"].values()):
                shard_path = cached_file(model_name, shard_file)
                if shard_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    shard_dict = load_file(shard_path, device=str(device))
                else:
                    shard_dict = torch.load(shard_path, map_location=device)
                state_dict.update(shard_dict)

            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(dtype).to(device)

        # Sync the lm_head_weight buffer with the loaded weights for CUDA graph compatibility
        # The buffer is created at __init__ with random weights, so we need to update it
        # after loading the trained weights
        self.model.talker.code_predictor.lm_head_weight.copy_(
            torch.stack([head.weight for head in self.model.talker.code_predictor.lm_head], dim=0)
        )

        self.supported_speakers = self.config.talker_config.spk_id.keys()
        self.supported_languages = ["auto"]
        for language_id in self.config.talker_config.codec_language_id.keys():
            if "dialect" not in language_id:
                self.supported_languages.append(language_id)

        self.speaker_encoder_sample_rate = self.config.speaker_encoder_config.sample_rate
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type

        tokenizer_path = model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)

        with torch.cuda.device(self.audio_decoder_device):
            if config.tokenizer_type == "qwen3_tts_tokenizer_12hz":
                self.audio_decoder = Qwen3TTSDecoder(
                    device=self.audio_decoder_device,
                    dtype=self.dtype,
                )
                # Note: The audio_decoder.model also supports encoding via the encode() method
                # which is used for voice cloning in the base model
            else:
                raise NotImplementedError(f"Tokenizer type {config.tokenizer_type} not supported")

        self._num_attention_heads = self.config.talker_config.num_attention_heads
        self._num_key_value_heads = self.config.talker_config.num_key_value_heads
        self._num_hidden_layers = self.config.talker_config.num_hidden_layers
        self._hidden_size = self.config.talker_config.hidden_size

        self._depth_num_attention_heads = self.config.talker_config.code_predictor_config.num_attention_heads
        self._depth_num_key_value_heads = self.config.talker_config.code_predictor_config.num_key_value_heads
        self._depth_num_hidden_layers = self.config.talker_config.code_predictor_config.num_hidden_layers
        self._depth_hidden_size = self.config.talker_config.code_predictor_config.hidden_size

        self._vocab_size = self.config.talker_config.vocab_size
        self.stop_token_id = self.config.talker_config.codec_eos_token_id
        self.suppress_tokens = [
            i
            for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
            if i not in (self.config.talker_config.codec_eos_token_id,)
        ]

        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=1.0,
            min_p=None,
            temperature=0.9,
            repetition_penalty=1.05,
            repetition_window=-1,  # global window so repetition_cache is always created when penalty is used
            cfg_scale=None,
        )

    @property
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        return self.config.talker_config.num_code_groups + 1

    @property
    def depth_n_codebooks(self) -> int:
        """Number of codebooks in the depth transformer."""
        return self.config.talker_config.num_code_groups

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        return self._num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        return self._num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        return self._num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Hidden size of the model."""
        return self._hidden_size

    @property
    def head_dim(self) -> int:
        """Head dimension of the model."""
        return self.config.talker_config.head_dim

    @property
    def depth_vocab_size(self) -> int:
        """Vocabulary size of the depth transformer output."""
        return self.config.talker_config.code_predictor_config.vocab_size

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return self._detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 0

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the model generates in a single request."""
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 2048

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return self.detokenize_interval * 1920

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.config.talker_config.vocab_size

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from path, URL, or base64 string.

        Args:
            x: Audio path, URL, or base64 encoded string

        Returns:
            Tuple of (waveform, sample_rate)
        """
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract speaker embedding from audio using the speaker encoder.

        Args:
            audio: Audio waveform (should be resampled to speaker_encoder_sample_rate)
            sr: Sample rate of audio (should match speaker_encoder_sample_rate)

        Returns:
            Speaker embedding tensor of shape (enc_dim,)
        """
        assert sr == 24000, "Only support 24kHz audio for speaker encoder"

        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        # Get mel_dim from speaker encoder config
        mel_dim = self.config.speaker_encoder_config.mel_dim

        # Extract mel spectrogram
        # Parameters from reference implementation:
        # n_fft=1024, num_mels=128 (for base) or 80 (for custom_voice), sampling_rate=24000,
        # hop_size=256, win_size=1024, fmin=0, fmax=12000
        with torch.no_grad():
            mels = mel_spectrogram(
                audio_tensor,
                n_fft=1024,
                num_mels=mel_dim,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000
            ).transpose(1, 2)  # (B, T, num_mels)

            # Pass through speaker encoder
            speaker_embedding = self.model.speaker_encoder(
                mels.to(self.device).to(self.dtype)
            )[0]  # [0] to get first element of batch

        return speaker_embedding

    def _encode_audio_to_codes(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Encode audio to codec tokens using the speech tokenizer encoder.

        Args:
            audio: Audio waveform (numpy array)
            sr: Sample rate of audio

        Returns:
            Audio codes tensor of shape (T, Q) where T is sequence length and Q is num quantizers (16)
        """
        # Convert numpy to torch and ensure correct shape
        audio_tensor = torch.from_numpy(audio).to(self.audio_decoder_device)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

        # Resample if needed (tokenizer expects 24kHz)
        if sr != 24000:
            audio_np = librosa.resample(
                y=audio.astype(np.float32),
                orig_sr=int(sr),
                target_sr=24000
            )
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self.audio_decoder_device)

        # Encode using the tokenizer's encode method
        with torch.no_grad():
            # Create padding mask (all ones = no padding)
            padding_mask = torch.ones_like(audio_tensor, dtype=torch.long)

            # Encode using the audio_decoder's model (which has encode/decode methods)
            # Returns a list of audio codes, one per item in batch
            audio_codes_list = self.audio_decoder.model.encode(
                input_values=audio_tensor,
                padding_mask=padding_mask
            )

            # Get the first (and only) item in the batch
            # audio_codes has shape (T, Q) where T is sequence length, Q is num quantizers (16)
            audio_codes = audio_codes_list[0]

        return audio_codes.to(self.device)

