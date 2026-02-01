from typing import Any, List, Tuple
import json

import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.qwen3_codec import Qwen3TTSDecoder
from .base import BaseLM, PreprocessOutput


from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


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
        speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(
            **config_dict.get('speaker_encoder_config', {})
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
        return cls(
            **{k: v for k, v in config_dict.items()
               if k not in ['speaker_encoder_config', 'talker_config']},
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

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # TODO: mrope
        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            # rotary_dim=self.head_dim // 2,
            interleave=self.rope_scaling.interleaved,
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
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
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
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.codec_embedding = nn.Embedding(
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
    def __init__(self, config: Qwen3TTSCodePredictorConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerCodePredictorModel(config)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
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
    
    def depth_forward(
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
        logits = self.talker.code_predictor.lm_head[position_ids](hidden_states)

        return logits


class Qwen3TTSModel(BaseLM):
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
    ):
        if model_name == "qwen3-tts":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.model_name = model_name

        config_path = hf_hub_download(
            repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            filename="config.json",
        )
        config = Qwen3TTSConfig.from_json(config_path)
        self.config = config

        self.model = Qwen3TTSForCausalLM(config)
        self.model.to(dtype).to(device)

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
                self.audio_decoder = Qwen3TTSDecoder(device=self.audio_decoder_device)
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

        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=1.0,
            min_p=None,
            temperature=0.9,
            repetition_penalty=1.05,
            repetition_window=None,
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
    def depth_num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        return self._depth_num_attention_heads

    @property
    def depth_num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        return self._depth_num_key_value_heads

    @property
    def depth_num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        return self._depth_num_hidden_layers

    @property
    def depth_hidden_size(self) -> int:
        """Hidden size of the model."""
        return self._depth_hidden_size

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 10

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
        return 2048

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.config.talker_config.vocab_size

    @property
    def supports_audio_input(self) -> bool:
        """Indicates if the model accepts audio input."""
        # TODO: support later
        return False

    @property
    def needs_watermarking(self) -> bool:
        """Indicates if the model requires watermarking."""
        return False

    @property
    def watermarker_type(self) -> str:
        """Indicates the watermarker type to use."""
        return None

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return True

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return True

    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return True

    def is_stop_id(self, token_ids: List[int]) -> bool:
        """Check if the given token ID is a stop token."""
        return token_ids[0] == self.stop_token_id

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)

    def preprocess(
        self,
        prompt: str = None,
        audio_path: str = None,
        **kwargs
    ) -> PreprocessOutput:
        """
        Preprocess following the reference generate() implementation.

        Structure from reference:
        1. Role tokens (3): text_projection(text_embedding(input_id[:, :3]))
        2. Codec prefix mixed with tts_pad/tts_bos
        3. First text token: text_projection(text_embedding()) + codec_embedding(codec_bos)

        Args:
            prompt: Input text prompt
            audio_path: Path to input audio file (not supported)
            **kwargs: Additional parameters (language, etc.)

        Returns:
            PreprocessOutput with input_tokens [seq_len, n_codebooks] and input_masks
        """
        assert audio_path is None

        language = kwargs.get("language", "auto")

        # Tokenize prompt
        input_id = self.text_tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Language setup
        language_id = None
        if language.lower() != "auto":
            if language.lower() in self.config.talker_config.codec_language_id:
                language_id = self.config.talker_config.codec_language_id[language.lower()]

        # Codec prefix tokens
        if language_id is None:
            codec_prefix = [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
            ]
        else:
            codec_prefix = [
                self.config.talker_config.codec_think_id,
                self.config.talker_config.codec_think_bos_id,
                language_id,
                self.config.talker_config.codec_think_eos_id,
            ]

        # Calculate sequence length
        seq_len = (
            3 +                  # role tokens
            len(codec_prefix) +  # ALL codec prefix tokens with tts_pad
            1 +                  # codec_pad with tts_bos
            1                    # first text token with codec_bos
        )

        # Initialize tensors [seq_len, n_codebooks]
        input_tokens = torch.zeros(seq_len, self.n_codebooks, dtype=torch.long, device=self.device)
        input_masks = torch.zeros(seq_len, self.n_codebooks, dtype=torch.bool, device=self.device)

        pos = 0

        # 1. Role tokens (3): text only
        for i in range(3):
            input_tokens[pos, 0] = input_id[0, i]
            # input_masks[pos, 0] = False (default, text only)
            pos += 1

        # 2. ALL codec prefix tokens with tts_pad: text + codec
        for codec_tok in codec_prefix:
            input_tokens[pos, 0] = self.config.tts_pad_token_id
            input_tokens[pos, 1] = codec_tok
            input_masks[pos, 0] = True  # needs codec addition
            pos += 1

        # 3. Codec pad token with tts_bos: text + codec
        input_tokens[pos, 0] = self.config.tts_bos_token_id
        input_tokens[pos, 1] = self.config.talker_config.codec_pad_id
        input_masks[pos, 0] = True  # needs codec addition
        pos += 1

        # 4. First text token: text + codec
        input_tokens[pos, 0] = input_id[0, 3]
        input_tokens[pos, 1] = self.config.talker_config.codec_bos_id
        input_masks[pos, 0] = True  # needs codec addition
        pos += 1

        # Create repetition cache
        repetition_cache = None
        config = self.default_sampling_config
        if (config.repetition_penalty is not None and
            config.repetition_window is not None and
            config.repetition_penalty != 1.0):
            repetition_cache = torch.zeros(
                config.repetition_window if config.repetition_window > 0 else 1,
                self.n_codebooks,
                self.vocab_size,
                dtype=torch.bool,
                device=self.device,
            )

        return PreprocessOutput(
            input_tokens=input_tokens,
            input_masks=input_masks,
            repetition_cache=repetition_cache
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor = None,
        input_features: torch.Tensor = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs. Shape: (batch_size, n_codebooks)
            position_ids: Position IDs for the tokens. Shape: (batch_size)
            attn_wrapper: FlashInfer attention wrapper
            kv_cache: KV cache tensor
            input_masks: Token type masks. Shape: (batch_size, n_codebooks)
                [:, 0] = False: text only (text_embedding + text_projection)
                [:, 0] = True: text + codec (text_projection(text_embedding(col0)) + codec_embedding(col1))
            input_features: Unused for now
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (logits, hidden_states)
        """
        # Prefill phase: construct embeddings based on input_ids and masks
        # Text embeddings from column 0 (role tokens, tts_pad, tts_bos, text tokens)
        text_embeds = self.model.talker.text_projection(
            self.model.talker.model.text_embedding(input_ids[:, 0])
        )  # [batch_size, hidden_size]

        # Codec embeddings from column 1
        codec_embeds = self.model.talker.model.codec_embedding(input_ids[:, 1])  # [batch_size, hidden_size]

        # Combine based on mask (CUDA graph compatible)
        # mask[:, 0] = False: text only
        # mask[:, 0] = True: text + codec
        needs_codec = input_masks[:, 0].unsqueeze(-1)  # [batch_size, 1]
        inputs_embeds = torch.where(needs_codec, text_embeds + codec_embeds, text_embeds)

        logits, backbone_last_hidden = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return logits[:, None, :], backbone_last_hidden

    def sampling(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> tuple:
        """
        Sampling and other model-specific logic for generating output tokens.

        Args:
            logits: Output logits from the model. Shape: (batch_size, n_codebooks, vocab_size)
            requests: List of Request objects containing sampling configurations
            sampling_params: Optional common sampling configurations
            repetition_cache: Optional tensor for repetition penalty
            cfg_scale: Optional classifier-free guidance scale
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple containing:
                - Output token IDs from sampling. Shape: (batch_size, n_codebooks)
                - Coroutine for request state update
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        assert logits.shape[1] == 1, "Logits should have shape [bs, 1, vocab_size]"

        if repetition_cache is not None:
            logits = Sampler.apply_repetition_penalty(
                logits, repetition_cache, sampling_params.repetition_penalty
            )

        # there are 33 codebooks (32 audio + 1 text), but the output from backbone transformer is single codebook
        # so here we allocate output_ids for all codebooks but do sampling only for the first one
        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=sampling_params)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])
        output_ids = output_ids.repeat(1, self.n_codebooks)  # [bs, 33]

        if repetition_cache is not None:
            Sampler.update_repetition_penalty_cache(
                repetition_cache,
                output_ids,
                sampling_params.repetition_window,
            )

        c0_embed = self.model.talker.model.codec_embedding(output_ids[:, 0])
        hidden_for_depth = torch.cat([hidden_states[:, None, :], c0_embed[:, None, :]], dim=1) # (bs, 2, hidden_size)

        for i, req in enumerate(requests):
            req.input_tokens = torch.zeros(1, self.n_codebooks, dtype=torch.long, device=self.device)
            req.input_tokens[0, 0] = output_ids[i, 0].item()

            req.input_masks = torch.ones(self.n_codebooks, dtype=torch.bool, device=self.device)[None, :]
            req.input_masks[:, -1] = False  # only the audio streams are used in decode phase

            # no additional logic for CSM model
            req.lm_output_tokens.append(output_ids[i : i + 1])
            if not self.is_stop_id(output_ids[i].tolist()):
                # Don't add the EOS token to lm_output_audio_tokens
                req.lm_output_audio_tokens.append(output_ids[i : i + 1])
            elif req.next_position_id > self.max_tokens:
                req.done_lm_generation = True
                req.finish_reason = "max_tokens_reached"
            else:
                req.done_lm_generation = True
                req.finish_reason = "stop_id_encountered"

        return output_ids, hidden_for_depth

    def depth_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        depth_logits = self.model.forward_depth(
            inputs_embeds=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return depth_logits
    
    def depth_sampling(
        self,
        logits: torch.Tensor,
        i_iteration: int,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        output_ids = Sampler.run_sampling(logits, config=sampling_params)
        ci_embed = self.model.talker.code_predictor.model.codec_embedding[i_iteration](output_ids)

        for i, req in enumerate(requests):
            token_id = output_ids[i].item()
            req.input_tokens[0, i_iteration] = token_id
            req.lm_output_tokens[-1][0, i_iteration] = token_id
            req.lm_output_audio_tokens[-1][0, i_iteration] = token_id

        return output_ids, ci_embed

    def postprocess(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert model output tokens to audio waveform.

        Args:
            token_ids: Token IDs generated by the model. Shape: (batch_size, interval, n_codebooks)
            **kwargs: Additional model-specific parameters

        Returns:
            Tensor of audio data. Shape: (batch_size, n_channels, audio_length)
        """
        # token_ids: (batch_size, interval, 17)
        # there are 17 codebooks including text
        tokens_to_process = token_ids[:, :, :-1].transpose(1, 2)  # (batch_size, 16, interval)

        # Clamp tokens to valid Mimi decoder range [0, 2047]
        # Mimi cardinality is 2048, so valid tokens are in [0, 2047]
        tokens_to_process = tokens_to_process.clamp(0, 2047)

        # mimi decoder
        # TODO: caching for mimi
        audio_tensor = self.audio_decoder.decode(tokens_to_process)
        # audio_tensor: (batch_size, 1, N)

        return audio_tensor
    