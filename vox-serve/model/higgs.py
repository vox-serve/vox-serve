import torch
import torch.nn as nn
import math
import numpy as np
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from safetensors.torch import load_file
from typing import Optional, Tuple, Union, List, Dict, Any
import gc
from transformers.generation import GenerationMixin
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import ACT2FN

from ..model.base import BaseLM, Request, PreprocessOutput, SamplingConfig
from ..sampling import Sampler
from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..tokenizer.higgs import load_higgs_audio_tokenizer
from ..encoder.higgs import HiggsAudioEncoderConfig

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
@dataclass
class HiggsAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class for the HiggsAudioModel.

    Args:
        text_config (`Union[AutoConfig, dict]`):
            The config object or dictionary of the text backbone.
        audio_encoder_config (`Union[AutoConfig, dict]`):
            The config object or dictionary of the whisper encoder.
            The audio encoder will be bidirectional and will be only available for audio understanding.
        audio_tokenizer_config
            The config object or dictionary of the audio tokenizer.
        audio_adapter_type
            The type of audio adapter to use. We support two types of adapter:
            - stack:
                We stack additional Transformer layers after the main LLM backbone for audio generation.
            - dual_ffn:
                For selected part of the LLM backbone, we replace the text FFN with a dual FFN architecture
                that contains an additional audio FFN. The audio FFN will be triggered when the location is marked for audio tokens.
            - dual_ffn_fast_forward:
                We pick a few layers in the LLM backbone to plug-in the audio FFN. For the remaining layers,
                the audio hidden states will be directly fast-forward to the next layer.
                This reduces the computational cost for audio generation.
        audio_embed_avg (`bool`, *optional*, defaults to False):
            Whether to average the audio embeddings before sending them to the text attention layer.
        audio_ffn_hidden_size
            The hidden size of the audio feedforward network in dual-path FFN
        audio_ffn_intermediate_size
            The intermediate size of the audio feedforward network in dual-path FFN
        audio_dual_ffn_layers
            The layers in the LLM backbone to plug-in the dual FFN layer (mixture of audio FFN and text FFN).
        audio_decoder_proj_num_attention (`int`, *optional*, defaults to 0):
            The number of attention heads in the audio decoder projection layer.
        use_delay_pattern (`bool`, *optional*, defaults to False):
            Whether to use delay pattern in the audio decoder.
        skip_audio_tower (`bool`, *optional*, defaults to False):
            Whether to skip the audio tower in the audio encoder.
        use_audio_out_embed_projector (`bool`, *optional*, defaults to False):
            Whether to use an embedding projector to map audio out embeddings.
        use_audio_out_self_attention (`bool`, *optional*, defaults to False):
            Whether to use self-attention to aggregate information from audio-tokens before sending to the text attention layer.
        audio_num_codebooks (`int`, *optional*, defaults to 12):
            The number of codebooks in RVQGAN.
        audio_codebook_size (`int`, *optional*, defaults to 1024):
            The size of each codebook in RVQGAN.
        audio_stream_bos_id
            The id of the bos in the audio stream
        audio_stream_eos_id
            The id of the eos in the audio stream
        audio_bos_token (`str`, *optional*, defaults to "<|audio_bos|>"):
            The special `<|audio_bos|>` token. In Higgs-Audio, it is mapped to 128011,
            which is the index of `<|reserved_special_token_3|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_eos_token (`str`, *optional*, defaults to "<|audio_eos|>"):
            The special `<|audio_eos|>` token. We use 128012 as the default value,
            which is the index of `<|reserved_special_token_4|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_out_bos_token (`str`, *optional*, defaults to "<|audio_out_bos|>"):
            The special `<|audio_out_bos|>` token. We use 128013 as the default value,
            which is the index of `<|reserved_special_token_5|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_token (`str`, *optional*, defaults to "<|AUDIO|>"):
            The special `<|AUDIO|>` token. We use 128015 as the default value,
            which is the index of `<|reserved_special_token_7|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with whisper features.
        audio_out_token (`str`, *optional*, defaults to "<|AUDIO_OUT|>"):
            The special `<|AUDIO_OUT|>` token. We use 128016 as the default value,
            which is the index of `<|reserved_special_token_8|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with audio tokens extracted via audio tokenizer.
    """

    model_type = "higgs_audio"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        audio_encoder_config=None,
        audio_tokenizer_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_ffn_hidden_size=4096,
        audio_ffn_intermediate_size=14336,
        audio_dual_ffn_layers=None,
        audio_decoder_proj_num_layers=0,
        encode_whisper_embed=True,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        skip_audio_tower=False,
        use_audio_out_embed_projector=False,
        use_audio_out_self_attention=False,
        use_rq_transformer=False,
        rq_transformer_hidden_size=None,
        rq_transformer_intermediate_size=None,
        rq_transformer_num_attention_heads=None,
        rq_transformer_num_key_value_heads=None,
        rq_transformer_num_hidden_layers=3,
        audio_num_codebooks=12,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        pad_token_id=128001,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        **kwargs,
    ):
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config["model_type"] = (
                audio_encoder_config["model_type"] if "model_type" in audio_encoder_config else "higgs_audio_encoder"
            )
            audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        assert audio_adapter_type in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ], f"Invalid audio adapter type: {audio_adapter_type}"
        if audio_adapter_type.startswith("dual_ffn"):
            assert audio_dual_ffn_layers is not None, (
                "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
            )
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.encode_whisper_embed = encode_whisper_embed
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        self.use_rq_transformer = use_rq_transformer

        if self.use_rq_transformer:
            assert not self.use_delay_pattern, "Delay pattern is not supported if you turned on RQ-Transformer!"
        self.rq_transformer_hidden_size = rq_transformer_hidden_size
        self.rq_transformer_intermediate_size = rq_transformer_intermediate_size
        self.rq_transformer_num_attention_heads = rq_transformer_num_attention_heads
        self.rq_transformer_num_key_value_heads = rq_transformer_num_key_value_heads
        self.rq_transformer_num_hidden_layers = rq_transformer_num_hidden_layers

        if use_rq_transformer:
            # For RQ-Transformer, we set the hidden_size to the same as the text model's hidden size if it is not specified.
            if self.rq_transformer_hidden_size is None:
                self.rq_transformer_hidden_size = text_config.hidden_size
            assert self.rq_transformer_hidden_size % 128 == 0
            if self.rq_transformer_intermediate_size is None:
                self.rq_transformer_intermediate_size = text_config.intermediate_size
            if self.rq_transformer_num_attention_heads is None:
                self.rq_transformer_num_attention_heads = self.rq_transformer_hidden_size // 128
            if self.rq_transformer_num_key_value_heads is None:
                self.rq_transformer_num_key_value_heads = self.rq_transformer_hidden_size // 128 // 4
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_attention_heads == 0
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_key_value_heads == 0

        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        
        
class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.audio_encoder_config.init_std

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
# - (Keeping encoder here to avoid circular import)
class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # Flash Attention 2 does not support zero shape tensor, so we have to use sdpa implementation for the Whisper component.
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if check_seq_length and (input_features.shape[-1] != expected_seq_length):
            raise ValueError(
                f"HiggsAudio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        # If the sequence length after average pooling is not divisible by the sequence parallel size, we would duplicate it across the sequence parallel ranks.
        # In this case, gradients need to be scaled up because the subsequent scaling up in the function _apply_audio_tower is skipped.
        hidden_states = self.avg_pooler(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths
    

class HiggsMLP(nn.Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.text_config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[config.text_config.hidden_act]
        
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

class HiggsAttention(nn.Module):
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.text_config.hidden_size // config.text_config.num_attention_heads)
        self.num_key_value_groups = config.text_config.num_attention_heads // config.text_config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        # self.rope_scale = config.rope_scaling.get("factor", 32.0)
        # self.rope_theta = config.rope_theta
        # self.low_freq_factor = config.rope_scaling.get("low_freq_factor", 1.0)
        # self.high_freq_factor = config.rope_scaling.get("high_freq_factor", 4.0)
        # self.old_context_len = config.rope_scaling.get("original_max_position_embeddings", 8192)

        self.q_proj = nn.Linear(
            config.hidden_size, config.text_config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(
            config.hidden_size, config.text_config.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(
            config.hidden_size, config.text_config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(
            config.text_config.num_attention_heads * self.head_dim, config.text_config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        key_states = self.k_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)

        # # reshape. squash to 3d
        # b, s, h, d = query_states.shape
        # query_states = query_states.reshape(b * s, h, d)
        # b_k, s_k, h_k, d_k = key_states.shape
        # key_states = key_states.reshape(b_k * s_k, h_k, d_k)
        # value_states = value_states.reshape(b_k * s_k, h_k, d_k)

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            # rope_scale=self.rope_scale,
            # rope_theta=self.rope_theta,
            # low_freq_factor=self.low_freq_factor,
            # high_freq_factor=self.high_freq_factor,
            # old_context_len=self.old_context_len,
        )
        
        # query_states = query_states.reshape(b, s, h, d)
        # key_states = key_states.reshape(b_k, s_k, h_k, d_k)
        # value_states = value_states.reshape(b_k, s_k, h_k, d_k)

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    

class HiggsDecoderLayer(nn.Module):
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size

        self.self_attn = HiggsAttention(config=config, layer_idx=layer_idx)
        self.mlp = HiggsMLP(config=config)

        self.input_layernorm = LlamaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

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
 
class HiggsTransformer(nn.Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        # TODO: Pick audio attention model
        # #1. Support Dual FFN:
        # #2. Support Dual FFN with fast forward
        # #3. Stack: (Usual decoder layers)
        # elif config.audio_adapter_type == "stack":
        
        self.layers = nn.ModuleList([HiggsDecoderLayer(config, layer_idx) for layer_idx in range(config.text_config.num_hidden_layers)])
        self.final_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.text_config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class HiggsEmbedding(nn.Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()

        self.hidden_size = config.text_config.hidden_size
        self.padding_idx = config.pad_token_id

        self.word_embeddings = nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size, self.padding_idx)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class HiggsBackboneModel(nn.Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embedding = HiggsEmbedding(config)
        self.encoder = HiggsTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = inputs_embeds

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        logits = self.output_layer(hidden_states)

        return logits


class HiggsForCausalLM(nn.Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.transformer = HiggsBackboneModel(config)
        self.vocab_size = config.text_config.vocab_size

    def embed_tokens(self, input_ids):
        return self.transformer.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        logits = self.transformer(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        return logits

class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
     
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = config.audio_out_bos_token_id if "audio_out_bos_token_id" in config else None
        self.audio_eos_token_id = config.audio_eos_token_id if "audio_eos_token_id" in config else None
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        self.model = HiggsForCausalLM(config)

    def embed_tokens(self, input_ids):
        return self.model.embed_tokens(input_ids)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return logits   
    
class HiggsModel(BaseLM):
    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        device="cuda:0",
        tokenizer_path: Optional[str] = None,
        enable_torch_compile=False,
    ):
        if model_name == "higgs":
            model_name = MODEL_PATH
        super().__init__(model_name, device, dtype, enable_torch_compile)
        self.model_name = model_name
        
        self.model = HiggsAudioModel.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)

        # use provided tokenizer path or default to model_name
        if tokenizer_path is None:
            tokenizer_path = model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)
        self.audio_tokenizer = load_higgs_audio_tokenizer(AUDIO_TOKENIZER_PATH, device=device)

        cfg = self.model.config
        self.model.config.audio_adapter_type = cfg.audio_adapter_type
        self._num_attention_heads = cfg.text_config.num_attention_heads
        self._num_key_value_heads = cfg.text_config.num_key_value_heads
        self._num_hidden_layers = cfg.text_config.num_hidden_layers
        self._hidden_size = cfg.text_config.hidden_size

        self.stop_token_id = 128258

        self.default_sampling_config = SamplingConfig(
            top_k=None,
            top_p=0.8,
            min_p=None,
            temperature=0.6,
            repetition_penalty=1.3,
            repetition_window=-1,
            cfg_scale=None,
        )
        # for cuda graph-ing of postprocess
        self.idx_14 = torch.tensor([1, 4], dtype=torch.long, device="cuda")
        self.idx_2356 = torch.tensor([2, 3, 5, 6], dtype=torch.long, device="cuda")

    @property
    def n_codebooks(self):
        """Number of codebooks in the model."""
        return 1

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
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 28

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 21

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 1200

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return 2048  # Based on slice [2048:4096] in postprocess

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.model.config.text_config.vocab_size

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.stop_token_id

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _validate_voice(self, voice):
        """Validate if the given voice is supported by the model."""
        if voice and voice not in self.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")

    def _orpheus_format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        elif voice:
            adapted_prompt = f"{voice}: {prompt}"
            prompt_tokens = self.text_tokenizer(adapted_prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            # prompt_string = self.text_tokenizer.decode(all_input_ids[0])
            return all_input_ids
        else:
            prompt_tokens = self.text_tokenizer(prompt, return_tensors="pt")
            start_token = torch.tensor([[128259]], dtype=torch.int64)
            end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
            all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
            # prompt_string = self.text_tokenizer.decode(all_input_ids[0])
            return prompt_tokens

    def preprocess(
        self,
        prompt: str = None,
        audio_path: str = None,
        voice="tara",
        model_type="larger",
    ) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        assert audio_path is None
        self._validate_voice(voice)
        input_ids = self._orpheus_format_prompt(prompt, voice, model_type)
        input_ids = input_ids.view(-1, 1)  # add codebook dimension

        # Create repetition cache if repetition penalty is enabled
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

        return PreprocessOutput(input_tokens=input_ids, repetition_cache=repetition_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        
        # remove codebook dimension
        inputs_embeds = self.model.embed_tokens(input_ids[:, 0])

        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        return logits[:, None, :]  # add codebook dimension

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        if repetition_cache is not None:
            logits = Sampler.apply_repetition_penalty(
                logits, repetition_cache, sampling_params.repetition_penalty
            )

        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=sampling_params)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        if repetition_cache is not None:
            Sampler.update_repetition_penalty_cache(
                repetition_cache,
                output_ids,
                sampling_params.repetition_window,
            )

        for i, req in enumerate(requests):
            req.input_tokens = output_ids[i : i + 1]

        async def update_req_states():
            stop_mask = output_ids[:, 0] == self.stop_token_id
            stop_indices = torch.nonzero(stop_mask, as_tuple=True)[0]

            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                req.lm_output_audio_tokens.append(output_ids[i : i + 1])

            # Remove from stop requests
            for idx in stop_indices:
                req = requests[idx.item()]
                # Remove the EOS token from lm_output_audio_tokens
                req.lm_output_audio_tokens.pop()
                req.done_lm_generation = True
                req.finish_reason = "stop_id_encountered"

            for req in requests:
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

            if repetition_cache is not None:
                # Update repetition cache in requests
                for i, req in enumerate(requests):
                    req.repetition_cache = repetition_cache[i]

        task = update_req_states()

        return output_ids, task

    def _turn_token_into_id(self, output_ids):
        """Modoel's output ids to audio ids"""
        return (output_ids - 128256 - 10) % 4096

    def postprocess(self, token_ids: torch.Tensor):
        wv_list = []
        print("Postprocessing token ids shape:", token_ids[1].shape)
        for output_audio in token_ids[1]:
            vq_code = revert_delay_pattern(output_audio).clip(0, self.model.audio_codebook_size - 1)[:, 1:-1]
            wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
            wv_list.append(wv_numpy)
        wv_numpy = np.concatenate(wv_list)
        
        return wv_numpy

def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)
