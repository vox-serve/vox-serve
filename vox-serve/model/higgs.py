import torch
import torchaudio
import torch.nn as nn
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from transformers.generation import GenerationMixin
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, ACT2FN

from ..model.base import BaseLM, Request, PreprocessOutput, SamplingConfig
from ..sampling import Sampler
from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids
from ..tokenizer.higgs import load_higgs_audio_tokenizer
from ..encoder.higgs import HiggsAudioEncoderConfig

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


@dataclass
class HiggsAudioConfig(PretrainedConfig):
    """
    Configuration class for HiggsAudioModel.
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
        # initialize audio encoder config
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config["model_type"] = (
                audio_encoder_config.get("model_type", "higgs_audio_encoder")
            )
            audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()

        # initialize text config
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        # validate adapter type
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
            assert not self.use_delay_pattern, "Delay pattern is not supported with RQ-Transformer!"
            
        self.rq_transformer_hidden_size = rq_transformer_hidden_size
        self.rq_transformer_intermediate_size = rq_transformer_intermediate_size
        self.rq_transformer_num_attention_heads = rq_transformer_num_attention_heads
        self.rq_transformer_num_key_value_heads = rq_transformer_num_key_value_heads
        self.rq_transformer_num_hidden_layers = rq_transformer_num_hidden_layers

        if use_rq_transformer:
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
        
        # (add hidden_size at config level for consistency)
        self.hidden_size = text_config.hidden_size


class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.init_std 
            if hasattr(self.config, "init_std") 
            else self.config.audio_encoder_config.init_std
        )

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """
    Transformer encoder for audio features.
    Based on WhisperEncoder architecture.
    """
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

        # convolutional layers for downsampling
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # position embeddings
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # transformer layers
        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        check_seq_length: bool = True,
    ) -> BaseModelOutput:
        """
        Args:
            input_features: [batch_size, feature_size, sequence_length] mel features
        """
        expected_seq_length = (
            self.max_source_positions * 
            self.conv1.stride[0] * 
            self.conv2.stride[0]
        )
        
        if check_seq_length and (input_features.shape[-1] != expected_seq_length):
            raise ValueError(
                f"Expected mel features length {expected_seq_length}, "
                f"but got {input_features.shape[-1]}"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # convert to correct dtype and device
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, 
            device=self.conv1.weight.device
        )

        # convolutional encoding
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # add position embeddings
        embed_pos = self.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Transformer layers
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # layerDrop
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

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

        # average pooling
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        # final layer norm
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions
        )


class HiggsMLP(nn.Module):
    """Standard FFN with SwiGLU activation."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.text_config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.text_config.hidden_act]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class HiggsAttention(nn.Module):
    """Grouped Query Attention with RoPE."""
    
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_key_value_heads = config.text_config.num_key_value_heads
        
        self.head_dim = getattr(
            config, 
            "head_dim", 
            self.hidden_size // self.num_attention_heads
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.LongTensor,
            attn_wrapper: FlashInferWrapper,
            kv_cache: torch.Tensor,
        ):
        
        # original shapes
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # project to Q, K, V
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # apply RoPE
        query_states, key_states_flat = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
        )

        # update KV cache
        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output


class HiggsDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = HiggsAttention(config=config, layer_idx=layer_idx)
        self.mlp = HiggsMLP(config=config)

        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, 
            eps=config.text_config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, 
            eps=config.text_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        # self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HiggsTransformer(nn.Module):
    """Stack of decoder layers."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.layers = nn.ModuleList([
            HiggsDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.text_config.num_hidden_layers)
        ])
        self.final_layernorm = LlamaRMSNorm(
            config.hidden_size, 
            eps=config.text_config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
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
    """Token embedding layer."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id

        self.word_embeddings = nn.Embedding(
            config.text_config.vocab_size,
            config.hidden_size,
            self.padding_idx
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class HiggsBackboneModel(nn.Module):
    """Main backbone model."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        self.embedding = HiggsEmbedding(config)
        self.encoder = HiggsTransformer(config)
        self.output_layer = nn.Linear(
            config.hidden_size, 
            config.text_config.vocab_size, 
            bias=False
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        logits = self.output_layer(hidden_states)
        return logits


class HiggsForCausalLM(nn.Module):
    """Causal LM wrapper."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.transformer = HiggsBackboneModel(config)
        self.vocab_size = config.text_config.vocab_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )


class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    """Main Higgs Audio model for generation."""
    
    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
     
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = config.audio_out_bos_token_id
        self.audio_eos_token_id = config.audio_eos_token_id
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        self.model = HiggsForCausalLM(config)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )


class HiggsModel(BaseLM):
    """
    High-level interface for Higgs Audio model.
    """
    
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        tokenizer_path: Optional[str] = None,
        enable_torch_compile: bool = False,
    ):
        if model_name == "higgs":
            model_name = MODEL_PATH
            
        super().__init__(model_name, device, dtype, enable_torch_compile)
        self.model_name = model_name
        
        # load model
        self.model = HiggsAudioModel.from_pretrained(
            MODEL_PATH, 
            torch_dtype=dtype
        ).to(device)

        # load tokenizers
        tokenizer_path = tokenizer_path or model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            AUDIO_TOKENIZER_PATH, 
            device=device
        )
        
        # audio resampling
        self.resample_44k_to_24k = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=24000
        ).to(self.device)
        
        # config shortcuts
        cfg = self.model.config
        self.audio_adapter_type = cfg.audio_adapter_type
        self._num_attention_heads = cfg.text_config.num_attention_heads
        self._num_key_value_heads = cfg.text_config.num_key_value_heads
        self._num_hidden_layers = cfg.text_config.num_hidden_layers
        self._hidden_size = cfg.hidden_size
        self.audio_num_codebooks = cfg.audio_num_codebooks
        self.stop_token_id = 128258

        # default sampling config
        self.default_sampling_config = SamplingConfig(
            top_k=None,
            top_p=0.95,
            min_p=None,
            temperature=0.7,
            repetition_penalty=1.3,
            repetition_window=-1,
            cfg_scale=None,
        )

    @property
    def n_codebooks(self) -> int:
        return self.audio_num_codebooks

    @property
    def num_attention_heads(self) -> int:
        return self._num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self._num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        return self._num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def detokenize_interval(self) -> int:
        return 28

    @property
    def detokenize_overlap(self) -> int:
        return 21

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the model generates in a single request."""
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
        return 2048

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.model.config.text_config.vocab_size

    def is_stop_id(self, token_ids: List[int]) -> bool:
        """Check if token is a stop token."""
        return token_ids[0] == self.stop_token_id

    def _load_tokenizer(self, tokenizer_path: str) -> AutoTokenizer:
        """Load tokenizer from local path or HuggingFace hub."""
        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _validate_voice(self, voice: str) -> None:
        """Validate if the given voice is supported by the model."""
        if hasattr(self, 'available_voices') and voice and voice not in self.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")

    def _format_higgs_prompt(self, prompt: str) -> torch.Tensor:
        """
        Format prompt for Higgs Audio generation.
        """
        chat_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        input_ids = self.text_tokenizer(chat_prompt, return_tensors="pt").input_ids
        
        # add Higgs audio generation tokens
        audio_out_bos = torch.tensor([[128013]], dtype=torch.long)
        audio_out_token = torch.tensor([[128016]], dtype=torch.long)
        
        full_input_ids = torch.cat([input_ids, audio_out_bos, audio_out_token], dim=1)
        return full_input_ids

    def preprocess(
        self,
        prompt: str = None,
        audio_path: str = None,
        voice: str = "tara",
        model_type: str = "larger",
    ) -> PreprocessOutput:
        """
        Prepare the prompt for the model.
        """
        assert audio_path is None, "Audio input not yet supported"
        assert prompt is not None, "Prompt must be provided"
        
        self._validate_voice(voice)
        
        # format higgs prompt and get input_ids
        input_ids = self._format_higgs_prompt(prompt)
        
        # move to device
        input_ids = input_ids.to(self.device)
        
        # validate shape
        assert input_ids.shape[0] == 1, (
            "Currently HiggsAudioModel.generate() only supports batch_size=1."
        )
        
        # add codebook dimension: [batch, seq_len] -> [batch, seq_len, n_codebooks]
        input_ids = input_ids.unsqueeze(-1)  # [1, seq_len, 1]

        # create repetition cache if repetition penalty is enabled
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
            input_tokens=input_ids, 
            repetition_cache=repetition_cache
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        """

        # embed tokens
        inputs_embeds = self.model.embed_tokens(input_ids[:, 0])
        
        # forward through model
        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        # add codebook dimension: [batch, vocab_size] -> [batch, 12, vocab_size]
        return logits[:, None, :]
    

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: Optional[SamplingConfig] = None,
        repetition_cache: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Sample next tokens from logits.
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        # apply repetition penalty
        if repetition_cache is not None:
            logits = Sampler.apply_repetition_penalty(
                logits, repetition_cache, sampling_params.repetition_penalty
            )

        # sample tokens
        output_ids = Sampler.run_sampling(
            logits.view(-1, self.vocab_size), 
            config=sampling_params
        )
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        # update repetition cache
        if repetition_cache is not None:
            Sampler.update_repetition_penalty_cache(
                repetition_cache,
                output_ids,
                sampling_params.repetition_window,
            )

        # update request input tokens
        for i, req in enumerate(requests):
            req.input_tokens = output_ids[i : i + 1]

        async def update_req_states():
            """Update request states and check for completion."""
            stop_mask = output_ids[:, 0] == self.stop_token_id
            stop_indices = torch.nonzero(stop_mask, as_tuple=True)[0]

            # append generated tokens to requests
            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                req.lm_output_audio_tokens.append(output_ids[i : i + 1])

            # mark stopped requests
            for idx in stop_indices:
                req = requests[idx.item()]
                # remove the EOS token from audio tokens
                req.lm_output_audio_tokens.pop()
                req.done_lm_generation = True
                req.finish_reason = "stop_id_encountered"

            # check max tokens
            for req in requests:
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

            # update repetition cache in requests
            if repetition_cache is not None:
                for i, req in enumerate(requests):
                    req.repetition_cache = repetition_cache[i]

        task = update_req_states()
        return output_ids, task

    def postprocess(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        Convert token IDs to audio waveform.
        
        Args:
            token_ids: [batch, time, n_codebooks] ?
        
        Returns:
            Audio waveform as numpy array
        """
        
        batch_size, _, _ = token_ids.shape
        
        # process each batch item
        wv_list = []
        for batch_idx in range(batch_size):
            output_audio = token_ids[batch_idx] 
            
            # convert model output IDs to audio token IDs
            audio_ids = self._turn_token_into_id(output_audio)
            
            # revert delay pattern if used
            if self.model.config.use_delay_pattern:
                # transpose for delay pattern: [time, 12] -> [12, time]
                audio_ids = audio_ids.t()
                vq_code = revert_delay_pattern(audio_ids)
            else:
                vq_code = audio_ids.t()
            
            # clip to valid codebook range and remove boundary tokens
            vq_code = vq_code.clip(0, self.audio_num_codebooks - 1)[:, 1:-1]
            
            
            # decode only during inference
            with torch.inference_mode():
                try:
                    wv = self.audio_tokenizer.decode(
                        vq_code.unsqueeze(0)
                    )[0, 0]
                    wv_list.append(wv)
                except Exception as e:
                    print(f"Failed to decode audio for batch {batch_idx}: {e}")
                    continue
        
        if wv_list:
            wv_tensor = torch.cat(wv_list, dim=0)
        else:
            wv_tensor = torch.empty(0, device=token_ids.device)

        return wv_tensor

    def _turn_token_into_id(self, output_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert model's output token IDs to audio codebook IDs.
        
        Model outputs are in range [128256+10, 128256+10+4096]
        Audio IDs should be in range [0, 4096]
        """
        return (output_ids - 128256 - 10) % 4096


def revert_delay_pattern(data: torch.Tensor) -> torch.Tensor:
    """
    Convert samples encoded with delay pattern back to the original form.
    """
    if data.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor [num_codebooks, seq_len], got shape {data.shape}"
        )
    
    num_codebooks = data.shape[0]
    seq_len = data.shape[1] - num_codebooks + 1
    
    if seq_len <= 0:
        raise ValueError(
            f"Invalid sequence length after delay pattern removal: {seq_len}"
        )
    
    out_list = []
    for i in range(num_codebooks):
        # extract the valid range for this codebook
        start_idx = i
        end_idx = seq_len + i
        out_list.append(data[i:i+1, start_idx:end_idx])
    
    return torch.cat(out_list, dim=0)