from typing import List, Dict, Any
import json
from dataclasses import dataclass, field
# from hyperpyyaml import load_hyperpyyaml

import flashinfer
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

from ..flashinfer_utils import FlashInferWrapper
from ..sampling import SamplingConfig, Sampler
from ..requests import Request
from ..utils import load_hf_safetensor_state_dict
from ..tokenizer.glm_decoder import GLMAudioDecoder
from .base import BaseLM, PreprocessOutput

@dataclass
class GLMVoiceConfig:
    # _name_or_path: str = "THUDM/glm-4-voice-9b"
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    # apply_query_key_layer_scaling: bool = True
    # apply_residual_connection_post_layernorm: bool = False
    # attention_dropout: float = 0.0
    # attention_softmax_in_fp32: bool = True
    # bias_dropout_fusion: bool = True
    # classifier_dropout: Any = None
    eos_token_id: List[int] = field(default_factory=lambda: [151329, 151336, 151338])
    ffn_hidden_size: int = 13696
    # fp32_residual_connection: bool = False
    # hidden_dropout: float = 0.0
    hidden_size: int = 4096
    # kv_channels: int = 128
    layernorm_epsilon: float = 3.90625e-08
    # model_type: str = "chatglm"
    # multi_query_attention: bool = True
    multi_query_group_num: int = 2
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_layers: int = 40
    # original_rope: bool = True
    pad_token_id: int = 151329
    padded_vocab_size: int = 168960
    # post_layer_norm: bool = True
    rmsnorm: bool = True
    rope_ratio: int = 1
    torch_dtype: str = "bfloat16"
    # transformers_version: str = "4.44.1"
    # use_cache: bool = True
    vocab_size: int = 168960

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GLMVoiceConfig':
        # Get field names from the dataclass
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        # Filter config_dict to only include known fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


class GLMVoiceRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GLMVoiceRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GLMVoiceMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.ffn_hidden_size
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.add_bias_linear)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)
        self.act_fn = self.swiglu

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]
    
    def forward(self, x):
        output = self.dense_4h_to_h(self.act_fn(self.dense_h_to_4h(x)))
        return output


class GLMVoiceAttention(nn.Module):
    def __init__(self, config: GLMVoiceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_scale = config.rope_ratio
        self.rope_theta = 10_000

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.multi_query_group_num
        self.qkv_hidden_size = config.hidden_size + 2 * self.head_dim * config.multi_query_group_num
        
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size, bias=config.add_bias_linear or config.add_qkv_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.add_bias_linear)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        mixed_x_layer = self.query_key_value(hidden_states)

        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_q_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        query_states = query_layer.view(hidden_shape) #.transpose(0, 1)
        key_states = key_layer.view(hidden_shape) #.transpose(0, 1)
        value_states = value_layer.view(hidden_shape) #.transpose(0, 1)

        query_states, key_states = flashinfer.rope.apply_rope_pos_ids(
            query_states, key_states, 
            pos_ids=position_ids, 
            rotary_dim=self.head_dim // 2,
            interleave=True,
            rope_scale=self.rope_scale, 
            rope_theta=self.rope_theta, 
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output


class GLMVoiceDecoderLayer(nn.Module):
    def __init__(self, config: GLMVoiceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attention = GLMVoiceAttention(config=config, layer_idx=layer_idx)
        self.mlp = GLMVoiceMLP(config)

        self.input_layernorm = GLMVoiceRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = GLMVoiceRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

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
        hidden_states = self.self_attention(
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

class GLMVoiceTransformer(nn.Module):
    def __init__(self, config: GLMVoiceConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [GLMVoiceDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.final_layernorm = GLMVoiceRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

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


class GLMVoiceEmbedding(nn.Module):
    def __init__(self, config: GLMVoiceConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.padded_vocab_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class GLMVoiceBackboneModel(nn.Module):
    def __init__(self, config: GLMVoiceConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = GLMVoiceEmbedding(config)
        self.encoder = GLMVoiceTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False)

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


class GLMVoiceForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GLMVoiceBackboneModel(config)
        self.vocab_size = config.vocab_size

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


class GLMVoiceModel(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0"):
        super().__init__(model_name, device, dtype)
        config_path = hf_hub_download(repo_id=model_name, filename="config.json", revision=None)
        self.config = GLMVoiceConfig.from_dict(json.load(open(config_path)))

        self.model = GLMVoiceForCausalLM(self.config)
        self.model.load_state_dict(
            load_hf_safetensor_state_dict(repo_id=model_name, revision=None, token=None),
            strict=False,
        )
        self.model.to(dtype).to(device)

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(model_name)

        audio_decoder_repo = "zai-org/glm-4-voice-decoder"
        audio_decoder_config_path = hf_hub_download(repo_id=audio_decoder_repo, filename="config.yaml", revision=None)
        audio_decoder_flow_path = hf_hub_download(repo_id=audio_decoder_repo, filename="flow.pt", revision=None)
        audio_decoder_hift_path = hf_hub_download(repo_id=audio_decoder_repo, filename="hift.pt", revision=None)
        self.audio_decoder = GLMAudioDecoder(
            config_path=audio_decoder_config_path,
            flow_path=audio_decoder_flow_path,
            hift_path=audio_decoder_hift_path,
            device=device,
        )
        self.audio_decoder.to(device)

        self._num_attention_heads = self.config.num_attention_heads 
        self._num_key_value_heads = self.config.multi_query_group_num
        self._num_hidden_layers = self.config.num_layers
        self._hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        self.stop_token_ids = [151329, 151336, 151338]
        self.audio_offset = self.text_tokenizer.convert_tokens_to_ids("<|audio_0|>")

        self.default_sampling_config = SamplingConfig(
            top_k=None, 
            top_p=0.8,
            min_p=None,
            temperature=0.8,
            repetition_penalty=None,
            repetition_window=None, 
            cfg_scale=None,
        )

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
        return 25
    
    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 0
    
    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        return 512

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] in self.stop_token_ids

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    def _format_prompt(self, input_mode: str, prompt: str | None, audio_path: str | None) -> str:
        if input_mode == "audio":
            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [audio_path]
            )[0]
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        else:
            user_input = prompt
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        
        text_input = f"<|system|>\n{system_prompt}"
        text_input += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        return text_input
    
    def preprocess(
        self, 
        prompt: str | None = None, 
        audio_path: str | None = None,
        input_audio: torch.Tensor | None = None,
    ) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to GLMVoice specifications."""
        text_input = self._format_prompt(
            input_mode="audio" if audio_path is not None else "text",
            prompt=prompt,
            audio_path=audio_path,
        )
        input_ids = self.text_tokenizer(text_input, return_tensors="pt").input_ids
        input_ids = input_ids.view(-1, 1) # add codebook dimension
        
        return PreprocessOutput(input_tokens=input_ids.tolist())
    
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
        
        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            logits = logits[attn_wrapper.qo_indptr[:-1] - 1]
        
        return logits[:, None, :] # add codebook dimension

    def sampling(
        self, 
        logits: torch.Tensor, 
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        output_ids = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long, device=self.device)
        for i in range(self.n_codebooks):
            output_ids[:, i] = Sampler.run_sampling(logits[:, i], config=sampling_params)
        
        for i, req in enumerate(requests):
            # filter out the non-audio tokens
            req.lm_output_tokens.append(output_ids[i].tolist())
            if output_ids[i, 0] >= self.audio_offset:
                # if the first token is an audio token, append it to the audio tokens
                req.lm_output_audio_tokens.append(output_ids[i].tolist())

        return output_ids

    def postprocess(self, token_ids: torch.Tensor):
        return self.audio_decoder(token_ids[:, :, 0] - self.audio_offset)
