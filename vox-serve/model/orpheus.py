from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import asyncio
import threading
import queue
import wave

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import LlamaConfig, LlamaPreTrainedModel

from ..tokenizer.snac import SNAC
from ..flashinfer_utils import FlashInferWrapper
from ..sampling import top_p_sampling, apply_repetition_penalty, update_repetition_penalty_cache, SamplingConfig
from ..requests import Request
from .base import BaseLM


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
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


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # If position_ids is flat (i.e. 1D [total_seq_len]), unsqueeze it to [1, total_seq_len]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Core RoPE block: now position_ids is of shape [batch, seq_len] where batch=1 if it was flattened
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # (batch, num_freq, 1) @ (batch, 1, seq_len) -> (batch, num_freq, seq_len)
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Apply scaling
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        # Squeeze out the batch dimension if it was originally flattened
        return cos.squeeze(0).to(dtype=x.dtype), sin.squeeze(0).to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_wrapper.set_kv_cache(kv_cache, key_states.transpose(0, 1), value_states.transpose(0, 1))
        attn_output = attn_wrapper.run(query_states.transpose(0, 1), kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
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

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def embed_tokens(self, input_ids):
        return self.model.embed_tokens(input_ids)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs) 

        return logits


class OrpheusModel(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0", tokenizer_path="canopylabs/orpheus-3b-0.1-ft"):
        super().__init__(model_name, device, dtype)
        self.model_name = model_name
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.model.to(dtype).to(device)

        self.available_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)
        self.audio_tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

        self._num_attention_heads = self.model.config.num_attention_heads 
        self._num_key_value_heads = self.model.config.num_key_value_heads
        self._num_hidden_layers = self.model.config.num_hidden_layers
        self._hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        self.stop_token_id = 49158
        self.temperature = 0.6 
        self.top_p = 0.8 
        self.max_tokens = 1200 
        self.repetition_penalty=1.3
        self.repetition_penalty_window = -1  # -1 means no window, apply to all tokens

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
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.text_tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.text_tokenizer.decode(all_input_ids[0])
                return all_input_ids, prompt_string
            else:
                prompt_tokens = self.text_tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.text_tokenizer.decode(all_input_ids[0])
                return all_input_ids, prompt_string
    
    def preprocess(
        self, 
        prompt: str, 
        voice="tara", 
        model_type="larger",
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        self._validate_voice(voice)
        input_ids, _ = self._orpheus_format_prompt(prompt, voice, model_type)
        input_ids = input_ids.view(-1, 1) # add codebook dimension

        preprocess_dict = {
            "repetition_cache": torch.zeros(
                self.repetition_penalty_window if self.repetition_penalty_window > 0 else 1, 
                self.n_codebooks, 
                self.vocab_size, 
                dtype=torch.bool, 
                device=self.device,
            ),
        }
        print(f"{input_ids.shape=}")
        return input_ids.tolist(), preprocess_dict
    
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
        # apply repetition penalty
        for i, req in enumerate(requests):
            if req.repetition_cache is None:
                continue

            logits[i] = apply_repetition_penalty(
                logits[i], 
                req.repetition_cache, 
                self.repetition_penalty
            )

        output_ids = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long, device=self.device)
        for i in range(self.n_codebooks):
            output_ids[:, i] = top_p_sampling(logits[:, i], top_p=self.top_p, temperature=self.temperature)
        
        # update repetition cache
        for i, req in enumerate(requests):
            if req.repetition_cache is None:
                continue

            update_repetition_penalty_cache(
                req.repetition_cache, 
                output_ids[i], 
                self.repetition_penalty_window, 
            )

        return output_ids
    
    def _turn_token_into_id(self, output_ids):
        """Modoel's output ids to audio ids"""
        return (output_ids - 128256 - 10) % 4096

    def postprocess(self, token_ids: torch.Tensor):
        """Convert token IDs to audio bytes."""
        # 1) Turn the list of length=28 into a (4×7) tensor
        print(f"{token_ids.shape=} {token_ids.flatten().cpu().tolist()=}")  # [bs, 28, 1]
        mf = token_ids.view(-1, 4, 7)
        mf = self._turn_token_into_id(mf)

        # 2) codes_0: take column 0 from each of the 4 rows → shape (4,)
        codes_0 = mf[:, :, 0] # [[f0[0], f1[0], f2[0], f3[0]], ..]
        # #    then add a batch‐dim → (1, 4)
        # codes_0 = codes_0.unsqueeze(0)

        # 3) codes_1: for each row i, take [col 1, col 4] → gives shape (4, 2)
        #    then .reshape(-1) flattens row‐by‐row: [f0[1], f0[4], f1[1], f1[4], …]
        codes_1 = mf[:, :, [1, 4]].view(-1, 8)    # shape (8,)
        # codes_1 = codes_1.unsqueeze(0)         # → (1, 8)

        # 4) codes_2: for each row i, take [col 2, col 3, col 5, col 6] → shape (4, 4)
        #    then flatten row‐by‐row: [f0[2],f0[3],f0[5],f0[6], f1[2],…]
        codes_2 = mf[:, :, [2, 3, 5, 6]].view(-1, 16)  # shape (16,)
        # codes_2 = codes_2.unsqueeze(0)             # → (1, 16)

        codes = [codes_0, codes_1, codes_2]

        # # 5) Range‐check every code
        # for c in codes:
        #     if torch.any(c < 0) or torch.any(c > 4096):
        #         return

        # 6) Decode under inference_mode (all shapes are now fixed)
        with torch.inference_mode():
            audio_tensor = self.audio_tokenizer.decode(codes)
            #    audio_hat.shape == [1, 1, 8192]

        # 7) Slice [2048:4096], move to CPU, convert to int16, return bytes
        audio_tensor = audio_tensor[:, :, 2048:4096]            # (1, 1, 2048)
        return audio_tensor
        # arr = audio_slice.detach().cpu().numpy()            # (1, 1, 2048)
        # audio_int16 = (arr * 32767).astype(np.int16)        # still (1, 1, 2048)
        # audio_bytes = audio_int16.tobytes()
        # return audio_bytes, next_audio_decode_idx + 7
