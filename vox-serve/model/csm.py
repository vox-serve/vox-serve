from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
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
from transformers import LlamaConfig, CsmConfig, CsmDepthDecoderConfig, CsmPreTrainedModel

# from ..tokenizer.mimi import Mimi
from ..flashinfer_utils import FlashInferWrapper
from ..sampling import top_k_sampling, apply_repetition_penalty
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


class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.hidden_size)
        self.register_buffer(
            "audio_tokens_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, input_ids):
        input_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        # input_embeds = input_embeds.sum(dim=2)
        return input_embeds
    

class CsmBackboneModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size

        self.embed_tokens = CsmBackboneModelEmbeddings(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
    
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


class CsmDepthDecoderModel(nn.Module):
    def __init__(self, config: CsmDepthDecoderConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.backbone_hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)
        
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.inputs_embeds_projector(inputs_embeds)

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


class CsmCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(torch.empty(self.num_codebooks - 1, hidden_size, vocab_size))

    def forward(self, hidden_states, cache_position=None):
        if cache_position is None:
            seq_length = hidden_states.shape[1]
            codebook_weight = self.weight[torch.arange(seq_length)]
        else:
            codebook_idxs = cache_position - 1
            codebook_weight = self.weight[codebook_idxs]

        # print(f"{hidden_states.shape=}, {cache_position=}") # [seq_len, 1024], [0, 1]
        hidden_states = [
            nn.functional.linear(hidden_states[codebook_idx, :], codebook_weight[codebook_idx].T)
            for codebook_idx in range(codebook_weight.shape[0])
        ]
        hidden_states = torch.stack(hidden_states, dim=0)

        return hidden_states


class CsmDepthDecoderForCausalLM(nn.Module):
    def __init__(self, config: CsmConfig):
        super().__init__()
        self.model = CsmDepthDecoderModel(config)
        self.vocab_size = config.vocab_size
        self.codebooks_head = CsmCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor, # [bs]
        position_ids: torch.LongTensor, # [bs]
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        return self.model(inputs_embeds, position_ids, attn_wrapper, kv_cache)


class CsmForConditionalGeneration(CsmPreTrainedModel):
    def __init__(self, config: CsmConfig):
        super().__init__(config)
        print(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.backbone_model = CsmBackboneModel(config=config)
        self.depth_decoder = CsmDepthDecoderForCausalLM(config=config.depth_decoder_config)
        
        from transformers import AutoModel
        self.codec_model = AutoModel.from_config(config.codec_config)
    
    def forward_backbone(
        self,
        inputs_embeds: torch.Tensor, # [bs]
        position_ids: torch.LongTensor, # [bs]
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        outputs = self.backbone_model(inputs_embeds, position_ids, attn_wrapper, kv_cache)
        logits = self.lm_head(outputs)

        return logits, outputs

    def forward_depth(
        self,
        inputs_embeds: torch.Tensor, # [bs]
        position_ids: torch.LongTensor, # [bs]
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        outputs = self.depth_decoder(inputs_embeds, position_ids, attn_wrapper, kv_cache)

        logits = self.depth_decoder.codebooks_head(outputs, cache_position=position_ids)
        return logits
        

class CSMModel(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0", tokenizer_path="meta-llama/Llama-3.2-1B"):
        super().__init__(model_name, device, dtype)
        self.model = CsmForConditionalGeneration.from_pretrained(model_name)
        self.model.to(dtype).to(device)
        
        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)

        from huggingface_hub import hf_hub_download
        import moshi 
        mimi_weight = hf_hub_download('kyutai/moshiko-pytorch-bf16', 'tokenizer-e351c8d8-checkpoint125.safetensors')
        self.audio_tokenizer = moshi.models.loaders.get_mimi(mimi_weight, device=device)
        self.audio_tokenizer.set_num_codebooks(32)

        self._num_attention_heads = self.model.config.num_attention_heads 
        self._num_key_value_heads = self.model.config.num_key_value_heads
        self._num_hidden_layers = self.model.config.num_hidden_layers
        self._hidden_size = self.model.config.hidden_size
        self._depth_num_attention_heads = self.model.config.depth_decoder_config.num_attention_heads
        self._depth_num_key_value_heads = self.model.config.depth_decoder_config.num_key_value_heads
        self._depth_num_hidden_layers = self.model.config.depth_decoder_config.num_hidden_layers
        self._depth_hidden_size = self.model.config.depth_decoder_config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        self.top_k = 50
        self.temperature = 0.9
        self.stop_token_id = 0
        self.max_tokens = 1200 

        self._set_default_context()


    @property
    def has_depth_transformer(self) -> bool:
        """Indicates if the model has a depth transformer."""
        return True
    
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
    def depth_n_codebooks(self) -> int:
        """Number of codebooks in the depth transformer."""
        return self.model.config.depth_decoder_config.num_codebooks
    
    @property
    def embed_text_tokens(self):
        """Embedding layer for text tokens."""
        return self.model.embed_text_tokens
    
    @property
    def embed_audio_tokens_all(self):
        """Embedding layer for audio tokens, all codebooks at a time."""
        return self.model.backbone_model.embed_tokens

    def embed_audio_tokens_single(self, ids, i):
        """Embed audio tokens for a specific codebook."""
        return self.model.backbone_model.embed_tokens.embed_audio_tokens(ids + i * self.model.config.vocab_size)

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        from transformers import AutoTokenizer
        from tokenizers.processors import TemplateProcessing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )

        return tokenizer
    
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)
    
    def _set_default_context(self):
        # Example from https://github.com/SesameAILabs/csm/blob/main/run_csm.py
        # Default prompts are available at https://hf.co/sesame/csm-1b
        from huggingface_hub import hf_hub_download
        import torchaudio

        prompt_filepath_conversational_a = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        prompt_filepath_conversational_b = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_b.wav"
        )

        def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_tensor = audio_tensor.squeeze(0)
            # Resample is lazy so we can always call it
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
            )
            return audio_tensor

        SPEAKER_PROMPTS = {
            "conversational_a": {
                "text": (
                    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                    "start really early I'd be like okay I'm gonna start revising now and then like "
                    "you're revising for ages and then I just like start losing steam I didn't do that "
                    "for the exam we had recently to be fair that was a more of a last minute scenario "
                    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                    "sort of start the day with this not like a panic but like a"
                ),
                "audio": load_prompt_audio(prompt_filepath_conversational_a, self.audio_tokenizer.sample_rate)
            },
            "conversational_b": {
                "text": (
                    "like a super Mario level. Like it's very like high detail. And like, once you get "
                    "into the park, it just like, everything looks like a computer game and they have all "
                    "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                    "will have like a question block. And if you like, you know, punch it, a coin will "
                    "come out. So like everyone, when they come into the park, they get like this little "
                    "bracelet and then you can go punching question blocks around."
                ),
                "audio": load_prompt_audio(prompt_filepath_conversational_b, self.audio_tokenizer.sample_rate)
            }
        }

        tokens, tokens_mask = [], []
        for speaker, prompt in SPEAKER_PROMPTS.items():
            text_tokens, text_mask = self._tokenize_text_segment(prompt["text"], speaker=0 if speaker == "conversational_a" else 1)
            audio_tokens, audio_mask = self._tokenize_audio(prompt["audio"])
            tokens.append(torch.cat([text_tokens, audio_tokens], dim=0))
            tokens_mask.append(torch.cat([text_mask, audio_mask], dim=0))

        self.default_context = {
            "tokens": tokens,
            "tokens_mask": tokens_mask,
        }
    
    def is_stop_id(self, token_id):
        return token_id[-1] == self.stop_token_id
    
    def preprocess(self, prompt, speaker=0, context=None):
        """Prepare the prompt for the model, formatting it according to CSM specifications."""
        # TODO: add reference context to API argument
        prompt_tokens, prompt_tokens_mask = self._tokenize_text_segment(prompt, speaker)
        if context is None:
            prompt_tokens = torch.cat(self.default_context["tokens"] + [prompt_tokens], dim=0)
            prompt_tokens_mask = torch.cat(self.default_context["tokens_mask"] + [prompt_tokens_mask], dim=0)
        
        print(f"{prompt_tokens=}, {prompt_tokens_mask=}") # [seq_len, 33]
        return prompt_tokens.tolist(), prompt_tokens_mask.tolist()
    
    def forward(
        self, 
        input_ids: torch.LongTensor, 
        position_ids: torch.LongTensor, 
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor, 
        repetition_cache: Optional[torch.Tensor] = None,
        depth_attn_wrapper: Optional[FlashInferWrapper] = None,
        depth_kv_cache: Optional[torch.Tensor] = None,
        tokens_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the model."""
        # input_ids: [bs, n_codebook]
        # print(f"{input_ids=} {input_ids.shape=}")
        text_embeds = self.embed_text_tokens(input_ids[:, -1:])
        audio_embeds = self.embed_audio_tokens_all(input_ids[:, :-1])
        # print(f"{input_ids=} {input_ids.shape=}, {text_embeds.shape=}, {audio_embeds.shape=}") # [bs, 33], [bs, 1, 2048], [bs, 32, 2048]
        # print(f"{text_embeds=} {audio_embeds=}") # [bs, 1, 2048], [bs, 32, 2048]
        # print(f"{tokens_mask.shape=}, {tokens_mask=}") # [bs, 33]
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1) # [bs, 33, 2048]
        inputs_embeds = inputs_embeds * tokens_mask[:, :, None]
        inputs_embeds = inputs_embeds.sum(dim=1)

        # print(f"{inputs_embeds=}")
        backbone_logits, backbone_last_hidden = self.model.forward_backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        # logits = apply_repetition_penalty(logits, repetition_cache, self.repetition_penalty)
        
        # select last token for each request for prefill 
        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            backbone_logits = backbone_logits[attn_wrapper.qo_indptr[:-1] - 1]
            backbone_last_hidden = backbone_last_hidden[attn_wrapper.qo_indptr[:-1] - 1]

        backbone_ids = top_k_sampling(backbone_logits, top_k=self.top_k, temperature=self.temperature)
        
        c0_embed = self.embed_audio_tokens_single(backbone_ids, 0)
        # backbone_ids.shape=torch.Size([1])
        # print(f"{backbone_last_hidden.shape=}, {c0_embed.shape=}") # [bs, 2048], [bs, 2048]
        curr_h = torch.cat([backbone_last_hidden[:, None, :], c0_embed[:, None, :]], dim=1).view(-1, c0_embed.shape[-1]) 

        depth_position_ids = torch.tensor([0, 1] * c0_embed.shape[0], device=c0_embed.device)
        depth_qo_indptr = torch.arange(c0_embed.shape[0] + 1, device=curr_h.device) * 2
        depth_kv_indptr = torch.arange(c0_embed.shape[0] + 1, device=curr_h.device)
        depth_kv_indices = torch.arange(c0_embed.shape[0], device=curr_h.device)
        depth_kv_last_page_len = torch.tensor([2] * c0_embed.shape[0], device=curr_h.device)
        depth_kv_cache.zero_()

        output_ids = backbone_ids[:, None] # (bs, 1)

        for i in range(1, self.model.config.num_codebooks):
            depth_attn_wrapper.plan(
                qo_indptr=depth_qo_indptr,
                paged_kv_indptr=depth_kv_indptr, 
                paged_kv_indices=depth_kv_indices, 
                paged_kv_last_page_len=depth_kv_last_page_len,
                dtype=self.dtype,
            )
            torch.cuda.synchronize()

            # print(f"{curr_h=}")
            # print(f"{curr_h.shape=} {depth_position_ids.shape=}, {depth_position_ids=}") # [bs, 2048], [bs, 2]
            depth_logits = self.model.forward_depth(
                inputs_embeds=curr_h,
                position_ids=depth_position_ids,
                attn_wrapper=depth_attn_wrapper,
                kv_cache=depth_kv_cache,
            )
            # print(f"{depth_logits.shape=}")
            depth_ids = top_k_sampling(depth_logits, top_k=self.top_k, temperature=self.temperature)
            depth_ids = depth_ids.view(-1, c0_embed.shape[0])[-1, :]
            ci_embed = self.embed_audio_tokens_single(depth_ids, i)
            curr_h = ci_embed
            # print(f"{depth_ids.shape=} {depth_ids=}")
            output_ids = torch.cat([output_ids, depth_ids[:, None]], dim=1) # (bs, N)

            depth_position_ids = torch.tensor([i + 1] * c0_embed.shape[0], device=c0_embed.device)
            depth_qo_indptr = torch.arange(c0_embed.shape[0] + 1, device=curr_h.device)
            depth_kv_last_page_len += 1
        
        # output_ids = torch.cat([output_ids, torch.zeros(1, 1, device=output_ids.device, dtype=torch.long)], dim=1) # text stream is 0
        print(f"{output_ids.shape=} {output_ids=}") # [bs, 33]
        return output_ids
    
    def decode_text_token(self, token_id):
        return self.text_tokenizer.decode(token_id)

    def postprocess(self, tokens_list):
        # mimi decoder
        print(torch.tensor(tokens_list).shape) # (seq_len, 32)
        audio = self.audio_tokenizer.decode(torch.tensor(tokens_list, device="cuda").transpose(1, 0)[None, :, :])
        audio = audio.detach().cpu().numpy() 
        audio_int16 = (audio * 32767).astype(np.int16) 
        audio_bytes = audio_int16.tobytes()
        # TODO: watermarking
        return audio_bytes


if __name__ == "__main__":
    model = CSMModel(model_name="sesame/csm-1b", dtype=torch.bfloat16, device="cuda:0")

    model.preprocess("Hello from Sesame.", speaker=0, context=[])