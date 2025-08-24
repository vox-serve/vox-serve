from typing import Any, List

import flashinfer
import torch
from torch import nn
from transformers import LlamaConfig, LlamaPreTrainedModel
from transformers.activations import ACT2FN

from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.snac import SNAC
from .base import BaseLM, PreprocessOutput


class OrpheusRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OrpheusRMSNorm is equivalent to T5LayerNorm
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


class OrpheusMLP(nn.Module):
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


class OrpheusAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rope_scale = config.rope_scaling.get("factor", 32.0)
        self.rope_theta = config.rope_theta
        self.low_freq_factor = config.rope_scaling.get("low_freq_factor", 1.0)
        self.high_freq_factor = config.rope_scaling.get("high_freq_factor", 4.0)
        self.old_context_len = config.rope_scaling.get("original_max_position_embeddings", 8192)

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
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        key_states = self.k_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)

        query_states, key_states = flashinfer.rope.apply_llama31_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_scale=self.rope_scale,
            rope_theta=self.rope_theta,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class OrpheusDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OrpheusAttention(config=config, layer_idx=layer_idx)

        self.mlp = OrpheusMLP(config)
        self.input_layernorm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class OrpheusBackboneModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OrpheusDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OrpheusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
                hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class OrpheusForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OrpheusBackboneModel(config)
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
    def __init__(
        self, model_name, dtype=torch.bfloat16, device="cuda:0", tokenizer_path="canopylabs/orpheus-3b-0.1-ft"
    ):
        if model_name == "orpheus":
            model_name = "canopylabs/orpheus-3b-0.1-ft"
        super().__init__(model_name, device, dtype)
        self.model_name = model_name
        self.model = OrpheusForCausalLM.from_pretrained(model_name)
        self.model.to(dtype).to(device)

        self.available_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)
        self.audio_tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

        self._num_attention_heads = self.model.config.num_attention_heads
        self._num_key_value_heads = self.model.config.num_key_value_heads
        self._num_hidden_layers = self.model.config.num_hidden_layers
        self._hidden_size = self.model.config.hidden_size
        # self.vocab_size = self.model.config.vocab_size

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
        return self.model.config.vocab_size

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
        prompt: str = None,
        audio_path: str = None,
        voice="tara",
        model_type="larger",
    ) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        assert audio_path is None
        self._validate_voice(voice)
        input_ids, _ = self._orpheus_format_prompt(prompt, voice, model_type)
        input_ids = input_ids.view(-1, 1)  # add codebook dimension

        repetition_cache = torch.zeros(
            self.default_sampling_config.repetition_window if self.default_sampling_config.repetition_window > 0 else 1,
            self.n_codebooks,
            self.vocab_size,
            dtype=torch.bool,
            device=self.device,
        )

        return PreprocessOutput(input_tokens=input_ids.tolist(), repetition_cache=repetition_cache)

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

        return logits[:, None, :]  # add codebook dimension

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

        # apply repetition penalty
        for i, req in enumerate(requests):
            if req.repetition_cache is None:
                continue

            logits[i] = Sampler.apply_repetition_penalty(
                logits[i], req.repetition_cache, sampling_params.repetition_penalty
            )

        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=sampling_params)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        # update repetition cache
        for i, req in enumerate(requests):
            if req.repetition_cache is None:
                continue

            Sampler.update_repetition_penalty_cache(
                req.repetition_cache,
                output_ids[i],
                sampling_params.repetition_window,
            )

            # no additional logic for Orpheus model for now
            req.lm_output_tokens.append(output_ids[i].tolist())
            req.lm_output_audio_tokens.append(output_ids[i].tolist())

        return output_ids

    def _turn_token_into_id(self, output_ids):
        """Modoel's output ids to audio ids"""
        return (output_ids - 128256 - 10) % 4096

    def postprocess(self, token_ids: torch.Tensor):
        """Convert token IDs to audio bytes."""
        mf = token_ids.view(-1, 4, 7)
        mf = self._turn_token_into_id(mf)

        # codes_0 = mf[:, :, 0]
        # codes_1 = mf[:, :, [1, 4]].view(-1, 8)
        # codes_2 = mf[:, :, [2, 3, 5, 6]].view(-1, 16)

        codes_0 = mf[:, :, 0]

        c1 = torch.index_select(mf, dim=2, index=self.idx_14)
        codes_1 = c1.reshape(-1, 8)

        c2 = torch.index_select(mf, dim=2, index=self.idx_2356)
        codes_2 = c2.reshape(-1, 16)

        codes = [codes_0, codes_1, codes_2]

        with torch.inference_mode():
            audio_tensor = self.audio_tokenizer.decode(codes)

        # Slice [2048:4096]
        audio_tensor = audio_tensor[:, :, 2048:4096]
        return audio_tensor
