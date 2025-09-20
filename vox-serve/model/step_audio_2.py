import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F

from ..encoder.step_audio_2 import StepAudio2Encoder
from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.step_audio_2 import StepAudio2Decoder
from ..utils import get_logger, load_hf_safetensor_state_dict
from .base import BaseLM, PreprocessOutput


@dataclass
class StepAudio2TextConfig:
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_attention_groups: int = 4
    num_key_value_heads: int = 4
    num_hidden_layers: int = 28
    max_seq_len: int = 16384
    vocab_size: int = 158720
    rms_norm_eps: float = 1e-6
    eos_token_id: int = 151643
    pad_token_id: int = 151643
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 16384
    rope_scaling: Optional[dict] = None
    torch_dtype: str = "bfloat16"


@dataclass
class StepAudio2AudioEncoderConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    n_codebook_size: int = 4096
    llm_dim: int = 3584
    kernel_size: int = 3
    adapter_stride: int = 2


@dataclass
class StepAudio2Config:
    text_config: StepAudio2TextConfig = StepAudio2TextConfig()
    audio_encoder_config: StepAudio2AudioEncoderConfig = StepAudio2AudioEncoderConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StepAudio2Config":
        text_config = StepAudio2TextConfig(**config_dict.get("text_config", {}))
        audio_encoder_config = StepAudio2AudioEncoderConfig(**config_dict.get("audio_encoder_config", {}))
        return cls(text_config=text_config, audio_encoder_config=audio_encoder_config)


class StepAudio2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        StepAudio2RMSNorm is equivalent to T5LayerNorm
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


class StepAudio2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output


class StepAudio2Attention(nn.Module):
    def __init__(self, config: StepAudio2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_theta = config.rope_theta

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

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

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            # rotary_dim=self.head_dim // 2,
            # interleave=True,
            rope_theta=self.rope_theta,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class StepAudio2DecoderLayer(nn.Module):
    def __init__(self, config: StepAudio2TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = StepAudio2Attention(config=config, layer_idx=layer_idx)
        self.mlp = StepAudio2MLP(config)

        self.input_layernorm = StepAudio2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = StepAudio2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class StepAudio2BackboneModel(nn.Module):
    def __init__(self, config: StepAudio2TextConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([StepAudio2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = StepAudio2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        hidden_states = self.norm(hidden_states)

        return hidden_states


class StepAudio2Adaptor(nn.Module):
    def __init__(
        self,
        n_state: int = 1280,
        n_hidden: int = 3072,
        kernel_size: int = 7,
        stride: int = 4
    ):
        super().__init__()
        self.stride = stride
        if self.stride != -1:
            self.conv = nn.Conv1d(n_state, n_state, kernel_size, stride, padding=1)
        self.linear1 = nn.Linear(n_state, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        if self.stride != -1:
            x = x.permute(0, 2, 1)
            x = F.gelu(self.conv(x))
            x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class StepAudio2ForCausalLM(nn.Module):
    def __init__(self, config: StepAudio2Config):
        super().__init__()
        self.model = StepAudio2BackboneModel(config.text_config)
        self.encoder = StepAudio2Encoder(
            n_mels=config.audio_encoder_config.n_mels,
            n_ctx=config.audio_encoder_config.n_audio_ctx,
            n_state=config.audio_encoder_config.n_audio_state,
            n_head=config.audio_encoder_config.n_audio_head,
            n_layer=config.audio_encoder_config.n_audio_layer
        )
        self.adapter = StepAudio2Adaptor(
            n_state=config.audio_encoder_config.n_audio_state,
            n_hidden=config.audio_encoder_config.llm_dim,
            kernel_size=config.audio_encoder_config.kernel_size,
            stride=config.audio_encoder_config.adapter_stride
        )
        self.lm_head = torch.nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.vocab_size = config.vocab_size

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


class StepAudio2Model(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0"):
        if model_name == "step":
            model_name = "stepfun-ai/Step-Audio-2-mini"
        super().__init__(model_name, device, dtype)
        self.logger = get_logger(__name__)
        config_path = hf_hub_download(repo_id=model_name, filename="config.json", revision=None)
        self.config = StepAudio2Config.from_dict(json.load(open(config_path)))

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(model_name)
        self.text_tokenizer.eos_token = "<|EOT|>"
        self.config.text_config.eos_token_id = self.text_tokenizer.convert_tokens_to_ids("<|EOT|>")

        self.model = StepAudio2ForCausalLM(self.config)
        self.model.load_state_dict(
            load_hf_safetensor_state_dict(repo_id=model_name, revision=None, token=None),
            strict=False,
        )
        self.model.to(dtype).to(device)

        # audio_decoder_repo = "zai-org/glm-4-voice-decoder"
        # audio_decoder_config_path = hf_hub_download(repo_id=audio_decoder_repo, filename="config.yaml", revision=None)
        # audio_decoder_flow_path = hf_hub_download(repo_id=audio_decoder_repo, filename="flow.pt", revision=None)
        # audio_decoder_hift_path = hf_hub_download(repo_id=audio_decoder_repo, filename="hift.pt", revision=None)
        self.audio_decoder = StepAudio2Decoder(
            model_path=model_name,
            float16=True,
        )
        self.audio_decoder.to(device)

        self._num_attention_heads = self.config.text_config.num_attention_heads
        self._num_key_value_heads = self.config.text_config.num_key_value_heads
        self._num_hidden_layers = self.config.text_config.num_hidden_layers
        self._hidden_size = self.config.text_config.hidden_size
        # self.vocab_size = self.config.vocab_size

        self.stop_token_id = self.text_tokenizer.convert_tokens_to_ids("<|EOT|>")
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

        # for cuda graph compatibility
        self.detokenize_token_len = torch.tensor([self.detokenize_interval], dtype=torch.int32, device=self.device)

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
    def supports_audio_input(self) -> bool:
        """Indicates if the model accepts audio input."""
        return True

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 28

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 3

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        raise NotImplementedError()

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 512

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.config.text_config.vocab_size

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.stop_token_id

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _extract_speech_token(self, audio_path: str) -> List[List[int]]:
        """Extract speech tokens from audio file."""
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            # audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)[0]
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)

        output_tokens = []
        time_step = 0
        while time_step * 16000 < audio.shape[0]:
            audio_segment = audio[0, time_step * 16000 : (time_step + 30) * 16000]
            tokens = self.audio_encoder.encode(audio_segment)

            output_tokens.extend(tokens.tolist())

            time_step += 30

        return output_tokens

    def _format_prompt(self, input_mode: str, prompt: str | None, audio_path: str | None) -> str:
        if input_mode == "audio":
            audio_tokens = self._extract_speech_token(audio_path)[0]
            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            user_input = audio_tokens
            system_prompt = (
                "User will provide you with a speech instruction. Do it step by step. First, think about the "
                "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
            )
        else:
            user_input = prompt
            system_prompt = (
                "User will provide you with a text instruction. Do it step by step. First, think about the "
                "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            )

        text_input = f"<|system|>\n{system_prompt}"
        text_input += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        return text_input

    def preprocess(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
    ) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to StepAudio2 specifications."""
        text_input = self._format_prompt(
            input_mode="audio" if audio_path is not None else "text",
            prompt=prompt,
            audio_path=audio_path,
        )
        input_ids = self.text_tokenizer(text_input, return_tensors="pt").input_ids
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
            stop_mask = (output_ids[:, 0] == self.stop_token_ids[0]) | \
                        (output_ids[:, 0] == self.stop_token_ids[1]) | \
                        (output_ids[:, 0] == self.stop_token_ids[2])
            audio_mask = output_ids[:, 0] >= self.audio_offset

            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                if audio_mask[i] and not stop_mask[i]:
                    req.lm_output_audio_tokens.append(output_ids[i : i + 1])
                if stop_mask[i]:
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

            if repetition_cache is not None:
                # Update repetition cache in requests
                for i, req in enumerate(requests):
                    req.repetition_cache = repetition_cache[i]

        task = update_req_states()

        return output_ids, task

    def postprocess(self, token_ids: torch.Tensor):
        audio_tensor = self.audio_decoder(token_ids[:, :, 0] - self.audio_offset, self.detokenize_token_len)
        return audio_tensor[:, None, :]  # add channel dimension
