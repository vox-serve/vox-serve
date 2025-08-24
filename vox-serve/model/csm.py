from typing import Any, List, Tuple

import flashinfer
import torch
import torchaudio
from tokenizers.processors import TemplateProcessing
from torch import nn
from transformers import AutoTokenizer, CsmConfig, CsmDepthDecoderConfig, CsmPreTrainedModel, LlamaConfig
from transformers.activations import ACT2FN

from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..utils import get_logger
from .base import BaseLMWithDepth, PreprocessOutput


class CsmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        CsmRMSNorm is equivalent to T5LayerNorm
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


class CsmMLP(nn.Module):
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


class CsmAttention(nn.Module):
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


class CsmDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CsmAttention(config=config, layer_idx=layer_idx)
        self.mlp = CsmMLP(config)

        self.input_layernorm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.hidden_size)
        self.register_buffer(
            "audio_tokens_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, input_ids):
        input_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        return input_embeds


class CsmBackboneModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = CsmBackboneModelEmbeddings(config)
        self.layers = nn.ModuleList(
            [CsmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class CsmDepthDecoderModel(nn.Module):
    def __init__(self, config: CsmDepthDecoderConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.backbone_hidden_size)
        self.layers = nn.ModuleList(
            [CsmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.inputs_embeds_projector(inputs_embeds)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
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
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        return self.model(inputs_embeds, position_ids, attn_wrapper, kv_cache)


class CsmForConditionalGeneration(CsmPreTrainedModel):
    def __init__(self, config: CsmConfig):
        super().__init__(config)
        logger = get_logger(__name__)
        logger.debug(f"CSM Config: {config}")
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.backbone_model = CsmBackboneModel(config=config)
        self.depth_decoder = CsmDepthDecoderForCausalLM(config=config.depth_decoder_config)

        from transformers import AutoModel

        self.codec_model = AutoModel.from_config(config.codec_config)

    def forward_backbone(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        outputs = self.backbone_model(inputs_embeds, position_ids, attn_wrapper, kv_cache)
        logits = self.lm_head(outputs)

        return logits, outputs

    def forward_depth(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        outputs = self.depth_decoder(inputs_embeds, position_ids, attn_wrapper, kv_cache)

        logits = self.depth_decoder.codebooks_head(outputs, cache_position=position_ids)
        return logits


class CSMModel(BaseLMWithDepth):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0", tokenizer_path="meta-llama/Llama-3.2-1B"):
        if model_name == "csm":
            model_name = "sesame/csm-1b"
        super().__init__(model_name, device, dtype)
        self.logger = get_logger(__name__)
        self.model = CsmForConditionalGeneration.from_pretrained(model_name)
        self.model.to(dtype).to(device)

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(tokenizer_path)

        import moshi
        from huggingface_hub import hf_hub_download

        # TODO: drop moshi dependency
        mimi_weight = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
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
        # self.vocab_size = self.model.config.vocab_size

        self.stop_token_id = 0

        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=None,
            min_p=None,
            temperature=0.9,
            repetition_penalty=None,
            repetition_window=None,
            cfg_scale=None,
        )

        self._set_default_context()

    @property
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        return self.model.config.depth_decoder_config.num_codebooks + 1

    @property
    def depth_n_codebooks(self) -> int:
        """Number of codebooks in the depth transformer."""
        return self.model.config.depth_decoder_config.num_codebooks

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
    def needs_watermarking(self) -> bool:
        """Indicates if the model requires watermarking."""
        return True

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

        prompt_filepath_conversational_a = hf_hub_download(
            repo_id="sesame/csm-1b", filename="prompts/conversational_a.wav"
        )
        prompt_filepath_conversational_b = hf_hub_download(
            repo_id="sesame/csm-1b", filename="prompts/conversational_b.wav"
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
                "audio": load_prompt_audio(prompt_filepath_conversational_a, self.audio_tokenizer.sample_rate),
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
                "audio": load_prompt_audio(prompt_filepath_conversational_b, self.audio_tokenizer.sample_rate),
            },
        }

        tokens, tokens_mask = [], []
        for speaker, prompt in SPEAKER_PROMPTS.items():
            text_tokens, text_mask = self._tokenize_text_segment(
                prompt["text"], speaker=0 if speaker == "conversational_a" else 1
            )
            audio_tokens, audio_mask = self._tokenize_audio(prompt["audio"])
            tokens.append(torch.cat([text_tokens, audio_tokens], dim=0))
            tokens_mask.append(torch.cat([text_mask, audio_mask], dim=0))

        self.default_context = {
            "tokens": tokens,
            "tokens_mask": tokens_mask,
        }

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 1

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 0

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return 1920  # 24000 / 12.5

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        return 1200

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.model.config.vocab_size

    def is_stop_id(self, token_ids: List[int]) -> int:
        # index -2 since we want to check the final audio codebook before text stream
        return token_ids[-2] == self.stop_token_id

    def preprocess(self, prompt: str = None, audio_path: str = None, speaker=0, context=None) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to CSM specifications."""
        # TODO: add reference context to API argument
        assert audio_path is None
        prompt_tokens, prompt_tokens_mask = self._tokenize_text_segment(prompt, speaker)
        if context is None:
            prompt_tokens = torch.cat(self.default_context["tokens"] + [prompt_tokens], dim=0)
            prompt_tokens_mask = torch.cat(self.default_context["tokens_mask"] + [prompt_tokens_mask], dim=0)

        return PreprocessOutput(input_tokens=prompt_tokens.tolist(), input_masks=prompt_tokens_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone model."""
        text_embeds = self.embed_text_tokens(input_ids[:, -1:])
        audio_embeds = self.embed_audio_tokens_all(input_ids[:, :-1])
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)  # [bs, 33, 2048]

        # input_masks = torch.cat(input_masks, dim=0)  # [bs, 33]
        inputs_embeds = inputs_embeds * input_masks[:, :, None]
        inputs_embeds = inputs_embeds.sum(dim=1)

        backbone_logits, backbone_last_hidden = self.model.forward_backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        # select last token for each request for prefill
        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            backbone_logits = backbone_logits[attn_wrapper.qo_indptr[:-1] - 1]
            backbone_last_hidden = backbone_last_hidden[attn_wrapper.qo_indptr[:-1] - 1]

        # add codebook dimension
        return backbone_logits[:, None, :], backbone_last_hidden

    def sampling(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        cfg_scale: float | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the first audio codebook from the backbone transoformer logits.
        The initial input for the depth transformer is concatenation of last hidden state and embedding for codebook 0,
        which essentially is the prefill with sequence length of 2.
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        assert logits.shape[1] == 1, "Logits should have shape [bs, 1, vocab_size]"

        # there are 33 codebooks (32 audio + 1 text), but the output from backbone transformer is single codebook
        # so here we allocate output_ids for all codebooks but do sampling only for the first one
        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=sampling_params)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        c0_embed = self.embed_audio_tokens_single(output_ids[:, 0], 0)
        # backbone_ids.shape=torch.Size([1])
        # print(f"{backbone_last_hidden.shape=}, {c0_embed.shape=}") # [bs, 2048], [bs, 2048]
        hidden_for_depth = torch.cat([hidden_states[:, None, :], c0_embed[:, None, :]], dim=1).view(
            -1, c0_embed.shape[-1]
        )

        for i, req in enumerate(requests):
            req.input_masks = torch.ones(self.n_codebooks, dtype=torch.bool, device=self.device)[None, :]
            req.input_masks[:, -1] = False  # only the audio streams are used in decode phase

            # no additional logic for CSM model
            req.lm_output_tokens.append(output_ids[i].tolist())
            req.lm_output_audio_tokens.append(output_ids[i].tolist())

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

        # We don't do this here since prefill is also captured in cuda graph
        # for depth transformer
        # if getattr(attn_wrapper, "qo_indptr", None) is not None:
        #     depth_logits = depth_logits[attn_wrapper.qo_indptr[:-1] - 1]

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
        ci_embed = self.embed_audio_tokens_single(output_ids, i_iteration)

        for i, req in enumerate(requests):
            req.lm_output_tokens[-1][i_iteration] = output_ids[i].item()
            req.lm_output_audio_tokens[-1][i_iteration] = output_ids[i].item()

        return output_ids, ci_embed

    def postprocess(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, interval, 33)
        # there are 33 codebooks including text
        tokens_to_process = token_ids[:, :, :-1].transpose(1, 2)  # (batch_size, 32, interval)

        # mimi decoder
        # TODO: caching for mimi
        audio_tensor = self.audio_tokenizer.decode(tokens_to_process)
        # audio_tensor: (batch_size, 1, N)

        return audio_tensor


if __name__ == "__main__":
    model = CSMModel(model_name="sesame/csm-1b", dtype=torch.bfloat16, device="cuda:0")

    model.preprocess("Hello from Sesame.", speaker=0, context=[])
