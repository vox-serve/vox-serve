import math
from typing import Any, List, Optional, Union

from dataclasses import dataclass
import librosa
from safetensors.torch import safe_open
import torch
from torch import nn
from tokenizers import Tokenizer
from transformers import LlamaConfig
from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..encoder.chatterbox import ChatterboxCondEnc, T3Cond
from ..tokenizer.chatterbox import ChatterboxDecoder
from .base import BaseLM, PreprocessOutput


class ChatterboxConfig:
    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

        # Arbitrary small number that won't cause problems when loading.
        # These param are unused due to custom input layers.
        self.vocab_size=8
        # default params needed for loading most pretrained 1B weights
        self.max_position_embeddings=131072
        self.hidden_size=1024
        self.intermediate_size=4096
        self.num_hidden_layers=30
        self.num_attention_heads=16
        self.head_dim=64
        self.hidden_act="silu"
        self.attention_bias=False
        self.mlp_bias=False
        self.num_key_value_heads=16
        self.rms_norm_eps=1e-05
        self.rope_scaling=dict(
            factor=8.0,
            high_freq_factor=4.0,
            low_freq_factor=1.0,
            original_max_position_embeddings=8192,
            rope_type="llama3",
        )
        self.rope_theta=500000.0
        self.pad_token_id = None

    @property
    def n_channels(self):
        return 1024
    
    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        # TODO: workaround to make tensor shapes at detokenizer constant
        kwargs['gen']['prompt_token'] = kwargs['gen']['prompt_token'][:, :128]
        kwargs['gen']['prompt_token_len'] = 128
        kwargs['gen']['prompt_feat'] = kwargs['gen']['prompt_feat'][:, :256, :]
        kwargs['gen']['prompt_feat_len'] = 256
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ChatterboxRMSNorm is equivalent to T5LayerNorm
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


class ChatterboxMLP(nn.Module):
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


class ChatterboxAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
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

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
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


class ChatterboxDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ChatterboxAttention(config=config, layer_idx=layer_idx)

        self.mlp = ChatterboxMLP(config)
        self.input_layernorm = ChatterboxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ChatterboxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class ChatterboxBackboneModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # not used
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ChatterboxDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ChatterboxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class ChatterboxLearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        """
        Returns positional embeddings for index 0 up to the length of x
        """
        return self.emb(x)

    def get_fixed_embedding(self, idx: Union[int, torch.Tensor]):
        """
        Args:
            idx: scalar int or an integer tensor of shape (T,) or (B, T)
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        """
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx = torch.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx)  # (B, T, dim)


class ChatterboxForCausalLM(nn.Module):
    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config 

        self.tfmr = ChatterboxBackboneModel(config)
        self.cond_enc = ChatterboxCondEnc(config)

        self.text_emb = nn.Embedding(config.text_tokens_dict_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_tokens_dict_size, config.hidden_size)
        
        self.text_pos_emb = ChatterboxLearnedPositionEmbeddings(config.max_text_tokens + 2, config.hidden_size)
        self.speech_pos_emb = ChatterboxLearnedPositionEmbeddings(config.max_speech_tokens + 4, config.hidden_size)
        
        # self.vocab_size = config.vocab_size
        self.text_head = nn.Linear(config.hidden_size, config.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_tokens_dict_size, bias=False)
    
    # def prepare_conditioning(self, t3_cond: T3Cond):
    #     """
    #     Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
    #     """
    #     if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
    #         t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
    #             self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
    #     return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    # def prepare_input_embeds(
    #     self,
    #     *,
    #     t3_cond: T3Cond,
    #     text_tokens: torch.LongTensor,
    #     speech_tokens: torch.LongTensor,
    #     cfg_weight: float = 0.0,
    # ):
    #     # prepare input embeddings (skip backbone tranformer embeddings)
    #     cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
    #     text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
    #     if cfg_weight > 0.0:
    #         text_emb[1].zero_()  # CFG uncond

    #     speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
    #     if self.hp.input_pos_emb == "learned":
    #         text_emb = text_emb + self.text_pos_emb(text_tokens)
    #         speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
    #     len_cond = cond_emb.size(1)

    #     if cond_emb.size(0) != text_emb.size(0):
    #          cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

    #     # concat
    #     embeds = torch.stack([
    #         torch.cat((ce, te, se))
    #         for ce, te, se in zip(cond_emb, text_emb, speech_emb)
    #     ])  # (B, length, dim)
    #     return embeds, len_cond

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        outputs = self.tfmr(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        logits = self.speech_head(outputs)

        return logits


class ChatterboxModel(BaseLM):
    def __init__(
        self,
        model_name,
        dtype=torch.bfloat16,
        device="cuda:0",
        enable_torch_compile=False,
    ):
        if model_name == "chatterbox":
            model_name = "ResembleAI/chatterbox"
        super().__init__(model_name, device, dtype, enable_torch_compile)
        self.model_name = model_name
        self.config = ChatterboxConfig.english_only() if "chatterbox" in model_name else ChatterboxConfig.multilingual()
        self.model = ChatterboxForCausalLM(self.config)
        self.audio_tokenizer = ChatterboxDecoder()

        state_dict = {}
        with safe_open(hf_hub_download(repo_id=model_name, filename="t3_cfg.safetensors", revision=None), framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(dtype).to(device)

        detokenizer_state_dict = {}
        with safe_open(hf_hub_download(repo_id=model_name, filename="s3gen.safetensors", revision=None), framework="pt") as f:
            for key in f.keys():
                detokenizer_state_dict[key] = f.get_tensor(key)
        self.audio_tokenizer.load_state_dict(detokenizer_state_dict, strict=False)
        self.audio_tokenizer.to(device).eval()

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(model_name)

        self.default_conds = Conditionals.load(
            hf_hub_download(repo_id=model_name, filename="conds.pt", revision=None), 
            map_location=device,
        )

        self._num_attention_heads = self.model.config.num_attention_heads
        self._num_key_value_heads = self.model.config.num_key_value_heads
        self._num_hidden_layers = self.model.config.num_hidden_layers
        self._hidden_size = self.model.config.hidden_size

        self.start_token_id = self.config.start_text_token
        self.stop_token_id = self.config.stop_text_token
        self.start_speech_token_id = self.config.start_speech_token
        self.stop_speech_token_id = self.config.stop_speech_token

        self.speech_vocab_size = 6561
        self.S3GEN_SR = 24000
        self.S3_SR = 16000

        self.default_sampling_config = SamplingConfig(
            top_k=None,
            top_p=0.95,
            min_p=None,
            temperature=0.8,
            repetition_penalty=1.2,
            repetition_window=-1,
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
        return 3

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 200

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return 21120
    
    @property
    def supports_audio_input(self) -> bool:
        """Indicates if the model accepts audio input."""
        return True

    @property
    def needs_watermarking(self) -> bool:
        """Indicates if the model requires watermarking."""
        return False # TODO!!

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return True

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return True

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.model.config.speech_tokens_dict_size

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.stop_token_id

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        tokenizer = Tokenizer.from_file(
            hf_hub_download(repo_id=tokenizer_path, filename="tokenizer.json", revision=None)
        )
        assert "[START]" in tokenizer.get_vocab()
        assert "[STOP]" in tokenizer.get_vocab()
        return tokenizer

    def _punc_norm(self, text: str) -> str:
        if len(text) == 0:
            return "You need to add some text for me to talk."

        # Capitalise first letter
        if text[0].islower():
            text = text[0].upper() + text[1:]

        # Remove multiple space chars
        text = " ".join(text.split())

        # Replace uncommon/llm punc
        punc_to_replace = [
            ("...", ", "),
            ("…", ", "),
            (":", ","),
            (" - ", ", "),
            (";", ", "),
            ("—", "-"),
            ("–", "-"),
            (" ,", ","),
            ("“", "\""),
            ("”", "\""),
            ("‘", "'"),
            ("’", "'"),
        ]
        for old_char_sequence, new_char in punc_to_replace:
            text = text.replace(old_char_sequence, new_char)

        # Add full stop if no ending punc
        text = text.rstrip(" ")
        sentence_enders = {".", "!", "?", "-", ","}
        if not any(text.endswith(p) for p in sentence_enders):
            text += "."

        return text

    def _prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=self.S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=self.S3GEN_SR, target_sr=self.S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, self.S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=self.S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        return Conditionals(t3_cond, s3gen_ref_dict)

    def preprocess(
        self,
        prompt: str = None,
        audio_path: str = None,
        exaggeration: int = None,
    ) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to Chatterbox specifications."""
        if audio_path:
            # conds = self._prepare_conditionals(audio_path)
            raise NotImplementedError("Audio conditioning not implemented yet.")
        else:
            conds = self.default_conds.t3.to(dtype=self.dtype)
        
        # TODO: exaggeration handling
        # TODO: cfg handling

        if conds.cond_prompt_speech_tokens is not None and conds.cond_prompt_speech_emb is None:
            conds.cond_prompt_speech_emb = self.model.speech_emb(conds.cond_prompt_speech_tokens) + \
                self.model.speech_pos_emb(
                    torch.arange(0, conds.cond_prompt_speech_tokens.shape[1], device=self.device)
                )

        cond_emb = self.model.cond_enc(conds)[0]
        
        prompt = self._punc_norm(prompt)
        prompt = prompt.replace(" ", "[SPACE]")
        input_ids = self.text_tokenizer.encode(prompt)
        input_ids = [0] * cond_emb.shape[0] + [self.start_token_id] + input_ids.ids + [
            self.stop_token_id, 
            self.start_speech_token_id,
            self.start_speech_token_id, # following official implementation
        ]
        input_ids = torch.tensor(input_ids, device=self.device).view(-1, 1)

        # 1 for audio, 0 for else
        input_masks = torch.zeros(
            input_ids.shape[0],
            self.n_codebooks,
            device=self.device,
            dtype=torch.bool,
        )

        input_features = torch.zeros(
            input_ids.shape[0],
            self.model.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        input_features[:cond_emb.shape[0]] = cond_emb

        text_embeds = self.model.text_emb(
            torch.clamp(
                input_ids[cond_emb.shape[0]:-2, 0], 0, self.model.config.text_tokens_dict_size - 1,
            )
        )
        text_embeds = text_embeds + self.model.text_pos_emb(torch.arange(0, text_embeds.shape[0], device=self.device))
        input_features[cond_emb.shape[0] : cond_emb.shape[0] + text_embeds.shape[0]] = text_embeds

        audio_embeds = self.model.speech_emb(torch.clamp(input_ids[-2:, 0], 0, self.model.config.speech_tokens_dict_size - 1))
        audio_embeds = audio_embeds + self.model.speech_pos_emb(torch.zeros(2, device=self.device, dtype=torch.long))
        input_features[-2:] = audio_embeds

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

        return PreprocessOutput(
            input_tokens=input_ids, 
            repetition_cache=repetition_cache,
            input_masks=input_masks,
            input_features=input_features,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor,
        input_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # text_embeds = self.model.text_emb(torch.clamp(input_ids[:, 0], 0, self.model.config.text_tokens_dict_size - 1))
        # text_embeds = text_embeds + self.model.text_pos_emb(position_ids)
        
        audio_embeds = self.model.speech_emb(torch.clamp(input_ids[:, 0], 0, self.model.config.speech_tokens_dict_size - 1))
        audio_embeds = audio_embeds + self.model.speech_pos_emb(torch.clamp(position_ids, 0, self.model.config.max_speech_tokens))
        # NOTE (keisuke): the position id here is not actually correct. For TTS task, we should do use position_ids - prompt_len,
        # since the position id count is independent for text and speech tokens in Chatterbox model.

        inputs_embeds = torch.where(input_masks, audio_embeds, input_features)

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

            req.input_masks = req.input_masks[:1].zero_() + 1
            req.input_features = req.input_features[:1].zero_()

        async def update_req_states():
            stop_mask = output_ids[:, 0] == self.stop_speech_token_id
            # SOS: speech_vocab_size, EOS: speech_vocab_size + 1
            audio_mask = (output_ids[:, 0] != self.speech_vocab_size) & \
                         (output_ids[:, 0] != self.speech_vocab_size + 1) & \
                         (output_ids[:, 0] < self.speech_vocab_size)

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
        """Convert token IDs to audio bytes."""
        # TODO: currently lacking the way to have request-specific ref_dict
        audio_tensor = self.audio_tokenizer.decode(
            token_ids[:, :, 0],
            speech_token_lens=self.detokenize_interval,
            ref_dict=self.default_conds.gen,
        )
        return audio_tensor[:, None, :]
