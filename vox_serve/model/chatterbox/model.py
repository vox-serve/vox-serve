from dataclasses import dataclass
from typing import Union

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from tokenizers import Tokenizer
from torch import nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN

from ...encoder.chatterbox import ChatterboxCondEnc, T3Cond, VoiceEncoder
from ...flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ...sampling import SamplingConfig
from ...tokenizer.chatterbox import ChatterboxRefEncoder
from ...tokenizer.cosyvoice2 import CosyVoice2Decoder
from ..base import BaseLM


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
        audio_decoder_device=None,
    ):
        if model_name == "chatterbox":
            model_name = "ResembleAI/chatterbox"
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.model_name = model_name
        self.config = ChatterboxConfig.english_only() if "chatterbox" in model_name else ChatterboxConfig.multilingual()
        self.model = ChatterboxForCausalLM(self.config)
        # S3Gen: CosyVoice2's flow + HiFT decoder, plus the encoders of reference clips.
        self.audio_decoder = CosyVoice2Decoder(None, self.audio_decoder_device, shared_prompt_cache_mode=True)
        self.ref_encoder = ChatterboxRefEncoder()
        self.voice_encoder = VoiceEncoder()

        state_dict = {}
        with safe_open(
            hf_hub_download(repo_id=model_name, filename="t3_cfg.safetensors", revision=None),
            framework="pt"
        ) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(dtype).to(device)

        # Audio decoder on its own device (may differ from the main one)
        with safe_open(hf_hub_download(repo_id=model_name, filename="s3gen.safetensors"), framework="pt") as f:
            s3gen = {key: f.get_tensor(key) for key in f.keys()}
        self.audio_decoder.load_s3gen(s3gen)
        self.ref_encoder.load_s3gen(s3gen)
        self.ref_encoder.to(self.audio_decoder_device).eval()
        with safe_open(hf_hub_download(repo_id=model_name, filename="ve.safetensors"), framework="pt") as f:
            self.voice_encoder.load_state_dict({key: f.get_tensor(key) for key in f.keys()}, strict=True)
        self.voice_encoder.to(device).eval()

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
        return 28

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 3

    @property
    def max_tokens(self) -> int:
        """Token limit of a request, prompt included (``--max-tokens`` overrides it): about 1000 generated
        tokens (40 s) after a typical prompt, the reference's ``max_new_tokens``."""
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 1200

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) of a detokenized chunk: its first ``interval - overlap`` tokens."""
        return 24000

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.model.config.speech_tokens_dict_size

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

    # Reference clip limits of the original implementation (ChatterboxTTS.prepare_conditionals).
    ENC_COND_LEN = 6 * 16000  # T3's conditioning prompt: the first 6 s at 16 kHz
    DEC_COND_LEN = 10 * 24000  # S3Gen's reference: the first 10 s at 24 kHz

    @torch.inference_mode()
    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> Conditionals:
        """T3 and S3Gen conditioning of a reference clip, as ``ChatterboxTTS.prepare_conditionals``."""
        import librosa

        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=self.S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=self.S3GEN_SR, target_sr=self.S3_SR)
        gen = self.ref_encoder.embed_ref(s3gen_ref_wav[: self.DEC_COND_LEN], self.S3GEN_SR)

        plen = self.config.speech_cond_prompt_len
        tok = self.ref_encoder.tokenizer
        prompt_tokens, _ = tok.forward([torch.from_numpy(ref_16k_wav[: self.ENC_COND_LEN])], max_len=plen)
        ve_embed = torch.from_numpy(self.voice_encoder.embeds_from_wavs([ref_16k_wav], sample_rate=self.S3_SR))
        t3 = T3Cond(
            speaker_emb=ve_embed.mean(axis=0, keepdim=True).to(self.device),
            cond_prompt_speech_tokens=torch.atleast_2d(prompt_tokens).to(self.device),
            emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device),
        )
        return Conditionals(t3, gen)
