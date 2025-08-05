from functools import cache
from typing import Any, Callable, List, Optional, Tuple, Union, Iterable
import numpy as np
import torch
import re
import unicodedata
import json 

import inflect
from phonemizer.backend import EspeakBackend
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import LlamaConfig, LlamaPreTrainedModel
from huggingface_hub import hf_hub_download
import safetensors

from ..tokenizer.dac import DAC
from ..encoder.zonos import ZonosSpeakerEmbeddingLDA
from ..flashinfer_utils import FlashInferWrapper
from ..sampling import min_p_sampling, SamplingConfig
from .base import BaseLM


from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class BackboneConfig:
    d_model: int = 1024
    d_intermediate: int = 0
    attn_mlp_d_intermediate: int = 0
    n_layer: int = 16
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = False
    residual_in_fp32: bool = False
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000  # Default value for RoPE theta
    head_dim: int = 0  # Will be set based on d_model and num_heads in the model config


@dataclass
class PrefixConditionerConfig:
    conditioners: list[dict]
    projection: Literal["none", "linear", "mlp"]


@dataclass
class ZonosConfig:
    backbone: BackboneConfig
    prefix_conditioner: PrefixConditionerConfig
    eos_token_id: int = 1024
    masked_token_id: int = 1025
    pad_vocab_to_multiple_of: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ZonosConfig":
        d = d.copy()
        backbone_config = BackboneConfig(**d.pop("backbone"))
        prefix_conditioner_config = PrefixConditionerConfig(**d.pop("prefix_conditioner"))
        config = cls(backbone_config, prefix_conditioner_config, **d)
        return config


class ZonosRotaryEmbedding(nn.Module):
    def __init__(self, config: BackboneConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = getattr(config, "max_position_embeddings", 16384)
        self.original_max_seq_len = getattr(config, "max_position_embeddings", 16384)

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.config.rope_theta = 10000 
        self.config.head_dim = config.d_model // config.attn_cfg["num_heads"]

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


class ZonosMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        y = self.fc2(y * F.silu(gate))
        return y


class ZonosAttention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        # self.num_key_value_groups = config.num_heads // config.num_key_value_heads
        # self.scaling = self.head_dim**-0.5
        # self.attention_dropout = config.attention_dropout
        # self.is_causal = True

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

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

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        query_states, key_states, value_states = self.in_proj(hidden_states).split([q_size, kv_size, kv_size], dim=-1)

        query_states = query_states.view(-1, self.num_heads, self.head_dim) #.transpose(0, 1)
        key_states = key_states.view(-1, self.num_heads_kv, self.head_dim) #.transpose(0, 1)
        value_states = value_states.view(-1, self.num_heads_kv, self.head_dim) #.transpose(0, 1)

        # cos, sin = position_embeddings
        import flashinfer
        query_states, key_states = flashinfer.rope.apply_rope_pos_ids(
            query_states, key_states, pos_ids=position_ids, rope_theta=10000, interleave=True,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class ZonosDecoderLayer(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.config = config 

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = ZonosAttention(config=config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = ZonosMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states = self.mixer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class ZonosBackboneModel(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [ZonosDecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.rotary_emb = ZonosRotaryEmbedding(config=config)

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

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text



PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [*_punctuation, *_letters, *_letters_ipa]
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


def _get_symbol_id(s: str) -> int:
    return _symbol_to_id.get(s, 1)


def get_symbol_ids(text: str) -> list[int]:
    return list(map(_get_symbol_id, text))


def tokenize_phonemes(phonemes: list[str]) -> tuple[torch.Tensor, list[int]]:
    phoneme_ids = [[BOS_ID, *get_symbol_ids(phonemes), EOS_ID] for phonemes in phonemes]
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths)
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    return torch.tensor(phoneme_ids), lengths


def normalize_jp_text(text: str, tokenizer) -> str:
    from kanjize import number2kanji
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    return final_text


def clean(texts: list[str], languages: list[str]) -> list[str]:
    texts_out = []
    for text, language in zip(texts, languages):
        if "ja" in language:
            from sudachipy import Dictionary, SplitMode
            text = normalize_jp_text(text, tokenizer=Dictionary(dict="full").create())
        else:
            text = normalize_numbers(text)
        texts_out.append(text)
    return texts_out


@cache
def get_backend(language: str) -> "EspeakBackend":
    import logging

    from phonemizer.backend import EspeakBackend

    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )
    logger.setLevel(logging.ERROR)
    return backend


def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    texts = clean(texts, languages)

    batch_phonemes = []
    for text, language in zip(texts, languages):
        backend = get_backend(language)
        phonemes = backend.phonemize([text], strip=True)
        batch_phonemes.append(phonemes[0])

    return batch_phonemes


class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes = phonemize(texts, languages)
        phoneme_ids, _ = tokenize_phonemes(phonemes)
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        return phoneme_embeds


class FourierConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs,
    ):
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        x = (x - self.min_val) / (self.max_val - self.min_val)  # [batch_size, seq_len, input_dim]
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # [batch_size, seq_len, output_dim // 2]
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]


class IntegerConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        return self.int_embedder(x.squeeze(-1) - self.min_val)  # [batch_size, seq_len, output_dim]


class PassthroughConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cond_dim
        return x


_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
    "FourierConditioner": FourierConditioner,
    "IntegerConditioner": IntegerConditioner,
}


def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]


class ZonosPrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def forward(self, cond_dict: dict) -> torch.Tensor:
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")
        conds = []
        for conditioner in self.conditioners:
            conds.append(conditioner(cond_dict.get(conditioner.name)))
        max_bsz = max(map(len, conds))
        assert all(c.shape[0] in (max_bsz, 1) for c in conds)
        conds = [c.expand(max_bsz, -1, -1) for c in conds]
        return self.norm(self.project(torch.cat(conds, dim=-2)))


supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]  # fmt: off


class ZonosForCausalLM(nn.Module):

    def __init__(self, config: ZonosConfig):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model

        self.autoencoder = DAC()

        self.backbone = ZonosBackboneModel(config.backbone)
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.prefix_conditioner = ZonosPrefixConditioner(config.prefix_conditioner, dim)
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

    def embed_tokens(self, input_ids):
        return sum(emb(input_ids[:, i]) for i, emb in enumerate(self.embeddings))
    
    # def get_input_embeddings(self):
    #     return self.model.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
    ):
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        logits = torch.stack([head(outputs) for head in self.heads], dim=1) 

        return logits



class ZonosModel(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0"):
        # TODO: Zonos hybrid model is not yet supported
        super().__init__(model_name, device, dtype)
        config_path = hf_hub_download(repo_id=model_name, filename="config.json", revision=None)
        model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors", revision=None)

        self.config = ZonosConfig.from_dict(json.load(open(config_path)))
        self.model = ZonosForCausalLM(self.config)
        self.model.to(dtype).to(device)
        self.model.autoencoder.dac.to(dtype).to(device)
        self.speaker_encoder = ZonosSpeakerEmbeddingLDA(device=device)
        self.device = device

        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                if k in self.model.state_dict():
                    if self.model.state_dict()[k].shape != f.get_tensor(k).shape:
                        raise ValueError(f"Shape mismatch for weight '{k}': expected {self.model.state_dict()[k].shape}, got {f.get_tensor(k).shape}")
                    self.model.state_dict()[k].copy_(f.get_tensor(k))
                else:
                    raise KeyError(f"Missing weight '{k}' in the model state_dict")
            # Check for missing weights in the checkpoint that are not in the model
            missing_in_model = set(f.keys()) - set(self.model.state_dict().keys())
            if missing_in_model:
                raise KeyError(f"The following weights are present in the checkpoint but missing in the model: {missing_in_model}")

        dim = self.config.backbone.d_model
        self.eos_token_id = self.config.eos_token_id
        self.masked_token_id = self.config.masked_token_id

        self.spk_clone_model = None

        self._num_attention_heads = self.model.config.backbone.attn_cfg["num_heads"]
        self._num_key_value_heads = self.model.config.backbone.attn_cfg["num_heads_kv"]
        self._num_hidden_layers = self.model.config.backbone.n_layer
        self._hidden_size = self.model.config.backbone.d_model
        self.vocab_size = 1025 # mask token is not included here since it's not predicted by LM heads

        self._n_codebooks = self.model.autoencoder.num_codebooks
        self.logit_bias = torch.zeros(self.n_codebooks, 1025, dtype=dtype, device=device)
        self.logit_bias[1:, self.eos_token_id] = -torch.inf # only allow codebook 0 to predict EOS

        self.min_p = 0.1
        self.temperature = 1.0
        self.max_tokens = 1200 
        self.repetition_penalty = 3.0 
        self.repetition_penalty_window = 2
        self._detokenize_interval = 50

        self._get_default_speaker_embedding()

    @property
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        return self._n_codebooks

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
        return self._detokenize_interval
    
    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return self._n_codebooks

    def _get_default_speaker_embedding(self) -> torch.Tensor:
        from ..utils import download_github_file
        try:
            wav, sr = torchaudio.load(
                download_github_file("Zyphra", "Zonos", "assets/exampleaudio.mp3")
            )
            _, spk_embedding = self.speaker_encoder(wav.to(self.device), sr)
            self.default_speaker_embedding = spk_embedding.unsqueeze(0).bfloat16()
        except:
            print("Failed to load default speaker embedding, using random embedding instead.")
            self.default_speaker_embedding = None

    def _make_cond_dict(
        self,
        text: str = "It would be nice to have time for testing, indeed.",
        language: str = "en-us",
        speaker: torch.Tensor | None = None,
        
        # Emotion vector from 0.0 to 1.0
        #   Is entangled with pitch_std because more emotion => more pitch variation
        #                     VQScore and DNSMOS because they favor neutral speech
        #
        #                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
        emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],

        # Maximum frequency (0 to 24000), should be 22050 or 24000 for 44.1 or 48 kHz audio
        # For voice cloning use 22050
        fmax: float = 22050.0,
        
        # Standard deviation for pitch (0 to 400), should be 
        #   20-45 for normal speech, 
        #   60-150 for expressive speech, 
        #   higher values => crazier samples
        pitch_std: float = 20.0,

        # Speaking rate in phonemes per minute (0 to 40). 30 is very fast, 10 is slow.
        speaking_rate: float = 15.0,

        # Target VoiceQualityScore for the generated speech (0.5 to 0.8).
        #   A list of values must be provided which represent each 1/8th of the audio.
        #   You should unset for expressive speech.
        # According to discord Chat this is only used for the hybrid model
        vqscore_8: list[float] = [0.78] * 8,

        # CTC target loss
        # Only used for the hybrid model
        ctc_loss: float = 0.0,
        # Only used for the hybrid model
        dnsmos_ovrl: float = 4.0,
        # Only used for the hybrid model
        speaker_noised: bool = False,
        unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"},
        device: torch.device | str = "cuda",
    ) -> dict:
        """
        A helper to build the 'cond_dict' that the model expects.
        By default, it will generate a random speaker embedding
        """
        assert language.lower() in supported_language_codes, "Please pick a supported language"

        if speaker is None:
            speaker = self.default_speaker_embedding

        language_code_to_id = {lang: i for i, lang in enumerate(supported_language_codes)}

        cond_dict = {
            "espeak": ([text], [language]),
            "speaker": speaker,
            "emotion": emotion,
            "fmax": fmax,
            "pitch_std": pitch_std,
            "speaking_rate": speaking_rate,
            "language_id": language_code_to_id[language],
            "vqscore_8": vqscore_8,
            "ctc_loss": ctc_loss,
            "dnsmos_ovrl": dnsmos_ovrl,
            "speaker_noised": int(speaker_noised),
        }

        for k in unconditional_keys:
            cond_dict.pop(k, None)

        for k, v in cond_dict.items():
            if isinstance(v, (float, int, list)):
                v = torch.tensor(v)
            if isinstance(v, torch.Tensor):
                cond_dict[k] = v.view(1, 1, -1).to(device)

            if k == "emotion":
                cond_dict[k] /= cond_dict[k].sum(dim=-1)

        return cond_dict
    
    def _prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        return self.model.prefix_conditioner(cond_dict)[0]

        # TODO: uncond_dict for generation with CFG

        # if uncond_dict is None:
        #     uncond_dict = {k: cond_dict[k] for k in self.model.prefix_conditioner.required_keys}
        # return torch.cat(
        #     [
        #         self.model.prefix_conditioner(cond_dict),
        #         self.model.prefix_conditioner(uncond_dict),
        #     ]
        # )
    
    def preprocess(self, prompt: str):
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        # TODO: add API support for custom voice
        cond_dict = self._make_cond_dict(text=prompt)
        input_features = self._prepare_conditioning(cond_dict)

        prefix_tokens = torch.cat([
            torch.zeros(input_features.shape[0], self.n_codebooks, dtype=torch.long),
            torch.full((1, self.n_codebooks), self.masked_token_id, dtype=torch.long), 
        ], dim=0)

        preprocess_dict = {
            "input_features": input_features,
            "repetition_cache": torch.zeros(
                self.repetition_penalty_window, self.n_codebooks, self.vocab_size, 
                dtype=torch.bool, device=self.device
            ),
        }

        return prefix_tokens.tolist(), preprocess_dict

    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        attn_wrapper: FlashInferWrapper, 
        kv_cache: torch.Tensor,
        input_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        inputs_embeds = self.model.embed_tokens(input_ids)

        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            # includes prefill request
            for i, feature in enumerate(input_features):
                if feature is None:
                    continue 
                
                # In Zonos, the prompt tokens except for the last one are filled with feature embeddings
                inputs_embeds[attn_wrapper.qo_indptr[i] : attn_wrapper.qo_indptr[i + 1] - 1] = feature
        
        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        logits += self.logit_bias

        if getattr(attn_wrapper, "qo_indptr", None) is not None:
            logits = logits[attn_wrapper.qo_indptr[:-1] - 1]

        return logits
    
    def sampling(
        self, 
        logits: torch.Tensor, 
        sampling_params: List[SamplingConfig] | None = None,
        repetition_cache: List[torch.Tensor] | None = None, 
        cfg_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # apply repetition penalty
        for i, cache in enumerate(repetition_cache):
            if cache is None:
                continue
            # OR operation over window_size dimension. 
            # TODO: penalty accumulation logic
            appearance_mask = cache.any(dim=0)
            logits[i] = torch.where(
                (logits[i] > 0) & appearance_mask, 
                logits[i] / self.repetition_penalty, 
                logits[i],
            )
            logits[i] = torch.where(
                (logits[i] <= 0) & appearance_mask, 
                logits[i] * self.repetition_penalty, 
                logits[i],
            )

        output_ids = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.long, device=self.device)
        for i in range(self.n_codebooks):
            output_ids[:, i] = min_p_sampling(logits[:, i], min_p=self.min_p, temperature=self.temperature)
        
        # update repetition cache
        for i, cache in enumerate(repetition_cache):
            if cache is None:
                continue
            # shift the cache to the left and add the new token
            cache[:-1] = cache[1:]
            cache[-1] = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            cache[-1, torch.arange(self.n_codebooks), output_ids[i]] = True

        return output_ids
    
    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.eos_token_id

    def postprocess(self, token_ids: torch.Tensor):
        interval = token_ids.shape[1]
        n_q = token_ids.shape[2]

        # revert delay patterns
        codes = torch.stack([token_ids[:, k : interval - n_q + k, k] for k in range(n_q)], dim=1)

        wavs = self.model.autoencoder.decode(codes)
        audio_tensor = torchaudio.functional.resample(wavs, orig_freq=44100, new_freq=24000)

        return audio_tensor