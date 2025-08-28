import json
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Iterable, List, Literal

import flashinfer
import inflect
import safetensors
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from phonemizer.backend import EspeakBackend
from torch import nn
from torch.nn import functional as F

from ..encoder.zonos import ZonosSpeakerEmbeddingLDA
from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.dac import DAC
from ..utils import get_logger
from .base import BaseLM, PreprocessOutput


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
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        query_states, key_states, value_states = self.in_proj(hidden_states).split([q_size, kv_size, kv_size], dim=-1)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)  # .transpose(0, 1)
        key_states = key_states.view(-1, self.num_heads_kv, self.head_dim)  # .transpose(0, 1)
        value_states = value_states.view(-1, self.num_heads_kv, self.head_dim)  # .transpose(0, 1)

        query_states, key_states = flashinfer.rope.apply_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_theta=10000,
            interleave=True,
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
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states = self.mixer(
            hidden_states=hidden_states,
            position_ids=position_ids,
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

        self.layers = nn.ModuleList([ZonosDecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

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


class ZonosUtils:
    _inflect = inflect.engine()
    _comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
    _decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
    _pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
    _dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
    _ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
    _number_re = re.compile(r"[0-9]+")

    PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
    SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

    _punctuation = ';:,.!?¡¿—…"«»""() *~-/\\&'
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = (
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    )

    symbols = [*_punctuation, *_letters, *_letters_ipa]
    _symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}

    @staticmethod
    def _remove_commas(m: re.Match) -> str:
        return m.group(1).replace(",", "")

    @staticmethod
    def _expand_decimal_point(m: re.Match) -> str:
        return m.group(1).replace(".", " point ")

    @classmethod
    def _expand_dollars(cls, m: re.Match) -> str:
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

    @classmethod
    def _expand_ordinal(cls, m: re.Match) -> str:
        return cls._inflect.number_to_words(m.group(0))

    @classmethod
    def _expand_number(cls, m: re.Match) -> str:
        num = int(m.group(0))
        if num > 1000 and num < 3000:
            if num == 2000:
                return "two thousand"
            elif num > 2000 and num < 2010:
                return "two thousand " + cls._inflect.number_to_words(num % 100)
            elif num % 100 == 0:
                return cls._inflect.number_to_words(num // 100) + " hundred"
            else:
                return cls._inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
        else:
            return cls._inflect.number_to_words(num, andword="")

    @classmethod
    def normalize_numbers(cls, text: str) -> str:
        text = re.sub(cls._comma_number_re, cls._remove_commas, text)
        text = re.sub(cls._pounds_re, r"\1 pounds", text)
        text = re.sub(cls._dollars_re, cls._expand_dollars, text)
        text = re.sub(cls._decimal_number_re, cls._expand_decimal_point, text)
        text = re.sub(cls._ordinal_re, cls._expand_ordinal, text)
        text = re.sub(cls._number_re, cls._expand_number, text)
        return text

    @classmethod
    def _get_symbol_id(cls, s: str) -> int:
        return cls._symbol_to_id.get(s, 1)

    @classmethod
    def get_symbol_ids(cls, text: str) -> list[int]:
        return list(map(cls._get_symbol_id, text))

    @classmethod
    def tokenize_phonemes(cls, phonemes: list[str]) -> tuple[torch.Tensor, list[int]]:
        phoneme_ids = [[cls.BOS_ID, *cls.get_symbol_ids(phonemes), cls.EOS_ID] for phonemes in phonemes]
        lengths = list(map(len, phoneme_ids))
        longest = max(lengths)
        phoneme_ids = [[cls.PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
        return torch.tensor(phoneme_ids), lengths

    # @classmethod
    # def normalize_jp_text(cls, text: str, tokenizer) -> str:
    #     from kanjize import number2kanji

    #     text = unicodedata.normalize("NFKC", text)
    #     text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    #     final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    #     return final_text

    @classmethod
    def clean(cls, texts: list[str], languages: list[str]) -> list[str]:
        texts_out = []
        for text, language in zip(texts, languages, strict=False):
            if "ja" in language:
                raise NotImplementedError(
                    "Japanese text normalization is not implemented"
                )

                # from sudachipy import Dictionary
                # text = cls.normalize_jp_text(text, tokenizer=Dictionary(dict="full").create())
            else:
                text = cls.normalize_numbers(text)
            texts_out.append(text)
        return texts_out

    @classmethod
    @cache
    def get_backend(cls, language: str) -> "EspeakBackend":
        import logging

        from phonemizer.backend import EspeakBackend

        logger = logging.getLogger("phonemizer")
        backend = EspeakBackend(
            language,
            preserve_punctuation=True,
            with_stress=True,
            punctuation_marks=cls._punctuation,
            logger=logger,
        )
        logger.setLevel(logging.ERROR)
        return backend

    @classmethod
    def phonemize(cls, texts: list[str], languages: list[str]) -> list[str]:
        texts = cls.clean(texts, languages)

        batch_phonemes = []
        for text, language in zip(texts, languages, strict=False):
            backend = cls.get_backend(language)
            phonemes = backend.phonemize([text], strip=True)
            batch_phonemes.append(phonemes[0])

        return batch_phonemes


class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(ZonosUtils.SPECIAL_TOKEN_IDS) + len(ZonosUtils.symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes = ZonosUtils.phonemize(texts, languages)
        phoneme_ids, _ = ZonosUtils.tokenize_phonemes(phonemes)
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


class ZonosPrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(self.build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

    def build_conditioners(self, conditioners: list[dict], output_dim: int) -> list[Conditioner]:
        _cond_cls_map = {
            "PassthroughConditioner": PassthroughConditioner,
            "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
            "FourierConditioner": FourierConditioner,
            "IntegerConditioner": IntegerConditioner,
        }
        return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]

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
        if model_name == "zonos":
            model_name = "Zyphra/Zonos-v0.1-transformer"
        super().__init__(model_name, device, dtype)
        self.logger = get_logger(__name__)
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
                        raise ValueError(
                            f"Shape mismatch for weight '{k}': "
                            f"expected {self.model.state_dict()[k].shape}, got {f.get_tensor(k).shape}"
                        )
                    self.model.state_dict()[k].copy_(f.get_tensor(k))
                else:
                    raise KeyError(f"Missing weight '{k}' in the model state_dict")
            # Check for missing weights in the checkpoint that are not in the model
            missing_in_model = set(f.keys()) - set(self.model.state_dict().keys())
            if missing_in_model:
                raise KeyError(
                    f"The following weights are present in the checkpoint but missing in the model: {missing_in_model}"
                )

        dim = self.config.backbone.d_model
        self.eos_token_id = self.config.eos_token_id
        self.masked_token_id = self.config.masked_token_id

        self.spk_clone_model = None

        self._num_attention_heads = self.model.config.backbone.attn_cfg["num_heads"]
        self._num_key_value_heads = self.model.config.backbone.attn_cfg["num_heads_kv"]
        self._num_hidden_layers = self.model.config.backbone.n_layer
        self._hidden_size = self.model.config.backbone.d_model
        # self.vocab_size = 1025 # mask token is not included here since it's not predicted by LM heads

        self._n_codebooks = self.model.autoencoder.num_codebooks
        self.logit_bias = torch.zeros(self.n_codebooks, 1025, dtype=dtype, device=device)
        self.logit_bias[1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS

        self.resample_44k_to_24k = torchaudio.transforms.Resample(orig_freq=44100, new_freq=24000).to(self.device)

        self.default_sampling_config = SamplingConfig(
            top_k=None,
            top_p=None,
            min_p=0.1,
            temperature=1.0,
            repetition_penalty=3.0,
            repetition_window=2,
            cfg_scale=None,
        )

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
        return 50

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return self._n_codebooks

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return 11425

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        return 2048

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return 1025

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return True

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return True

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] == self.eos_token_id

    def _get_default_speaker_embedding(self) -> torch.Tensor:
        from ..utils import download_github_file

        try:
            wav, sr = torchaudio.load(download_github_file("Zyphra", "Zonos", "assets/exampleaudio.mp3"))
            _, spk_embedding = self.speaker_encoder(wav.to(self.device), sr)
            self.default_speaker_embedding = spk_embedding.unsqueeze(0).bfloat16()
        except Exception as e:
            self.logger.error(f"Failed to load default speaker embedding: {e}")
            self.logger.error("Using random embedding instead.")
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

    def preprocess(self, prompt: str = None, audio_path: str = None) -> PreprocessOutput:
        """Prepare the prompt for the model, formatting it according to Orpheus specifications."""
        # TODO: add API support for custom voice
        assert audio_path is None
        cond_dict = self._make_cond_dict(text=prompt)
        input_features = self._prepare_conditioning(cond_dict)

        prefix_tokens = torch.cat(
            [
                torch.zeros(input_features.shape[0], self.n_codebooks, dtype=torch.long),
                torch.full((1, self.n_codebooks), self.masked_token_id, dtype=torch.long),
            ],
            dim=0,
        )

        # Create expanded input_features to match token sequence length
        # Repeat each feature embedding for all positions except the last one
        seq_len = prefix_tokens.shape[0]
        expanded_features = torch.zeros(
            seq_len, input_features.shape[-1],
            device=input_features.device,
            dtype=input_features.dtype
        )
        expanded_features[:-1] = input_features.squeeze(0)  # Fill all positions except last with features

        # Create input mask for feature positions (all positions except the last one)
        input_mask = torch.ones(seq_len, dtype=torch.bool, device=input_features.device)
        input_mask[-1] = False  # Last position should not be filled with features
        input_mask = input_mask[:, None].expand(-1, self.n_codebooks) # Expand for n_codebooks

        repetition_cache = torch.zeros(
            self.default_sampling_config.repetition_window if self.default_sampling_config.repetition_window > 0 else 1,
            self.n_codebooks,
            self.vocab_size,
            dtype=torch.bool,
            device=self.device,
        )

        return PreprocessOutput(
            input_tokens=prefix_tokens.tolist(),
            input_features=expanded_features,
            input_masks=input_mask,
            repetition_cache=repetition_cache
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_features: torch.Tensor | None = None,
        input_masks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        inputs_embeds = self.model.embed_tokens(input_ids)

        # for prefill requests, except for the last token, replace embeddings with input_features
        inputs_embeds = torch.where(input_masks[:, :1], input_features, inputs_embeds)

        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        logits += self.logit_bias

        return logits

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        cfg_scale: float | None = None,
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
            if req.repetition_cache is not None:
                Sampler.update_repetition_penalty_cache(
                    req.repetition_cache,
                    output_ids[i],
                    sampling_params.repetition_window,
                )

            # mask part of the output tokens for the first n_codebooks - 1 tokens
            # use len(req.lm_output_tokens) + 1 since the output is not yet appended
            if len(req.lm_output_tokens) + 1 < self.n_codebooks:
                for j in range(len(req.lm_output_tokens) + 1, self.n_codebooks):
                    output_ids[i, j] = self.masked_token_id

            # for decode phase, input_features are not used
            req.input_features = torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.bfloat16)
            req.input_masks = torch.zeros(1, self.n_codebooks, dtype=torch.bool, device=self.device)

            # no additional logic for CSM model for now. TODO: revert delay patterns here?
            req.lm_output_tokens.append(output_ids[i].tolist())

            if not self.is_stop_id(output_ids[i].tolist()):
                # Don't add the EOS token to lm_output_audio_tokens
                req.lm_output_audio_tokens.append(output_ids[i].tolist())

        return output_ids

    def postprocess(self, token_ids: torch.Tensor):
        interval = token_ids.shape[1]
        n_q = token_ids.shape[2]

        # revert delay patterns
        codes = torch.stack([token_ids[:, k : interval - n_q + k, k] for k in range(n_q)], dim=1)

        wavs = self.model.autoencoder.decode(codes)
        # audio_tensor = torchaudio.functional.resample(wavs, orig_freq=44100, new_freq=24000)
        audio_tensor = self.resample_44k_to_24k(wavs)

        return audio_tensor
