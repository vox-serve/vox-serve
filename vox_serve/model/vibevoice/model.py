"""VibeVoice-Realtime-0.5B modules and weights (``microsoft/VibeVoice-Realtime-0.5B``).

Adapted from microsoft/VibeVoice (MIT): ``vibevoice/modular/modeling_vibevoice_streaming*.py`` and
``modular_vibevoice_diffusion_head.py``. A Qwen2.5-0.5B split in two: the lower 4 layers (``lm``) encode
text only; the upper 20 (``tts``) see the lower layers' text hidden states and the generated speech latents
(through ``acoustic_connector``), each marked by a type embedding. A diffusion head turns the TTS hidden
state into the next 64-d acoustic latent (DPM-Solver++ with classifier-free guidance against a second,
"negative" TTS context that never sees the text), a binary classifier on the same hidden state ends the
speech, and the acoustic decoder (``vox_serve.tokenizer.vibevoice``) turns latents into 24 kHz audio.

Voices are not cloned from audio: the model ships voice prompts as precomputed KV caches (``.pt`` files of
the VibeVoice repository, ``demo/voices/streaming_model``) for its four contexts (lm, tts, and their
negative counterparts), downloaded on first use.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from ...tokenizer.vibevoice import VibeVoiceAcousticDecoder
from ...utils import download_github_file, get_logger

logger = get_logger(__name__)

REPO = "microsoft/VibeVoice-Realtime-0.5B"
VOICE_REPO = ("microsoft", "VibeVoice", "demo/voices/streaming_model")
# The voices of demo/voices/streaming_model (all loaded by default, about 4 MB each).
ENGLISH_VOICES = [
    "en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man", "en-Grace_woman", "en-Mike_man", "in-Samuel_man",
]  # fmt: skip
OTHER_VOICES = [
    f"{lang}-{spk}"
    for lang, spks in {
        "de": ("Spk0_man", "Spk1_woman"), "fr": ("Spk0_man", "Spk1_woman"), "it": ("Spk0_woman", "Spk1_man"),
        "jp": ("Spk0_man", "Spk1_woman"), "kr": ("Spk0_woman", "Spk1_man"), "nl": ("Spk0_man", "Spk1_woman"),
        "pl": ("Spk0_man", "Spk1_woman"), "pt": ("Spk0_woman", "Spk1_man"), "sp": ("Spk0_woman", "Spk1_man"),
    }.items()
    for spk in spks
]  # fmt: skip


class _Holder(nn.Module):
    def __init__(self, **children: nn.Module):
        super().__init__()
        for name, m in children.items():
            setattr(self, name, m)


class RMSNorm(nn.Module):
    """Qwen2 / Llama RMSNorm: fp32 statistics, weight applied in the input dtype."""

    def __init__(self, dim: int, eps: float, affine: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return y if self.weight is None else self.weight * y


class _HeadRMSNorm(RMSNorm):
    """The diffusion head's RMSNorm (``type_as`` before the weight, like the Qwen2 one)."""


class Qwen2Layer(nn.Module):
    """One Qwen2 decoder layer, split around attention so a caller can route rows to different KV caches."""

    def __init__(self, H: int, n_q: int, n_kv: int, head_dim: int, inter: int, eps: float):
        super().__init__()
        self.n_q, self.n_kv, self.hd = n_q, n_kv, head_dim
        self.self_attn = _Holder(
            q_proj=nn.Linear(H, n_q * head_dim),
            k_proj=nn.Linear(H, n_kv * head_dim),
            v_proj=nn.Linear(H, n_kv * head_dim),
            o_proj=nn.Linear(n_q * head_dim, H, bias=False),
        )
        self.mlp = _Holder(
            gate_proj=nn.Linear(H, inter, bias=False),
            up_proj=nn.Linear(H, inter, bias=False),
            down_proj=nn.Linear(inter, H, bias=False),
        )
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)

    def qkv(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h [n, H] -> q [n, n_q * hd], k, v [n, n_kv * hd] (before RoPE)."""
        x = self.input_layernorm(h)
        a = self.self_attn
        return a.q_proj(x), a.k_proj(x), a.v_proj(x)

    def finish(self, h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Residual after attention (attn [n, n_q * hd]) and the MLP."""
        h = h + self.self_attn.o_proj(attn)
        m = self.mlp
        x = self.post_attention_layernorm(h)
        return h + m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))


class SpeechConnector(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.fc1 = nn.Linear(din, dout)
        self.norm = RMSNorm(dout, 1e-6)
        self.fc2 = nn.Linear(dout, dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


class BinaryClassifier(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.fc1 = nn.Linear(H, H)
        self.fc2 = nn.Linear(H, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _HeadLayer(nn.Module):
    def __init__(self, H: int, ffn: int, eps: float):
        super().__init__()
        self.ffn = _Holder(
            gate_proj=nn.Linear(H, ffn, bias=False),
            up_proj=nn.Linear(H, ffn, bias=False),
            down_proj=nn.Linear(ffn, H, bias=False),
        )
        self.norm = _HeadRMSNorm(H, eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 3 * H, bias=False))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        y = self.norm(x) * (1 + scale) + shift
        f = self.ffn
        return x + gate * f.down_proj(F.silu(f.gate_proj(y)) * f.up_proj(y))


class DiffusionHead(nn.Module):
    """``VibeVoiceDiffusionHead``: v-prediction for the latent given the TTS hidden state."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        H, latent = int(cfg["hidden_size"]), int(cfg["latent_size"])
        eps = float(cfg["rms_norm_eps"])
        self.noisy_images_proj = nn.Linear(latent, H, bias=False)
        self.cond_proj = nn.Linear(H, H, bias=False)
        mlp = nn.Sequential(nn.Linear(256, H, bias=False), nn.SiLU(), nn.Linear(H, H, bias=False))
        self.t_embedder = _Holder(mlp=mlp)
        ffn = int(H * float(cfg["head_ffn_ratio"]))
        self.layers = nn.ModuleList([_HeadLayer(H, ffn, eps) for _ in range(int(cfg["head_layers"]))])
        self.final_layer = _Holder(
            norm_final=_HeadRMSNorm(H, eps, affine=False),
            linear=nn.Linear(H, latent, bias=False),
            adaLN_modulation=nn.Sequential(nn.SiLU(), nn.Linear(H, 2 * H, bias=False)),
        )

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Timesteps [n] (in the model dtype, as the reference passes them) -> [n, H]."""
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)
        return self.t_embedder.mlp(emb)

    def forward(self, noisy: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """noisy [n, latent], c = cond_proj(condition) + t_embed(t) [n, H] -> [n, latent]."""
        x = self.noisy_images_proj(noisy)
        for layer in self.layers:
            x = layer(x, c)
        fl = self.final_layer
        shift, scale = fl.adaLN_modulation(c).chunk(2, dim=-1)
        return fl.linear(fl.norm_final(x) * (1 + scale) + shift)


def dpm_solver_plan(cfg: Dict[str, Any], steps: int) -> List[Dict[str, float]]:
    """Per-step constants of the reference sampler: diffusers' ``DPMSolverMultistepScheduler`` (DPM-Solver++
    2M, midpoint, v-prediction, cosine betas, ``lower_order_final``, final sigma 0) after
    ``set_timesteps(steps)``, computed with the scheduler's own fp32 tensors.

    Each step k: ``x0 = alpha * x - sigma * v``; first order (k = 0 and the last step):
    ``x' = a * x - b * x0``; second order: ``x' = a * x - b * x0 - 0.5 * b * (x0 - x0_prev) / r``."""
    from diffusers import DPMSolverMultistepScheduler

    beta = cfg["ddpm_beta_schedule"]
    s = DPMSolverMultistepScheduler(
        num_train_timesteps=int(cfg["ddpm_num_steps"]),
        beta_schedule="squaredcos_cap_v2" if beta == "cosine" else beta,
        prediction_type=cfg["prediction_type"],
    )
    s.set_timesteps(steps)
    sig = s.sigmas
    plan = []
    for k in range(steps):
        alpha_s, sigma_s = s._sigma_to_alpha_sigma_t(sig[k])
        alpha_t, sigma_t = s._sigma_to_alpha_sigma_t(sig[k + 1])
        lam = lambda a, b: torch.log(a) - torch.log(b)  # noqa: E731
        h = lam(alpha_t, sigma_t) - lam(alpha_s, sigma_s)
        step = {
            "t": float(s.timesteps[k]),
            "alpha": float(alpha_s),
            "sigma": float(sigma_s),
            "a": float(sigma_t / sigma_s),
            "b": float(alpha_t * (torch.exp(-h) - 1.0)),
            "order": 1 if k == 0 or k == steps - 1 else 2,
        }
        if step["order"] == 2:
            alpha_p, sigma_p = s._sigma_to_alpha_sigma_t(sig[k - 1])
            h0 = lam(alpha_s, sigma_s) - lam(alpha_p, sigma_p)
            step["inv_r"] = float(1.0 / (h0 / h))
            step["half_b"] = float(0.5 * (alpha_t * (torch.exp(-h) - 1.0)))
        plan.append(step)
    return plan


def sample_latent(
    head: "DiffusionHead",
    plan: List[Dict[str, float]],
    t_emb: torch.Tensor,
    cond: torch.Tensor,
    neg: torch.Tensor,
    x: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """``sample_speech_tokens``: DPM-Solver++ (``plan``) from noise ``x`` [n, latent] with classifier-free
    guidance between the conditions ``cond`` and ``neg`` [n, H]; the dtype steps follow the reference (bf16
    model outputs and x0, fp32 update)."""
    n = x.shape[0]
    cp = head.cond_proj(torch.cat([cond, neg]))
    x0_prev = None
    for k, st in enumerate(plan):
        v = head(torch.cat([x, x]), cp + t_emb[k])
        vc, vu = v[:n], v[n:]
        v = vu + cfg_scale * (vc - vu)
        x0 = st["alpha"] * x - st["sigma"] * v
        xn = st["a"] * x.float() - st["b"] * x0
        if st["order"] == 2:
            xn = xn - st["half_b"] * (st["inv_r"] * (x0 - x0_prev))
        x, x0_prev = xn.to(x.dtype), x0
    return x


def _cache_layers(cache: Any) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Keys and values of a pickled ``DynamicCache`` ([1, n_kv, L, hd] each), old or new transformers layout."""
    d = getattr(cache, "__dict__", {})
    if "key_cache" in d:
        return list(d["key_cache"]), list(d["value_cache"])
    return [layer.keys for layer in cache.layers], [layer.values for layer in cache.layers]


def load_voice(path: str) -> Dict[str, torch.Tensor]:
    """A voice prompt ``.pt`` -> per context ``k`` / ``v`` [layers, L, n_kv, hd] and the last hidden state."""
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        d = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for ctx in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        keys, values = _cache_layers(d[ctx]["past_key_values"])
        out[f"{ctx}_k"] = torch.stack([k[0].transpose(0, 1) for k in keys])
        out[f"{ctx}_v"] = torch.stack([v[0].transpose(0, 1) for v in values])
        out[f"{ctx}_h"] = d[ctx]["last_hidden_state"][0, -1]
    return out


def voice_path(name: str, voices_dir: Optional[str]) -> str:
    if voices_dir:
        return os.path.join(voices_dir, f"{name}.pt")
    owner, repo, folder = VOICE_REPO
    return str(download_github_file(owner, repo, f"{folder}/{name}.pt"))


def resolve_voices(spec: Optional[str]) -> List[str]:
    """``None`` / "all" -> every voice, "english" -> the English ones, else a comma-separated list."""
    if spec == "english":
        return list(ENGLISH_VOICES)
    if spec is None or spec == "all":
        return ENGLISH_VOICES + OTHER_VOICES
    return [v.strip() for v in spec.split(",") if v.strip()]


def match_voice(requested: str, names: List[str]) -> Optional[str]:
    """Exact (case-insensitive) name, else the speaker part (``carter`` -> ``en-Carter_man``)."""
    r = requested.strip().lower()
    for n in names:
        if n.lower() == r:
            return n
    for n in names:
        speaker = re.split(r"[-_]", n.lower())
        if r in speaker[1:2] or r == n.lower().split("-", 1)[-1]:
            return n
    return None


class VibeVoiceRealtime(nn.Module):
    """Every weight of the checkpoint except the unused TTS-LM embedding table."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        dc = cfg["decoder_config"]
        H, nq, nkv = int(dc["hidden_size"]), int(dc["num_attention_heads"]), int(dc["num_key_value_heads"])
        hd, inter, eps = H // nq, int(dc["intermediate_size"]), float(dc["rms_norm_eps"])
        n_tts = int(cfg["tts_backbone_num_hidden_layers"])
        n_lm = int(dc["num_hidden_layers"]) - n_tts
        self.hidden_size, self.n_q, self.n_kv, self.head_dim = H, nq, nkv, hd
        self.rope_theta = float(dc["rope_theta"])
        self.vocab_size = int(dc["vocab_size"])
        self.latent_dim = int(cfg["acoustic_vae_dim"])
        self.embed_tokens = nn.Embedding(self.vocab_size, H)
        self.lm_layers = nn.ModuleList([Qwen2Layer(H, nq, nkv, hd, inter, eps) for _ in range(n_lm)])
        self.tts_layers = nn.ModuleList([Qwen2Layer(H, nq, nkv, hd, inter, eps) for _ in range(n_tts)])
        self.tts_norm = RMSNorm(H, eps)
        self.tts_input_types = nn.Embedding(2, H)
        self.acoustic_connector = SpeechConnector(self.latent_dim, H)
        self.prediction_head = DiffusionHead(cfg["diffusion_head_config"])
        self.eos = BinaryClassifier(H)
        self.register_buffer("speech_scaling_factor", torch.tensor(1.0))
        self.register_buffer("speech_bias_factor", torch.tensor(0.0))
        self.decoder = VibeVoiceAcousticDecoder(cfg["acoustic_tokenizer_config"])

    _RENAMES = [
        ("model.language_model.embed_tokens.", "embed_tokens."),
        ("model.language_model.layers.", "lm_layers."),
        ("model.tts_language_model.layers.", "tts_layers."),
        ("model.tts_language_model.norm.", "tts_norm."),
        ("model.tts_input_types.", "tts_input_types."),
        ("model.acoustic_connector.", "acoustic_connector."),
        ("model.prediction_head.", "prediction_head."),
        ("tts_eos_classifier.", "eos."),
        ("model.speech_scaling_factor", "speech_scaling_factor"),
        ("model.speech_bias_factor", "speech_bias_factor"),
        ("model.acoustic_tokenizer.decoder.", "decoder."),
    ]

    @classmethod
    def from_pretrained(cls, model_dir: str) -> Tuple["VibeVoiceRealtime", Dict[str, Any]]:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        if cfg.get("model_type") != "vibevoice_streaming":
            raise ValueError(f"{model_dir} is not a VibeVoice-Realtime checkpoint (model_type {cfg.get('model_type')})")
        with torch.device("meta"):
            model = cls(cfg)
        sd = {}
        for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items():
            if k == "model.tts_language_model.embed_tokens.weight":
                continue
            for old, new in cls._RENAMES:
                if k.startswith(old):
                    sd[new + k[len(old) :]] = v
                    break
            else:
                raise RuntimeError(f"unexpected VibeVoice weight {k}")
        model.load_state_dict(sd, strict=True, assign=True)
        return model, cfg
