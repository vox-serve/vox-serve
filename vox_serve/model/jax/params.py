"""Load Qwen3-TTS safetensors checkpoint into a JAX param pytree."""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Config dataclass (frozen dict would work but a simple namespace is clearer)
# ---------------------------------------------------------------------------
class Qwen3TTSConfig:
    """Minimal config for the JAX forward pass."""

    # Backbone
    backbone_hidden: int = 2048
    backbone_layers: int = 28
    backbone_heads: int = 16
    backbone_kv_heads: int = 8
    head_dim: int = 128
    backbone_intermediate: int = 6144
    codec_vocab: int = 3072
    text_vocab: int = 151_936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    # Depth transformer
    depth_hidden: int = 1024
    depth_layers: int = 5
    depth_heads: int = 16
    depth_kv_heads: int = 8
    depth_intermediate: int = 3072
    depth_vocab: int = 2048
    num_codebooks: int = 15  # depth codebooks (codec 1-15)


# ---------------------------------------------------------------------------
# Key remapping helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(
    r"^talker\.(?P<block>model|code_predictor\.model)\.layers\.(?P<idx>\d+)\.(?P<rest>.+)$"
)


def _remap_layer_key(m: re.Match) -> tuple[str, int, str]:
    block = "backbone" if m.group("block") == "model" else "depth"
    idx = int(m.group("idx"))
    rest = m.group("rest")
    return block, idx, rest


def _attn_subkey(rest: str) -> tuple[str, str]:
    """Split 'self_attn.q_proj.weight' → ('self_attn', 'q_proj.weight')."""
    parts = rest.split(".", 1)
    if parts[0] == "self_attn":
        return "attn", parts[1]
    if parts[0] == "mlp":
        return "mlp", parts[1]
    # layernorm keys
    return parts[0], parts[1] if len(parts) > 1 else "weight"


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _resolve_checkpoint(model_name: str) -> Path:
    """Find the safetensors checkpoint for a HuggingFace model."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache / f"models--{model_name.replace('/', '--')}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found in HF cache: {model_dir}")
    # Pick the latest snapshot
    snapshots = sorted(model_dir.glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {model_dir}")
    ckpt = snapshots[-1] / "model.safetensors"
    if not ckpt.exists():
        # Try sharded
        shards = sorted(snapshots[-1].glob("model-*.safetensors"))
        if shards:
            return shards  # type: ignore[return-value]
        raise FileNotFoundError(f"No safetensors checkpoint in {snapshots[-1]}")
    return ckpt


def _set_nested(d: dict, keys: list[str], value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _transpose_linear(w: np.ndarray) -> np.ndarray:
    """Transpose PyTorch linear weight (out, in) → JAX convention (in, out)."""
    return w.T


def load_params(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[dict, Qwen3TTSConfig]:
    """Load checkpoint into a nested JAX param dict.

    Returns (params, config) where params has structure::

        params["backbone"]["embed_codec"]          # (3072, 2048)
        params["backbone"]["embed_text"]            # (151936, 2048)
        params["backbone"]["text_projection"]["fc1"]["weight"]  # (2048, 2048)
        params["backbone"]["text_projection"]["fc1"]["bias"]    # (2048,)
        params["backbone"]["text_projection"]["fc2"]["weight"]  # (2048, 2048)
        params["backbone"]["text_projection"]["fc2"]["bias"]    # (2048,)
        params["backbone"]["codec_head"]            # (2048, 3072) — transposed
        params["backbone"]["norm"]                  # (2048,)
        params["backbone"]["layers"][i]["attn"]["q_proj"]  # (2048, 2048)
        params["backbone"]["layers"][i]["attn"]["k_proj"]  # (2048, 1024)
        params["backbone"]["layers"][i]["attn"]["v_proj"]  # (2048, 1024)
        params["backbone"]["layers"][i]["attn"]["o_proj"]  # (2048, 2048)
        params["backbone"]["layers"][i]["attn"]["q_norm"]  # (128,)
        params["backbone"]["layers"][i]["attn"]["k_norm"]  # (128,)
        params["backbone"]["layers"][i]["input_layernorm"]   # (2048,)
        params["backbone"]["layers"][i]["post_attn_layernorm"]  # (2048,)
        params["backbone"]["layers"][i]["mlp"]["gate_proj"]  # (2048, 6144)
        params["backbone"]["layers"][i]["mlp"]["up_proj"]    # (2048, 6144)
        params["backbone"]["layers"][i]["mlp"]["down_proj"]  # (6144, 2048)

        params["depth"]["projection"]["weight"]    # (2048, 1024) — transposed
        params["depth"]["projection"]["bias"]      # (1024,)
        params["depth"]["codec_embeddings"]        # (15, 2048, 2048) — stacked
        params["depth"]["lm_heads"]                # (15, 1024, 2048) — stacked+transposed
        params["depth"]["norm"]                    # (1024,)
        params["depth"]["layers"][i]["attn"][...]   # same structure, depth dims
        params["depth"]["layers"][i]["mlp"][...]
        params["depth"]["layers"][i]["input_layernorm"]
        params["depth"]["layers"][i]["post_attn_layernorm"]

    All linear weights are transposed to (in_features, out_features) for JAX.
    """
    config = Qwen3TTSConfig()
    ckpt_path = _resolve_checkpoint(model_name)

    # Handle single file or sharded
    paths = [ckpt_path] if isinstance(ckpt_path, Path) else ckpt_path

    # Load all tensors as numpy bf16
    raw: dict[str, np.ndarray] = {}
    for p in paths:
        with safe_open(str(p), framework="numpy") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    params: dict = {"backbone": {"layers": {}}, "depth": {"layers": {}}}

    # --- Backbone non-layer params ---
    params["backbone"]["embed_codec"] = raw["talker.model.codec_embedding.weight"]
    params["backbone"]["embed_text"] = raw["talker.model.text_embedding.weight"]
    params["backbone"]["norm"] = raw["talker.model.norm.weight"]
    params["backbone"]["codec_head"] = _transpose_linear(raw["talker.codec_head.weight"])

    # Text projection MLP
    params["backbone"]["text_projection"] = {
        "fc1": {
            "weight": _transpose_linear(raw["talker.text_projection.linear_fc1.weight"]),
            "bias": raw["talker.text_projection.linear_fc1.bias"],
        },
        "fc2": {
            "weight": _transpose_linear(raw["talker.text_projection.linear_fc2.weight"]),
            "bias": raw["talker.text_projection.linear_fc2.bias"],
        },
    }

    # --- Depth non-layer params ---
    params["depth"]["projection"] = {
        "weight": _transpose_linear(raw["talker.code_predictor.small_to_mtp_projection.weight"]),
        "bias": raw["talker.code_predictor.small_to_mtp_projection.bias"],
    }
    params["depth"]["norm"] = raw["talker.code_predictor.model.norm.weight"]

    # Stack 15 codec_embeddings → (15, 2048, 2048)
    codec_embs = [raw[f"talker.code_predictor.model.codec_embedding.{i}.weight"] for i in range(15)]
    params["depth"]["codec_embeddings"] = np.stack(codec_embs, axis=0)

    # Stack 15 lm_heads → (15, 1024, 2048) (transposed from [2048, 1024])
    lm_heads = [_transpose_linear(raw[f"talker.code_predictor.lm_head.{i}.weight"]) for i in range(15)]
    params["depth"]["lm_heads"] = np.stack(lm_heads, axis=0)

    # --- Layer params (backbone + depth) ---
    for key, tensor in raw.items():
        m = _LAYER_RE.match(key)
        if not m:
            continue

        block, idx, rest = _remap_layer_key(m)
        layer_dict = params[block]["layers"].setdefault(idx, {})

        if rest.startswith("self_attn."):
            attn_dict = layer_dict.setdefault("attn", {})
            attn_key = rest[len("self_attn."):]  # e.g. "q_proj.weight" or "q_norm.weight"

            if attn_key.endswith(".weight"):
                name = attn_key[:-len(".weight")]  # e.g. "q_proj" or "q_norm"
                if name.endswith("_proj"):
                    attn_dict[name] = _transpose_linear(tensor)
                else:
                    # norm weights — no transpose
                    attn_dict[name] = tensor
            elif attn_key.endswith(".bias"):
                name = attn_key[:-len(".bias")]
                attn_dict[name + "_bias"] = tensor

        elif rest.startswith("mlp."):
            mlp_dict = layer_dict.setdefault("mlp", {})
            # e.g. "mlp.gate_proj.weight"
            mlp_key = rest[len("mlp."):]
            name = mlp_key.replace(".weight", "")
            mlp_dict[name] = _transpose_linear(tensor)

        elif rest == "input_layernorm.weight":
            layer_dict["input_layernorm"] = tensor
        elif rest == "post_attention_layernorm.weight":
            layer_dict["post_attn_layernorm"] = tensor

    # Convert layer dicts from {int: dict} to sorted lists
    for block in ("backbone", "depth"):
        layer_map = params[block]["layers"]
        n_layers = config.backbone_layers if block == "backbone" else config.depth_layers
        params[block]["layers"] = [layer_map[i] for i in range(n_layers)]

    # Convert entire tree to jax.Arrays
    params = jax.tree.map(lambda x: jnp.array(x, dtype=dtype), params)

    return params, config
