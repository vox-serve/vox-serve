from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers.configuration_utils
from torch import nn
from transformers import AutoModel

from .base import DecoderCache

# The MOSS-Audio-Tokenizer remote code uses the old spelling "PreTrainedConfig"
# which was renamed to "PretrainedConfig" in newer transformers versions.
if not hasattr(transformers.configuration_utils, "PreTrainedConfig"):
    transformers.configuration_utils.PreTrainedConfig = transformers.configuration_utils.PretrainedConfig


# ---------------------------------------------------------------------------
# Stage configurations for the MOSS Audio Decoder
# ---------------------------------------------------------------------------

STAGE_CONFIGS = [
    {
        "hf_key": 0,
        "input_dim": 768,
        "d_model": 1280,
        "output_dim": None,  # Identity (no output_proj)
        "num_layers": 32,
        "num_heads": 20,
        "head_dim": 64,
        "ffn_dim": 5120,
        "context": 125,
    },
    {
        "hf_key": 2,
        "input_dim": 640,
        "d_model": 768,
        "output_dim": None,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "context": 250,
    },
    {
        "hf_key": 4,
        "input_dim": 384,
        "d_model": 768,
        "output_dim": None,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "context": 500,
    },
    {
        "hf_key": 6,
        "input_dim": 384,
        "d_model": 768,
        "output_dim": 240,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "context": 1000,
    },
]

PATCH_SIZES = [2, 2, 2, 240]


# ---------------------------------------------------------------------------
# Cache dataclass
# ---------------------------------------------------------------------------


@dataclass
class MossAudioDecoderCache(DecoderCache):
    """Streaming decoder cache for MOSS Audio Tokenizer.

    All tensors are stored batch-first so that the inherited
    ``__getitem__`` / ``cat`` / ``copy_from`` helpers work correctly.
    """

    # Per-stage KV ring buffers: (B, num_layers, num_heads, capacity, 2*head_dim)
    kv_caches: List[torch.Tensor]
    # Per-stage position offsets: (B,) — tracks RoPE offset + ring buffer write pos
    position_offsets: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Custom NN Modules for the MOSS Audio Decoder
# ---------------------------------------------------------------------------


def _apply_rope_paired(x: torch.Tensor, positions: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """Apply paired-dim RoPE rotation.

    Args:
        x: (..., head_dim) tensor
        positions: (...,) integer positions broadcastable to x shape minus last dim

    Returns:
        Rotated tensor of same shape as x.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    freqs = torch.arange(half, device=x.device, dtype=torch.float32)
    freqs = 1.0 / (max_period ** (freqs / half))
    # positions: expand to match x dims, (..., 1)
    angles = positions.unsqueeze(-1).float() * freqs  # (..., half)
    cos_vals = torch.cos(angles).to(x.dtype)
    sin_vals = torch.sin(angles).to(x.dtype)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos_vals - x1 * sin_vals
    out[..., 1::2] = x0 * sin_vals + x1 * cos_vals
    return out


class MossDecoderLayerScale(nn.Module):
    def __init__(self, d_model: int, init_value: float = 0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d_model,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class MossDecoderAttention(nn.Module):
    """Multi-head attention with combined QKV, paired-dim RoPE, and ring buffer KV cache."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int, context: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context = context
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch forward with causal mask. x: (B, T, d_model)."""
        B, T, _ = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(T, device=x.device)
        q = _apply_rope_paired(q, positions)
        k = _apply_rope_paired(k, positions)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)

    def forward_chunk_single(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        abs_position: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming forward for a single timestep with ring buffer KV cache.

        Args:
            x: (B, 1, d_model) single-step input
            kv_cache: (B, num_heads, capacity, 2*head_dim) ring buffer
            abs_position: (B,) absolute position of this timestep

        Returns:
            output: (B, 1, d_model)
        """
        B = x.shape[0]
        qkv = self.in_proj(x)  # (B, 1, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)

        # Apply RoPE — positions must be (B, 1, 1) for correct broadcast with (B, H, 1, D)
        rope_pos = abs_position.view(B, 1, 1)
        q = _apply_rope_paired(q, rope_pos)  # (B, H, 1, D)
        k = _apply_rope_paired(k, rope_pos)  # (B, H, 1, D)

        # Write K, V to ring buffer at abs_position % capacity
        capacity = kv_cache.shape[2]
        write_idx = (abs_position % capacity).view(B, 1, 1, 1)  # (B, 1, 1, 1)
        kv_new = torch.cat([k, v], dim=-1)  # (B, H, 1, 2*D)
        scatter_idx = write_idx.expand(B, self.num_heads, 1, 2 * self.head_dim)
        kv_cache.scatter_(2, scatter_idx, kv_new)

        # Read full cache
        full_k = kv_cache[..., : self.head_dim]  # (B, H, capacity, D)
        full_v = kv_cache[..., self.head_dim :]  # (B, H, capacity, D)

        # Build validity mask: a slot is valid if it has been written to
        pos = abs_position.view(B, 1, 1)  # (B, 1, 1)
        slot_indices = torch.arange(capacity, device=x.device).view(1, 1, capacity)
        total_written = pos + 1
        cur_write = pos % capacity
        buffer_full = total_written >= capacity
        valid = torch.where(buffer_full, torch.ones_like(slot_indices, dtype=torch.bool), slot_indices <= cur_write)
        attn_mask = torch.zeros(B, 1, 1, capacity, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(~valid.unsqueeze(1), float("-inf"))

        out = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        return self.out_proj(out)


class MossDecoderTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, ffn_dim: int, context: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MossDecoderAttention(d_model, num_heads, head_dim, context)
        self.layer_scale_1 = MossDecoderLayerScale(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.layer_scale_2 = MossDecoderLayerScale(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x)))
        x = x + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(x)))))
        return x

    def forward_chunk_single(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        abs_position: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step streaming forward. x: (B, 1, d_model)."""
        x = x + self.layer_scale_1(self.self_attn.forward_chunk_single(self.norm1(x), kv_cache, abs_position))
        x = x + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(x)))))
        return x


class MossDecoderTransformerStage(nn.Module):
    """Full transformer stage: input_proj + N layers + optional output_proj.

    Data flows in (B, C, T) layout (channel-first). Internally transposes
    to (B, T, d_model) for transformer processing.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: Optional[int],
        num_layers: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        context: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.layers = nn.ModuleList(
            [MossDecoderTransformerLayer(d_model, num_heads, head_dim, ffn_dim, context) for _ in range(num_layers)]
        )
        if output_dim is not None:
            self.output_proj = nn.Linear(d_model, output_dim, bias=False)
        else:
            self.output_proj = None

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context = context
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C_out, T)."""
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.input_proj(x)  # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x.transpose(1, 2)  # (B, C_out, T)

    def forward_chunk(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        position_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming forward for T timesteps (T may be >1 after upsampling).

        Each timestep is processed sequentially through all layers, updating
        the ring buffer KV cache.

        Args:
            x: (B, C, T) input
            kv_cache: (B, num_layers, num_heads, capacity, 2*head_dim)
            position_offset: (B,) current position offset for this stage

        Returns:
            output: (B, C_out, T)
        """
        B, C, T = x.shape
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.input_proj(x)  # (B, T, d_model)

        # Process each timestep sequentially through all layers
        outputs = []
        for t in range(T):
            x_t = x[:, t : t + 1, :]  # (B, 1, d_model)
            abs_pos = position_offset + t
            for i, layer in enumerate(self.layers):
                x_t = layer.forward_chunk_single(x_t, kv_cache[:, i], abs_pos)
            outputs.append(x_t)
        x = torch.cat(outputs, dim=1)  # (B, T, d_model)

        if self.output_proj is not None:
            x = self.output_proj(x)
        return x.transpose(1, 2)  # (B, C_out, T)


class MossDecoderPatchUpsample(nn.Module):
    """Stateless patch-based upsampling: (B, D*patch_size, T) -> (B, D, T*patch_size)."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        D = C // self.patch_size
        return x.view(B, D, self.patch_size, T).permute(0, 1, 3, 2).reshape(B, D, T * self.patch_size)


class MossDecoderLFQ(nn.Module):
    """Single LFQ quantizer decode path."""

    def __init__(self):
        super().__init__()
        self.codebook = nn.Embedding(1024, 8)
        self.out_proj = nn.Conv1d(8, 512, 1)
        # Will be wrapped with weight_norm after creation

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T) -> (B, 512, T)."""
        emb = self.codebook(codes)  # (B, T, 8)
        emb = emb.transpose(1, 2)  # (B, 8, T)
        return self.out_proj(emb)  # (B, 512, T)


class MossDecoderResidualLFQ(nn.Module):
    """Full quantizer with 32 LFQ quantizers."""

    def __init__(self, num_quantizers: int = 32):
        super().__init__()
        self.quantizers = nn.ModuleList([MossDecoderLFQ() for _ in range(num_quantizers)])
        self.output_proj = nn.Conv1d(512, 768, 1)
        # Will be wrapped with weight_norm after creation

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (32, B, T) -> (B, 768, T)."""
        result = None
        for i, quantizer in enumerate(self.quantizers):
            out = quantizer.decode(codes[i])  # (B, 512, T)
            if result is None:
                result = out
            else:
                result = result + out
        return self.output_proj(result)  # (B, 768, T)


class MossAudioDecoder(nn.Module):
    """Top-level MOSS audio decoder: codes -> waveform."""

    def __init__(self):
        super().__init__()
        self.quantizer = MossDecoderResidualLFQ(num_quantizers=32)
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, cfg in enumerate(STAGE_CONFIGS):
            self.stages.append(
                MossDecoderTransformerStage(
                    input_dim=cfg["input_dim"],
                    d_model=cfg["d_model"],
                    output_dim=cfg["output_dim"],
                    num_layers=cfg["num_layers"],
                    num_heads=cfg["num_heads"],
                    head_dim=cfg["head_dim"],
                    ffn_dim=cfg["ffn_dim"],
                    context=cfg["context"],
                )
            )
            self.upsamples.append(MossDecoderPatchUpsample(PATCH_SIZES[i]))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Batch decode: codes (32, B, T) -> audio (B, 1, T*1920)."""
        x = self.quantizer.decode(codes)  # (B, 768, T)
        for stage, upsample in zip(self.stages, self.upsamples, strict=False):
            x = stage(x)
            x = upsample(x)
        return x  # (B, 1, T*1920)

    def forward_chunk(self, codes: torch.Tensor, cache: MossAudioDecoderCache) -> torch.Tensor:
        """Streaming decode for a single timestep.

        Args:
            codes: (32, B, 1) single timestep of codes
            cache: MossAudioDecoderCache (updated in-place)

        Returns:
            audio: (B, 1, 1920) single frame of audio
        """
        x = self.quantizer.decode(codes)  # (B, 768, 1)
        for i, (stage, upsample) in enumerate(zip(self.stages, self.upsamples, strict=False)):
            T = x.shape[2]
            x = stage.forward_chunk(x, cache.kv_caches[i], cache.position_offsets[i])
            # Update position offset by number of timesteps processed
            cache.position_offsets[i].add_(T)
            x = upsample(x)
        return x  # (B, 1, 1920)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MossAudioDecoderCache:
        """Create empty cache for streaming decode."""
        kv_caches = []
        position_offsets = []
        for cfg in STAGE_CONFIGS:
            kv_caches.append(
                torch.zeros(
                    batch_size,
                    cfg["num_layers"],
                    cfg["num_heads"],
                    cfg["context"],
                    2 * cfg["head_dim"],
                    device=device,
                    dtype=dtype,
                )
            )
            position_offsets.append(torch.zeros(batch_size, device=device, dtype=torch.long))
        return MossAudioDecoderCache(kv_caches=kv_caches, position_offsets=position_offsets)


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------


def _apply_weight_norm_to_conv1d(conv: nn.Conv1d) -> nn.Conv1d:
    """Apply weight_norm parametrization to a Conv1d module."""
    return nn.utils.parametrizations.weight_norm(conv)


def _load_decoder_weights(decoder: MossAudioDecoder, state_dict: dict) -> Tuple[List[str], List[str]]:
    """Load HuggingFace weights into custom decoder architecture.

    Returns:
        (missing_keys, unexpected_keys)
    """
    # Apply weight_norm to quantizer conv layers before loading
    _apply_weight_norm_to_conv1d(decoder.quantizer.output_proj)
    for q in decoder.quantizer.quantizers:
        _apply_weight_norm_to_conv1d(q.out_proj)

    # Build mapped state dict
    mapped = {}

    for key, value in state_dict.items():
        new_key = _map_decoder_key(key)
        if new_key is not None:
            mapped[new_key] = value

    return decoder.load_state_dict(mapped, strict=False)


# Map HF decoder index to stage index
_HF_TO_STAGE = {0: 0, 2: 1, 4: 2, 6: 3}


def _map_decoder_key(key: str) -> Optional[str]:
    """Map a single HF weight key to our custom decoder architecture key."""
    # Quantizer keys pass through directly
    if key.startswith("quantizer."):
        # quantizer.input_proj.* -> skip (only needed for encode, not decode)
        if key.startswith("quantizer.input_proj."):
            return None
        # quantizer.quantizers.{i}.in_proj.* -> skip (encode-only)
        if ".in_proj." in key:
            return None
        # quantizer.output_proj.* -> quantizer.output_proj.*
        # quantizer.quantizers.{i}.codebook.* -> quantizer.quantizers.{i}.codebook.*
        # quantizer.quantizers.{i}.out_proj.* -> quantizer.quantizers.{i}.out_proj.*
        return key

    # Decoder stage keys
    if key.startswith("decoder."):
        parts = key.split(".", 2)  # ["decoder", "{hf_idx}", rest]
        if len(parts) < 3:
            return None
        try:
            hf_idx = int(parts[1])
        except ValueError:
            return None

        stage_idx = _HF_TO_STAGE.get(hf_idx)
        if stage_idx is None:
            return None

        rest = parts[2]

        # input_proj.weight -> stages.{s}.input_proj.weight
        if rest == "input_proj.weight":
            return f"stages.{stage_idx}.input_proj.weight"

        # output_proj.weight -> stages.{s}.output_proj.weight
        if rest == "output_proj.weight":
            return f"stages.{stage_idx}.output_proj.weight"

        # transformer.layers.{L}.self_attn.in_projs.0.weight -> stages.{s}.layers.{L}.self_attn.in_proj.weight
        if rest.startswith("transformer.layers."):
            # Remove "transformer." prefix
            layer_rest = rest[len("transformer.") :]
            # Replace "self_attn.in_projs.0." with "self_attn.in_proj."
            layer_rest = layer_rest.replace("self_attn.in_projs.0.", "self_attn.in_proj.")
            # Replace "self_attn.out_projs.0." with "self_attn.out_proj."
            layer_rest = layer_rest.replace("self_attn.out_projs.0.", "self_attn.out_proj.")
            return f"stages.{stage_idx}.{layer_rest}"

    return None


# ---------------------------------------------------------------------------
# Main tokenizer class
# ---------------------------------------------------------------------------


class MossAudioTokenizer(nn.Module):
    """Wrapper for OpenMOSS-Team/MOSS-Audio-Tokenizer.

    Encoding uses the HuggingFace model directly.
    Decoding uses a custom reimplemented decoder with streaming cache support.
    """

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()

        # Load HF model for encoding
        self.hf_model = AutoModel.from_pretrained(
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            trust_remote_code=True,
        )
        self.hf_model.to(device=device, dtype=dtype)
        self.hf_model.eval()

        # Build custom decoder
        self.decoder = MossAudioDecoder()

        # Load decoder weights from HF safetensors
        self._load_decoder_weights(device)

        self.decoder.to(device=device, dtype=torch.float32)
        self.decoder.eval()

        self.sample_rate = 24000
        self.num_codebooks = 32
        self.downsample_rate = 1920  # 24000 / 12.5 Hz

    def _load_decoder_weights(self, device):
        """Load weights from HF model files into custom decoder."""
        from transformers.utils import cached_file

        model_name = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

        try:
            index_file = cached_file(model_name, "model.safetensors.index.json")
            resolved_archive_file = None
        except Exception:
            try:
                resolved_archive_file = cached_file(model_name, "model.safetensors")
            except Exception:
                resolved_archive_file = cached_file(model_name, "pytorch_model.bin")
            index_file = None

        if resolved_archive_file:
            if resolved_archive_file.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(resolved_archive_file, device=str(device))
            else:
                state_dict = torch.load(resolved_archive_file, map_location=device)
        else:
            import json

            with open(index_file) as f:
                index_data = json.load(f)
            state_dict = {}
            for shard_file in set(index_data["weight_map"].values()):
                shard_path = cached_file(model_name, shard_file)
                if shard_file.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    shard_dict = load_file(shard_path, device=str(device))
                else:
                    shard_dict = torch.load(shard_path, map_location=device)
                state_dict.update(shard_dict)

        # Filter to only decoder/quantizer keys
        decoder_keys = {k: v for k, v in state_dict.items() if k.startswith(("quantizer.", "decoder."))}
        missing, unexpected = _load_decoder_weights(self.decoder, decoder_keys)
        if missing:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("MOSS decoder missing keys: %s", missing[:10])
        if unexpected:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("MOSS decoder unexpected keys: %s", unexpected[:10])

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode waveform to audio codes.

        Args:
            audio: (1, audio_length) mono waveform at 24kHz

        Returns:
            (T, 32) audio codes tensor on CPU
        """
        device = next(self.hf_model.parameters()).device
        audio = audio.to(device=device, dtype=torch.float32)

        if hasattr(self.hf_model, "batch_encode"):
            wav_1d = audio.squeeze(0)  # (audio_length,)
            enc = self.hf_model.batch_encode([wav_1d], num_quantizers=self.num_codebooks)
        else:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # (1, audio_length)
            input_values = audio.unsqueeze(0)  # (1, 1, audio_length)
            audio_len = audio.shape[-1]
            padding_mask = torch.ones(1, audio_len, device=device, dtype=torch.bool)
            enc = self.hf_model.encode(
                input_values,
                padding_mask=padding_mask,
                num_quantizers=self.num_codebooks,
                return_dict=True,
            )

        audio_codes = enc.audio_codes  # (NQ, 1, T)
        length = int(enc.audio_codes_lengths[0].item())
        codes = audio_codes[:, 0, :length].transpose(0, 1).contiguous().to(torch.long).cpu()
        return codes  # (T, 32)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform (batch mode).

        Args:
            codes: (batch, 32, time) audio codes

        Returns:
            (batch, 1, audio_length) waveform tensor
        """
        # Transpose to (32, batch, time)
        codes_nqbt = codes.transpose(0, 1)
        return self.decoder.forward(codes_nqbt)

    @torch.inference_mode()
    def decode_chunk(
        self,
        codes: torch.Tensor,
        cache: MossAudioDecoderCache,
    ) -> torch.Tensor:
        """Decode a chunk of tokens using streaming state.

        Args:
            codes: (batch, 32, time) audio codes
            cache: MossAudioDecoderCache (updated in-place)

        Returns:
            (batch, 1, audio_length) waveform tensor
        """
        batch_size, n_codebooks, time = codes.shape
        # Process each timestep sequentially through the decoder
        audio_frames = []
        for t in range(time):
            codes_t = codes[:, :, t : t + 1].transpose(0, 1)  # (32, B, 1)
            frame = self.decoder.forward_chunk(codes_t, cache)  # (B, 1, 1920)
            audio_frames.append(frame)
        return torch.cat(audio_frames, dim=2)  # (B, 1, time*1920)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MossAudioDecoderCache:
        return self.decoder.init_cache(batch_size, device, dtype)
