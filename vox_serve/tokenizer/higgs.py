"""Higgs Audio v2 tokenizer, the codec of Higgs Audio v3 TTS: 8 residual codebooks of 1024 codes at
25 Hz, 24 kHz mono audio. Port of transformers' ``HiggsAudioV2TokenizerModel`` (Apache 2.0, Boson AI
and the HuggingFace team), assembled from the transformers 4.x DAC and HuBERT modules. The weights are
bundled in the TTS checkpoint under ``tied.embedding.modality_embeddings.0.model.``.

* decode: per codebook ``project_out(embed[code])``, summed -> ``fc2`` -> DAC decoder (x960
  upsampling; every transposed conv uses ``output_padding = stride % 2`` and the final tanh is
  dropped). The embedding / projection / ``fc2`` chain is linear, so it folds into one table of
  256-d vectors per (codebook, code).
* encode: HuBERT on the 16 kHz resampled clip (mean of all hidden states, every second frame) ->
  semantic encoder, concatenated with the DAC encoder output -> ``fc`` -> residual VQ (nearest code).
"""

import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 24000
SEMANTIC_RATE = 16000
HOP = 960
N_CODEBOOKS = 8
CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 64
HIDDEN = 1024  # acoustic 256 + semantic 768
ACOUSTIC_HIDDEN = 256
SEMANTIC_HIDDEN = 768
PREFIX = "tied.embedding.modality_embeddings.0.model."

DAC_CONFIG = dict(
    encoder_hidden_size=64,
    downsampling_ratios=[8, 5, 4, 2, 3],
    decoder_hidden_size=1024,
    upsampling_ratios=[8, 5, 4, 2, 3],
    hidden_size=ACOUSTIC_HIDDEN,
    n_codebooks=9,
    codebook_size=1024,
    codebook_dim=8,
    sampling_rate=16000,
    hop_length=960,
)
# hubert-base (``semantic_model_config`` of bosonai/higgs-audio-v2-tokenizer), inference only.
HUBERT_CONFIG = dict(
    hidden_size=SEMANTIC_HIDDEN,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    feat_extract_norm="group",
    feat_extract_activation="gelu",
    feat_proj_layer_norm=True,
    conv_dim=[512] * 7,
    conv_stride=[5, 2, 2, 2, 2, 2, 2],
    conv_kernel=[10, 3, 3, 3, 3, 2, 2],
    conv_bias=False,
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    do_stable_layer_norm=False,
    layer_norm_eps=1e-5,
    mask_time_prob=0.0,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
)
SEMANTIC_DOWNSAMPLE = 2  # 960 / (24000 / 16000) / 320


class SemanticResidualUnit(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN, 3, padding=dilation, dilation=dilation, bias=False)
        self.conv2 = nn.Conv1d(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.elu(self.conv1(F.elu(x))))


class SemanticEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_units = nn.ModuleList([SemanticResidualUnit(1), SemanticResidualUnit(1)])
        self.conv = nn.Conv1d(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.res_units:
            x = unit(x)
        return self.conv(x)


class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(SEMANTIC_HIDDEN, SEMANTIC_HIDDEN, 3, padding=1, bias=False)
        self.conv_blocks = nn.ModuleList([SemanticEncoderBlock(), SemanticEncoderBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class Codebook(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("embed", torch.zeros(CODEBOOK_SIZE, CODEBOOK_DIM))


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook = Codebook()
        self.project_in = nn.Linear(HIDDEN, CODEBOOK_DIM)
        self.project_out = nn.Linear(CODEBOOK_DIM, HIDDEN)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, HIDDEN, T] -> nearest codes [B, T]."""
        h = self.project_in(x.transpose(1, 2))
        e = self.codebook.embed.t()
        dist = -(h.pow(2).sum(-1, keepdim=True) - 2 * h @ e + e.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T] -> [B, HIDDEN, T]."""
        return self.project_out(F.embedding(codes, self.codebook.embed)).transpose(1, 2)


class ResidualVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer() for _ in range(N_CODEBOOKS)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, HIDDEN, T] -> codes [B, N_CODEBOOKS, T]."""
        out = []
        for q in self.quantizers:
            idx = q.encode(x)
            x = x - q.decode(idx)
            out.append(idx)
        return torch.stack(out, dim=1)


def _adjust_dac_decoder(decoder: nn.Module) -> None:
    for m in decoder.modules():
        if isinstance(m, nn.ConvTranspose1d):
            m.output_padding = (m.stride[0] % 2,)
    decoder.tanh = nn.Identity()


def _conv_out_len(module: nn.Module, n: int) -> int:
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            k, s, p, d = m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0]
            n = (n + 2 * p - d * (k - 1) - 1) // s + 1
    return n


class HiggsAudioTokenizer(nn.Module):
    """The codec. Call ``from_checkpoint`` with the Higgs TTS checkpoint directory."""

    def __init__(self, with_encoder: bool = True):
        super().__init__()
        from transformers import DacConfig, DacModel

        dac = DacModel(DacConfig(**DAC_CONFIG))
        self.acoustic_decoder = dac.decoder
        _adjust_dac_decoder(self.acoustic_decoder)
        self.quantizer = ResidualVQ()
        self.fc2 = nn.Linear(HIDDEN, ACOUSTIC_HIDDEN)
        self.with_encoder = with_encoder
        if with_encoder:
            from transformers import HubertConfig, HubertModel

            self.acoustic_encoder = dac.encoder
            self.semantic_model = HubertModel(HubertConfig(**HUBERT_CONFIG))
            self.encoder_semantic = SemanticEncoder()
            self.fc = nn.Linear(HIDDEN, HIDDEN)
        # Folded decode input table: [N_CODEBOOKS * CODEBOOK_SIZE, ACOUSTIC_HIDDEN] (fc2 bias excluded).
        self.register_buffer("code_table", torch.zeros(N_CODEBOOKS * CODEBOOK_SIZE, ACOUSTIC_HIDDEN), persistent=False)
        # Optional fused decoder (decoder, z) -> same result, e.g. ``vox_serve.fusion.dac.fused_dac_decode``.
        self.fused_decode: Optional[Callable] = None

    @classmethod
    def from_checkpoint(cls, model_dir: str, with_encoder: bool = True) -> "HiggsAudioTokenizer":
        from safetensors import safe_open

        codec = cls(with_encoder=with_encoder)
        own = codec.state_dict()
        state = {}
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            for name in f.keys():
                if name.startswith(PREFIX):
                    key = name[len(PREFIX) :]
                    if key in own:
                        state[key] = f.get_tensor(name).float()
        missing = set(own) - set(state) - {"code_table"}
        if missing:
            raise RuntimeError(f"Higgs codec weights missing from checkpoint: {sorted(missing)[:8]}")
        codec.load_state_dict(state, strict=False)
        codec.eval().requires_grad_(False)
        codec.fold_code_table()
        return codec

    @torch.no_grad()
    def fold_code_table(self) -> None:
        rows = []
        for q in self.quantizer.quantizers:
            h = q.project_out(q.codebook.embed.float())  # [1024, HIDDEN]
            rows.append(F.linear(h, self.fc2.weight.float()))
        self.code_table = torch.cat(rows).to(self.code_table.device)

    # ---- decode ----------------------------------------------------------------------------------

    def latent(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, T, N_CODEBOOKS] (0..1023) -> DAC decoder input [B, ACOUSTIC_HIDDEN, T]."""
        offs = torch.arange(N_CODEBOOKS, device=codes.device) * CODEBOOK_SIZE
        z = F.embedding(codes + offs, self.code_table).sum(dim=2) + self.fc2.bias.float()
        return z.transpose(1, 2)

    def decode(self, codes: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """codes [B, T, N_CODEBOOKS] -> audio [B, T * HOP] (float32). Frames where ``valid`` [B, T] is
        False decode from a zero latent (closest to the zero padding at the ends of a whole sequence)."""
        z = self.latent(codes)
        if valid is not None:
            z = z * valid[:, None, :]
        dtype = next(self.acoustic_decoder.parameters()).dtype
        if self.fused_decode is not None:
            return self.fused_decode(self.acoustic_decoder, z.to(dtype))[:, 0].float()
        return self.acoustic_decoder(z.to(dtype))[:, 0].float()

    # ---- encode ----------------------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [L] float32 at 24 kHz -> codes [T, N_CODEBOOKS] (int64). Clips under 1 s are zero-padded
        (as SGLang-Omni's ``encode_reference`` does)."""
        import torchaudio

        if not self.with_encoder:
            raise RuntimeError("codec loaded without the encoder")
        dev = self.fc.weight.device
        x = wav.reshape(1, 1, -1).float().to(dev)
        if x.shape[-1] < SAMPLE_RATE:
            x = F.pad(x, (0, SAMPLE_RATE - x.shape[-1]))
        x16 = torchaudio.functional.resample(x, SAMPLE_RATE, SEMANTIC_RATE)[:, 0]
        hs = self.semantic_model(F.pad(x16, (160, 160)), output_hidden_states=True).hidden_states
        sem = torch.stack(list(hs), dim=1).mean(dim=1)[:, ::SEMANTIC_DOWNSAMPLE]
        e_sem = self.encoder_semantic(sem.transpose(1, 2))
        if _conv_out_len(self.acoustic_encoder, x.shape[-1]) != e_sem.shape[2]:
            x = F.pad(x, (HOP // 2, HOP // 2))
        e_ac = self.acoustic_encoder(x)
        emb = self.fc(torch.cat([e_ac, e_sem], dim=1).transpose(1, 2)).transpose(1, 2)
        return self.quantizer.encode(emb)[0].transpose(0, 1)


def receptive_field_frames() -> float:
    """One-sided receptive field of the DAC decoder, in frames (for sizing streaming context)."""
    rf, rate = 3.0, 1  # conv1 (k7) at the frame rate
    for stride in DAC_CONFIG["upsampling_ratios"]:
        rate *= stride
        rf += math.ceil(stride / 2) / rate + sum(3 * d for d in (1, 3, 9)) / rate
    return rf + 3 / rate
