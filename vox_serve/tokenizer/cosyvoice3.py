"""Fun-CosyVoice3 speech decoder: the flow-matching DiT (speech tokens -> mel) and the causal HiFT
vocoder (mel -> 24 kHz audio), ported from FunAudioLLM/CosyVoice (Apache 2.0).

The modules mirror the checkpoint layout (``flow.pt`` / ``hift.pt``) and expose the computation as
building blocks, so a caller can run either whole sequences (``CosyVoice3Flow.solve``,
``CausalHiFT.forward``) or bounded windows with carried state (the Rust-runtime program).

Reference behaviour reproduced here:

* flow: token embedding -> ``PreLookaheadLayer`` (3-token look-ahead) -> x2 upsampling (mu), 10
  Euler steps on a cosine schedule with classifier-free guidance 0.7 (the unconditional branch zeroes
  mu / speaker / prompt mel but shares x), fixed initial noise indexed by absolute frame, and a
  block-causal attention mask with 50-frame blocks in streaming mode.
* DiT: AdaLayerNormZero blocks, x_transformers rotary embedding on the first 64 channels of q / k
  (interleaved pairs, absolute positions), causal conv position embedding (2 x kernel 31).
* HiFT: causal convolutions, a causal F0 predictor (3-frame look-ahead), and ``SineGen2`` in causal
  mode, whose source is constant within a mel frame (phase = 480 * cumsum of per-frame radians) plus
  a fixed uniform noise table indexed by absolute sample.
"""

import math
import os
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

MEL = 80
DIM = 1024
HEADS = 16
HEAD_DIM = 64
DEPTH = 22
N_STEPS = 10
CFG_RATE = 0.7
BLOCK_FRAMES = 50  # static_chunk_size: 25 tokens x token_mel_ratio 2
LOOKAHEAD = 3
SAMPLES_PER_FRAME = 480
HOP = 4  # iSTFT hop: 120 STFT frames per mel frame
N_FFT = 16
UPSAMPLE = (8, 5, 3)
UP_KERNELS = (16, 11, 7)
RES_KERNELS = (3, 7, 11)
SRC_RES_KERNELS = (7, 7, 11)
DILATIONS = (1, 3, 5)
N_HARMONICS = 9  # fundamental + 8 overtones
SAMPLE_RATE = 24000


def fold_weight_norm(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Replaces ``parametrizations.weight.original0/1`` (g, v) pairs by plain ``weight`` tensors."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".parametrizations.weight.original1"):
            base = k[: -len(".parametrizations.weight.original1")]
            g = sd[base + ".parametrizations.weight.original0"]
            out[base + ".weight"] = torch._weight_norm(v, g, 0)
        elif not k.endswith(".parametrizations.weight.original0"):
            out[k] = v
    return out


# ---- flow ---------------------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(DIM, DIM)
        self.to_k = nn.Linear(DIM, DIM)
        self.to_v = nn.Linear(DIM, DIM)
        self.to_out = nn.ModuleList([nn.Linear(DIM, DIM)])


class _AdaLN(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.linear = nn.Linear(DIM, n * DIM)


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(nn.Sequential(nn.Linear(DIM, 2 * DIM)), nn.Identity(), nn.Linear(2 * DIM, DIM))


class _DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = _AdaLN(6)
        self.attn = _Attn()
        self.ff = _FF()


class _ConvPos(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(DIM, DIM, 31, groups=16))
        self.conv2 = nn.Sequential(nn.Conv1d(DIM, DIM, 31, groups=16))


class _InputEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4 * MEL, DIM)
        self.conv_pos_embed = _ConvPos()


class _TimeEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(256, DIM), nn.SiLU(), nn.Linear(DIM, DIM))


class _Rotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)))


class _Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = _TimeEmbed()
        self.input_embed = _InputEmbed()
        self.rotary_embed = _Rotary()
        self.transformer_blocks = nn.ModuleList([_DiTBlock() for _ in range(DEPTH)])
        self.norm_out = _AdaLN(2)
        self.proj_out = nn.Linear(DIM, MEL)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _Estimator()


class _PreLookahead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(MEL, DIM, LOOKAHEAD + 1)
        self.conv2 = nn.Conv1d(DIM, MEL, 3)


def t_span() -> torch.Tensor:
    """Cosine time schedule of the 10-step Euler solver (as the reference computes it, fp32)."""
    t = torch.linspace(0, 1, N_STEPS + 1, dtype=torch.float32)
    return 1 - torch.cos(t * 0.5 * torch.pi)


def initial_noise(n_frames: int = 50 * 300) -> torch.Tensor:
    """``CausalConditionalCFM.rand_noise``: [80, n_frames] float32, indexed by absolute frame."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, MEL, n_frames, generator=g)[0]


class CosyVoice3Flow(nn.Module):
    """``CausalMaskedDiffWithDiT`` (inference). Weights in ``flow.pt`` naming."""

    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(6561, MEL)
        self.pre_lookahead_layer = _PreLookahead()
        self.spk_embed_affine_layer = nn.Linear(192, MEL)
        self.decoder = _Decoder()

    @classmethod
    def from_checkpoint(cls, path: str) -> "CosyVoice3Flow":
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return m.eval()

    @property
    def est(self) -> _Estimator:
        return self.decoder.estimator

    # -- conditioning --

    def speaker(self, xvec: torch.Tensor) -> torch.Tensor:
        """campplus x-vector [B, 192] -> projected speaker embedding [B, 80]."""
        return self.spk_embed_affine_layer(F.normalize(xvec.float(), dim=1).to(self.spk_embed_affine_layer.weight))

    def token_mu(self, emb: torch.Tensor, n_left: int) -> torch.Tensor:
        """``PreLookaheadLayer`` over token embeddings [B, L, 80] whose first ``n_left`` rows are left
        context only; rows past the sequence end must be zero (the reference pads with zeros).
        Returns mu for tokens [n_left, L - 3) upsampled to frames: [B, 2 * (L - 3 - n_left), 80]."""
        x = emb.transpose(1, 2)
        y = F.leaky_relu(self.pre_lookahead_layer.conv1(x))  # [B, 1024, L - 3]
        y = F.pad(y, (2, 0))[:, :, n_left:]  # conv2 is causal (zero left padding at the sequence start)
        y = self.pre_lookahead_layer.conv2(y)  # [B, 80, L - 3 - n_left]
        h = y.transpose(1, 2) + emb[:, n_left : emb.shape[1] - LOOKAHEAD]
        return h.repeat_interleave(2, dim=1)

    def sequence_mu(self, tokens: torch.Tensor) -> torch.Tensor:
        """mu of a whole token sequence [B, L] with zero right padding (``finalize=True``)."""
        emb = self.input_embedding(tokens.clamp(min=0))
        return self.token_mu(F.pad(emb, (0, 0, 0, LOOKAHEAD)), 0)

    # -- DiT pieces (``ada`` / ``final`` modulations depend only on the step, see ``modulations``) --

    @torch.no_grad()
    def modulations(self) -> Dict[str, torch.Tensor]:
        """Per-step AdaLayerNormZero outputs, shared by every row (t is the same for all rows):
        ``ada`` [steps, depth, 6, 1024] and ``final`` [steps, 2, 1024] (scale, shift), fp32."""
        e = self.est
        ts = t_span()[:N_STEPS].to(e.proj_out.weight.device)
        half = 128
        freq = torch.exp(torch.arange(half, device=ts.device).float() * -(math.log(10000) / (half - 1)))
        arg = 1000 * ts[:, None] * freq[None]
        temb = e.time_embed.time_mlp(torch.cat([arg.sin(), arg.cos()], dim=-1).to(e.proj_out.weight))
        act = F.silu(temb)
        ada = torch.stack([b.attn_norm.linear(act) for b in e.transformer_blocks], dim=1)
        final = e.norm_out.linear(act)
        return {"ada": ada.float().view(N_STEPS, DEPTH, 6, DIM), "final": final.float().view(N_STEPS, 2, DIM)}

    def rope_tables(self, pos: torch.Tensor):
        """cos / sin [.., 64] for absolute frame positions (interleaved pairs, fp32)."""
        f = pos.float()[..., None] * self.est.rotary_embed.inv_freq.float()
        f = torch.stack((f, f), dim=-1).flatten(-2)
        return f.cos(), f.sin()

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [.., T, 1024]: rotates channels [0, 64) (x_transformers partial rotary), fp32 math."""
        r = x[..., :HEAD_DIM].float()
        pair = r.unflatten(-1, (HEAD_DIM // 2, 2))
        rot = torch.stack((-pair[..., 1], pair[..., 0]), dim=-1).flatten(-2)
        r = r * cos + rot * sin
        return torch.cat([r.to(x.dtype), x[..., HEAD_DIM:]], dim=-1)

    # Mixed precision as under autocast: matmuls / convolutions in the weight dtype, layer norms, the
    # residual stream and the solver state in fp32.

    @property
    def wdtype(self) -> torch.dtype:
        return self.est.proj_out.weight.dtype

    def input_proj(self, x: torch.Tensor, cond: torch.Tensor, mu: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """``InputEmbedding.proj(cat[x, cond, mu, spk])`` before the conv position embedding."""
        spk = spk[:, None, :].expand(-1, x.shape[1], -1)
        return self.est.input_embed.proj(torch.cat([x, cond, mu, spk], dim=-1).to(self.wdtype))

    def conv_pos(self, h: torch.Tensor, tail1: Optional[torch.Tensor] = None, tail2: Optional[torch.Tensor] = None):
        """Causal conv position embedding of h [B, T, 1024] whose preceding 30 frames (inputs of conv1
        / conv2) are ``tail1`` / ``tail2`` (zeros = sequence start). Returns (h + pos in fp32, conv1
        output)."""
        cp = self.est.input_embed.conv_pos_embed
        B, T, _ = h.shape
        if tail1 is None:
            tail1 = h.new_zeros(B, 30, DIM)
        if tail2 is None:
            tail2 = h.new_zeros(B, 30, DIM)
        c1 = F.mish(cp.conv1[0](torch.cat([tail1, h], dim=1).transpose(1, 2))).transpose(1, 2)
        c2 = F.mish(cp.conv2[0](torch.cat([tail2, c1], dim=1).transpose(1, 2))).transpose(1, 2)
        return h.float() + c2.float(), c1

    def block(
        self,
        i: int,
        h: torch.Tensor,
        ada: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attend: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """DiT block ``i`` on h [B, T, 1024]; ``ada`` [6, 1024]; ``attend(i, q, k, v)`` -> [B, T, 1024]
        with q / k already rotated."""
        b = self.est.transformer_blocks[i]
        wd = self.wdtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada.float().unbind(0)
        n = (F.layer_norm(h.float(), (DIM,), eps=1e-6) * (1 + scale_msa) + shift_msa).to(wd)
        q = self.apply_rope(b.attn.to_q(n), cos, sin)
        k = self.apply_rope(b.attn.to_k(n), cos, sin)
        v = b.attn.to_v(n)
        h = h + gate_msa * b.attn.to_out[0](attend(i, q, k, v)).float()
        n = (F.layer_norm(h, (DIM,), eps=1e-6) * (1 + scale_mlp) + shift_mlp).to(wd)
        ff = b.ff.ff
        h = h + gate_mlp * ff[2](F.gelu(ff[0][0](n), approximate="tanh")).float()
        return h

    def head(self, h: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
        """Final modulation + projection: h [B, T, 1024] -> velocity [B, T, 80] fp32."""
        scale, shift = final.float().unbind(0)
        n = F.layer_norm(h.float(), (DIM,), eps=1e-6) * (1 + scale) + shift
        return self.est.proj_out(n.to(self.wdtype)).float()

    # -- whole-sequence solve (prompt encoding, validation) --

    @torch.no_grad()
    def solve(
        self,
        mu: torch.Tensor,
        spk: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
        mods: Optional[Dict[str, torch.Tensor]] = None,
        capture: Optional[Callable[[int, int, torch.Tensor, torch.Tensor], None]] = None,
        capture_tails: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Batch-1 CFM solve over a whole sequence: mu / cond [1, T, 80], spk [1, 80], noise
        [1, T, 80] -> mel [1, T, 80] (fp32). ``block_mask`` [T, T] bool (True = attend; None = full).
        ``capture(step, layer, k, v)`` sees the rotated keys / values [2, T, 1024] of every layer
        (row 0 conditional, row 1 unconditional); ``capture_tails(step, h, c1)`` the conv position
        embedding inputs of conv1 / conv2 [2, T, 1024]."""
        mods = mods or self.modulations()
        T = mu.shape[1]
        cos, sin = self.rope_tables(torch.arange(T, device=mu.device))
        mask = None if block_mask is None else block_mask[None, None]
        mu2 = torch.cat([mu, torch.zeros_like(mu)]).float()
        spk2 = torch.cat([spk, torch.zeros_like(spk)]).float()
        cond2 = torch.cat([cond, torch.zeros_like(cond)]).float()
        ts = t_span()
        x = noise.float()
        for s in range(N_STEPS):
            h0 = self.input_proj(x.expand(2, -1, -1), cond2, mu2, spk2)
            h, c1 = self.conv_pos(h0)
            if capture_tails is not None:
                capture_tails(s, h0, c1)

            def attend(i, q, k, v, s=s):
                if capture is not None:
                    capture(s, i, k, v)
                qh, kh, vh = (z.view(2, T, HEADS, HEAD_DIM).transpose(1, 2) for z in (q, k, v))
                o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
                return o.transpose(1, 2).reshape(2, T, DIM)

            for i in range(DEPTH):
                h = self.block(i, h, mods["ada"][s, i], cos, sin, attend)
            v = self.head(h, mods["final"][s])
            x = x + (ts[s + 1] - ts[s]) * ((1 + CFG_RATE) * v[:1] - CFG_RATE * v[1:])
        return x


def block_causal_mask(n: int, block: int = BLOCK_FRAMES, device=None) -> torch.Tensor:
    """[n, n] bool: frame i attends to frames of blocks <= its own (``subsequent_chunk_mask``)."""
    idx = torch.arange(n, device=device)
    return (idx[None, :] // block) <= (idx[:, None] // block)


# ---- HiFT ----------------------------------------------------------------------------------------


class _Snake(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(ch))
        # Optional fused implementation (x, alpha, pad) -> same result, e.g. ``vox_serve.fusion.dit.snake_pad``.
        self.fused: Optional[Callable] = None

    def forward(self, x, pad: int = 0):
        """Snake activation, then ``pad`` zeros on the left."""
        if self.fused is not None:
            return self.fused(x, self.alpha, pad)
        a = self.alpha[None, :, None]
        return F.pad(x + (1.0 / (a + 1e-9)) * torch.sin(x * a).pow(2), (pad, 0))


class _ResBlock(nn.Module):
    """Causal HiFi-GAN residual block (left-padded dilated convolutions, Snake activations)."""

    def __init__(self, ch: int, k: int):
        super().__init__()
        self.k = k
        self.convs1 = nn.ModuleList([nn.Conv1d(ch, ch, k, dilation=d) for d in DILATIONS])
        self.convs2 = nn.ModuleList([nn.Conv1d(ch, ch, k) for _ in DILATIONS])
        self.activations1 = nn.ModuleList([_Snake(ch) for _ in DILATIONS])
        self.activations2 = nn.ModuleList([_Snake(ch) for _ in DILATIONS])
        # Set by ``CausalHiFT.set_snake(fn, fuse_bias=True)``: convolutions run without their bias and
        # the fused Snake adds it together with the residual (one pass instead of three).
        self.fuse_bias = False

    def forward(self, x):
        if self.fuse_bias:
            return self._forward_fused(x)
        for c1, c2, a1, a2, d in zip(
            self.convs1, self.convs2, self.activations1, self.activations2, DILATIONS, strict=True
        ):
            xt = c1(a1(x, (self.k - 1) * d))
            xt = c2(a2(xt, self.k - 1))
            x = x + xt
        return x

    def _forward_fused(self, x):
        xt = b2 = None
        for c1, c2, a1, a2, d in zip(
            self.convs1, self.convs2, self.activations1, self.activations2, DILATIONS, strict=True
        ):
            if xt is None:
                a = a1(x, (self.k - 1) * d)
            else:  # x += conv2 output + bias, then Snake
                x, a = a1.fused(xt, a1.alpha, (self.k - 1) * d, res=x, bias=b2, write_sum=True)
            a = a2.fused(F.conv1d(a, c1.weight, None, dilation=d), a2.alpha, self.k - 1, bias=c1.bias)
            xt, b2 = F.conv1d(a, c2.weight), c2.bias
        return a2.fused(xt, a2.alpha, 0, res=x, bias=b2, snake=False)


class _F0Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.condnet = nn.Sequential(
            nn.Conv1d(MEL, 512, 4), nn.ELU(), *[m for _ in range(4) for m in (nn.Conv1d(512, 512, 3), nn.ELU())]
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, 80, T] -> f0 [B, T]; the sequence ends at T (zero look-ahead, ``finalize``)."""
        x = F.elu(self.condnet[0](F.pad(mel, (0, LOOKAHEAD))))
        for i in range(2, len(self.condnet), 2):
            x = F.elu(self.condnet[i](F.pad(x, (2, 0))))
        return self.classifier(x.transpose(1, 2)).squeeze(-1).abs()


class _Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_linear = nn.Linear(N_HARMONICS, 1)


def noise_table(n_samples: int, seed: int = 1986) -> torch.Tensor:
    """Uniform [0, 1) table standing in for ``SineGen2.sine_waves`` [n, 9] (the reference draws it
    from the global RNG at construction, so its exact values are not reproducible)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_samples, N_HARMONICS, generator=g)


class CausalHiFT(nn.Module):
    """``CausalHiFTGenerator`` (inference). Weights in ``hift.pt`` naming, weight norm folded."""

    def __init__(self):
        super().__init__()
        self.m_source = _Source()
        self.conv_pre = nn.Conv1d(MEL, 512, 5)
        self.ups = nn.ModuleList([nn.Conv1d(512 // 2**i, 512 // 2 ** (i + 1), k) for i, k in enumerate(UP_KERNELS)])
        self.source_downs = nn.ModuleList(
            [nn.Conv1d(N_FFT + 2, 256, 30, 15), nn.Conv1d(N_FFT + 2, 128, 6, 3), nn.Conv1d(N_FFT + 2, 64, 1)]
        )
        self.source_resblocks = nn.ModuleList(
            [_ResBlock(512 // 2 ** (i + 1), k) for i, k in enumerate(SRC_RES_KERNELS)]
        )
        self.resblocks = nn.ModuleList([_ResBlock(512 // 2 ** (i + 1), k) for i in range(3) for k in RES_KERNELS])
        self.conv_post = nn.Conv1d(64, N_FFT + 2, 7)
        self.f0_predictor = _F0Predictor()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)

    @classmethod
    def from_checkpoint(cls, path: str) -> "CausalHiFT":
        m = cls()
        m.load_state_dict(fold_weight_norm(torch.load(path, map_location="cpu", weights_only=True)), strict=True)
        m.f0_predictor.double()  # causal F0 is precision sensitive (the reference runs it in fp64)
        return m.eval()

    def set_snake(self, fn: Callable, fuse_bias: bool = False) -> None:
        """Runs every Snake activation with ``fn`` (x, alpha, pad), e.g. a fused kernel. ``fuse_bias``:
        the residual blocks also fold their convolution biases and residual adds into ``fn``, which
        then must take ``res``, ``bias``, ``write_sum`` and ``snake`` (``vox_serve.fusion.dit.snake_pad``)."""
        for m in self.modules():
            if isinstance(m, _Snake):
                m.fused = fn
            elif isinstance(m, _ResBlock):
                m.fuse_bias = fuse_bias

    # -- source --

    def frame_rad(self, f0: torch.Tensor) -> torch.Tensor:
        """f0 [B, T] -> per-frame phase increments of the 9 harmonics [B, T, 9] (fp64, cycles)."""
        h = torch.arange(1, N_HARMONICS + 1, device=f0.device, dtype=torch.float64)
        return (f0.double()[..., None] * h / SAMPLE_RATE) % 1

    def source(self, f0: torch.Tensor, phase: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Merged excitation [B, T * 480] from f0 [B, T], cumulative phase [B, T, 9] (cycles, incl.
        the frame's own increment) and ``noise`` [B, T * 480] = sum_h w_h * table[sample, h]."""
        w = self.m_source.l_linear.weight[0].double()
        sines = 0.1 * torch.sin(2 * math.pi * ((SAMPLES_PER_FRAME * phase) % 1))  # [B, T, 9]
        harm = (sines * w).sum(-1).float()  # [B, T]
        uv = (f0 > 10).float()
        amp = uv * 0.003 + (1 - uv) * (0.1 / 3)
        rep = lambda z: z.repeat_interleave(SAMPLES_PER_FRAME, dim=1)  # noqa: E731
        return torch.tanh(rep(uv * harm) + rep(amp) * noise + self.m_source.l_linear.bias.float())

    # -- decoder --

    def stft(self, s: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(s, N_FFT, HOP, N_FFT, window=self.window, return_complex=True)
        spec = torch.view_as_real(spec)
        return torch.cat([spec[..., 0], spec[..., 1]], dim=1)

    def istft(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """``torch.istft`` (center=True) as irfft + overlap-add (graph safe)."""
        mag = mag.clamp(max=1e2)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        frames = torch.fft.irfft(spec.transpose(1, 2), n=N_FFT) * self.window  # [B, n, 16]
        n = frames.shape[1]
        length = N_FFT + HOP * (n - 1)
        y = F.fold(frames.transpose(1, 2), (1, length), (1, N_FFT), stride=(1, HOP)).flatten(1)
        env = F.fold(
            self.window.pow(2)[None, :, None].expand(1, N_FFT, n), (1, length), (1, N_FFT), stride=(1, HOP)
        ).flatten(1)
        y = y / torch.where(env > 1e-11, env, torch.ones_like(env))
        return y[:, N_FFT // 2 : length - N_FFT // 2]

    def decode(self, mel: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """mel [B, 80, T] (sequence ends at T: zero look-ahead) and source [B, T * 480] -> audio
        [B, T * 480] in [-0.99, 0.99]."""
        s_stft = self.stft(s)
        x = self.conv_pre(F.pad(mel, (0, 4)))
        for i, (u, k) in enumerate(zip(UPSAMPLE, UP_KERNELS, strict=True)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](F.pad(F.interpolate(x, scale_factor=u, mode="nearest"), (k - 1, 0)))
            if i == len(UPSAMPLE) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            down = self.source_downs[i]
            si = down(F.pad(s_stft, (down.stride[0] - 1, 0)) if down.stride[0] > 1 else s_stft)
            x = x + self.source_resblocks[i](si)
            x = sum(self.resblocks[i * 3 + j](x) for j in range(3)) / 3
        x = self.conv_post(F.pad(F.leaky_relu(x), (6, 0)))
        audio = self.istft(torch.exp(x[:, : N_FFT // 2 + 1]), torch.sin(x[:, N_FFT // 2 + 1 :]))
        return audio.clamp(-0.99, 0.99)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Whole sequence (``inference(finalize=True)``): mel [B, 80, T] fp32, noise [B, T * 480]."""
        f0 = self.f0_predictor(mel.double()).float()
        phase = torch.cumsum(self.frame_rad(f0), dim=1)
        return self.decode(mel, self.source(f0, phase, noise))


# ---- frontend -----------------------------------------------------------------------------------


def load_wav(path: str, sr: int) -> torch.Tensor:
    """Mono [1, n] float32 at ``sr`` (``cosyvoice.utils.file_utils.load_wav``)."""
    import torchaudio

    speech, rate = torchaudio.load(path, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if rate != sr:
        speech = torchaudio.transforms.Resample(orig_freq=rate, new_freq=sr)(speech)
    return speech


class CosyVoice3Frontend:
    """Reference-audio features: S3 v3 speech tokens (25 Hz), the 24 kHz prompt mel (50 Hz) used as
    the flow condition, and the campplus x-vector. Token / mel lengths are aligned to mel = 2 x tokens."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        import librosa
        import onnxruntime

        from .s3 import S3TokenizerV3

        self.device = device
        self.s3 = S3TokenizerV3(os.path.join(model_dir, "speech_tokenizer_v3.onnx")).to(device)
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1
        self.campplus = onnxruntime.InferenceSession(
            os.path.join(model_dir, "campplus.onnx"), sess_options=opt, providers=["CPUExecutionProvider"]
        )
        self.whisper_filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)).to(device)
        self.mel_basis = torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=MEL, fmin=0, fmax=None))
        self.mel_window = torch.hann_window(1920)

    @torch.inference_mode()
    def speech_tokens(self, wav16: torch.Tensor) -> torch.Tensor:
        """[1, n] 16 kHz -> int64 tokens [L] (``whisper.log_mel_spectrogram`` + S3 v3)."""
        audio = wav16.to(self.device)
        stft = torch.stft(audio, 400, 160, window=torch.hann_window(400, device=self.device), return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self.whisper_filters @ mag, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        feat = (log_spec + 4.0) / 4.0
        tok, tok_len = self.s3.quantize(feat, torch.tensor([feat.shape[2]], dtype=torch.int32, device=self.device))
        return tok[0, : int(tok_len[0])].long().cpu()

    def xvector(self, wav16: torch.Tensor) -> torch.Tensor:
        """[1, n] 16 kHz -> campplus embedding [192]."""
        from torchaudio.compliance import kaldi

        feat = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        name = self.campplus.get_inputs()[0].name
        return torch.from_numpy(self.campplus.run(None, {name: feat.unsqueeze(0).numpy()})[0]).flatten().float()

    def prompt_mel(self, wav24: torch.Tensor) -> torch.Tensor:
        """[1, n] 24 kHz -> log mel [T, 80] (``matcha.utils.audio.mel_spectrogram``, fmax = Nyquist)."""
        y = F.pad(wav24[:, None], (720, 720), mode="reflect")[:, 0]
        spec = torch.stft(y, 1920, 480, 1920, window=self.mel_window, center=False, return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))[0].T

    def encode(self, path: str):
        """Reference audio -> (speech tokens [L], prompt mel [2L, 80], x-vector [192])."""
        wav16 = load_wav(path, 16000)
        assert wav16.shape[1] / 16000 <= 30, "reference audio longer than 30 s is not supported"
        tok = self.speech_tokens(wav16)
        mel = self.prompt_mel(load_wav(path, 24000))
        n = min(mel.shape[0] // 2, tok.shape[0])
        return tok[:n], mel[: 2 * n], self.xvector(wav16)
