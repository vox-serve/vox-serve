"""Fun-CosyVoice3-0.5B graph program: Qwen2 LM over speech tokens, then a chunked flow-matching DiT
(speech tokens -> mel) and the causal HiFT vocoder, both with per-request state.

LM rows are ``tok = [id, n_text]`` and ``mask = [kind, voice]``: kind 1 embeds ``id`` with the text
table, kind 2 with ``speech_embedding`` (sos / task id, reference speech and generated tokens).
The last prompt row carries the number of synthesis-text tokens (min / max length 2x / 20x) and the
voice slot. Sampling is the reference's RAS (nucleus top-p 0.8 / top-k 25, resampled from the full
distribution when the pick repeats within the last 10 tokens); stop ids are 6561..6760, masked until
the minimum length. Output rows are ``[token, voice]``, or ``[token, -1]`` for a token the reference
drops (more than 5 consecutive silent tokens; it is still fed back to the LM).

Detokenization (stateful, one chunk per request per step): a chunk is a 30-token window, two tokens
of left context (the voice prompt's last two tokens on the first chunk), 25 core tokens and 3 of
look-ahead; the core becomes one 50-frame block of the reference's block-causal streaming flow.
Each flow step attends to three key/value segments (``seg_attention``): the voice prompt (computed
once per voice by ``hook_encode_voice``), a ring of the request's last ``W`` chunks, and the chunk
itself. The HiFT runs on a 90-frame window (40 frames of mel history + the new block) and carries
the source phase, the last 32 frames of excitation and the absolute sample index across chunks;
non-final chunks hold back the last 8 frames, which still lack look-ahead.

Deviations from the reference: the voice prompt is a standalone segment (its mel conditioning ends
at the prompt, and its last partial block does not see generated frames), generated frames attend
to at most ``W`` earlier chunks, and the DiT runs in fp16 with fp32 norms / residual / solver.
Audio rows hold up to 58 frames; the last int16 of a row is its valid length.
"""

import functools
import hashlib
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...fusion.dit import gconv_mish, gconv_weight, res_ln_mod, ring_write, rope_qk, snake_pad
from ...fusion.hift import hift_decode_cl
from ...fusion.seg_attention import seg_attention
from ...rt.core import Arena, EngineSpecBuilder, RtAttention
from ...rt.kernels import ops
from ...rt.program import TOKEN_AUDIO, TOKEN_OTHER, TOKEN_STOP, LMProgram, pcm16
from ...rt.prompt import Prompt, ResourceBusyError
from ...tokenizer.cosyvoice3 import (
    BLOCK_FRAMES,
    CFG_RATE,
    DEPTH,
    DIM,
    HEADS,
    MEL,
    N_STEPS,
    SAMPLES_PER_FRAME,
    CausalHiFT,
    CosyVoice3Flow,
    CosyVoice3Frontend,
    block_causal_mask,
    initial_noise,
    noise_table,
    t_span,
)
from ...utils import download_audio_from_url, get_logger
from ..cosyvoice2.model import CosyVoice2Config, CosyVoice2LLM
from ..cosyvoice2.program import text_normalize

logger = get_logger(__name__)

REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TEXT, SPEECH = 1, 2
SPEECH_VOCAB = 6561
SOS, EOS, TASK = SPEECH_VOCAB, SPEECH_VOCAB + 1, SPEECH_VOCAB + 2
VOCAB = SPEECH_VOCAB + 200
SILENT = (1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323)
MAX_SILENT = 5
RAS_TOP_P, RAS_TOP_K, RAS_WIN = 0.8, 25, 10
SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_REF_AUDIO = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
DEFAULT_REF_TEXT = (
    "The examination and testimony of the experts enabled the Commission to conclude that five shots may "
    "have been fired"
)

# chunk geometry (tokens / mel frames)
LEFT, CORE, AHEAD = 2, 25, 3
WINDOW = LEFT + CORE + AHEAD
CONV_TAIL = 30  # causal conv position embedding: 2 x kernel 31
NOISE_FRAMES = 50 * 300
HIFT_HIST, HIFT_SRC, HIFT_HOLD = 40, 32, 8  # mel history / carried excitation / held-back frames
HIFT_WIN = HIFT_HIST + BLOCK_FRAMES
MAX_EMIT = HIFT_WIN - HIFT_SRC  # frames
SOURCE_NOISE = 300 * 24000

_SPECIAL = (
    "<|im_start|> <|im_end|> <|endofprompt|> [breath] <strong> </strong> [noise] [laughter] [cough] [clucking] "
    "[accent] [quick_breath] <laughter> </laughter> [hissing] [sigh] [vocalized-noise] [lipsmack] [mn] "
    "<|endofsystem|>"
).split()
_PHONES = (
    "AA AA0 AA1 AA2 AE AE0 AE1 AE2 AH AH0 AH1 AH2 AO AO0 AO1 AO2 AW AW0 AW1 AW2 AY AY0 AY1 AY2 B CH D DH EH EH0 "
    "EH1 EH2 ER ER0 ER1 ER2 EY EY0 EY1 EY2 F G HH IH IH0 IH1 IH2 IY IY0 IY1 IY2 JH K L M N NG OW OW0 OW1 OW2 OY "
    "OY0 OY1 OY2 P R S SH T TH UH UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH a ai an ang ao b c ch d e ei en eng f g h "
    "i ian in ing iu ià iàn iàng iào iá ián iáng iáo iè ié iòng ióng iù iú iā iān iāng iāo iē iě iōng iū iǎ iǎn "
    "iǎng iǎo iǒng iǔ j k l m n o ong ou p q r s sh t u uang ue un uo uà uài uàn uàng uá uái uán uáng uè ué uì uí "
    "uò uó uā uāi uān uāng uē uě uī uō uǎ uǎi uǎn uǎng uǐ uǒ vè w x y z zh à ài àn àng ào á ái án áng áo è èi èn "
    "èng èr é éi én éng ér ì ìn ìng í ín íng ò òng òu ó óng óu ù ùn ú ún ā āi ān āng āo ē ēi ēn ēng ě ěi ěn ěng "
    "ěr ī īn īng ō ōng ōu ū ūn ǎ ǎi ǎn ǎng ǎo ǐ ǐn ǐng ǒ ǒng ǒu ǔ ǔn ǘ ǚ ǜ"
).split()


def special_tokens() -> List[str]:
    """``CosyVoice3Tokenizer`` additional special tokens."""
    return _SPECIAL + [f"[{p}]" for p in _PHONES]


class CosyVoice3LM(nn.Module):
    """``CosyVoice3LM`` (inference) in ``llm.pt`` naming."""

    def __init__(self):
        super().__init__()
        cfg = CosyVoice2Config()
        self.llm = CosyVoice2LLM(cfg)
        self.llm_decoder = nn.Linear(cfg.hidden_size, VOCAB, bias=False)
        self.speech_embedding = nn.Embedding(VOCAB, cfg.hidden_size)


class _Voice:
    """A reference voice encoded into a voice slot of the detokenizer; reusable once ``users`` is 0."""

    def __init__(self, slot: int, speech: List[int]):
        self.slot, self.speech, self.users = slot, speech, 0


class _VoiceUse:
    """A request's hold on its voice (``Prompt.hold``), released when the request lets go of it."""

    def __init__(self, voice: _Voice, lock: threading.Lock):
        self.voice, self._lock = voice, lock
        with lock:
            voice.users += 1

    def __del__(self):
        with self._lock:
            self.voice.users -= 1


class CosyVoice3Program(LMProgram):
    uses_input_mask = True
    uses_repetition_penalty = False
    detok_stateful = True
    # Detok state lives in per-request slots, so the two lanes never share any.
    supports_urgent_lane = True
    fb_columns = 1

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        name = self.config.get("model", REPO)
        if "/" not in name:
            name = REPO
        self.model_name = name
        self.model_dir = snapshot_download(name)
        cfg = CosyVoice2Config()
        self.n_codebooks = 2
        self.vocab_size = VOCAB
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.detok_interval = WINDOW
        self.detok_samples = MAX_EMIT * SAMPLES_PER_FRAME + 1
        # LMProgram.build overrides the token limit through ``model.default_sampling_config``.
        self.model = SimpleNamespace(default_sampling_config=SimpleNamespace(max_tokens=4096))

        self.lm = CosyVoice3LM()
        sd = torch.load(os.path.join(self.model_dir, "llm.pt"), map_location="cpu", weights_only=True)
        self.lm.load_state_dict(sd, strict=True)
        self.lm.to(self.dtype).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_dir, "CosyVoice-BlankEN"))
        self.tokenizer.add_special_tokens(
            {"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "additional_special_tokens": special_tokens()}
        )

        dd = self.detok_device
        flow = CosyVoice3Flow.from_checkpoint(os.path.join(self.model_dir, "flow.pt")).to(dd)
        mods = flow.modulations()
        self.ada, self.final = mods["ada"], mods["final"]
        self.mods = mods
        est = flow.est
        inv_freq = est.rotary_embed.inv_freq.float().clone()
        est.half()  # the DiT in fp16; token embedding, look-ahead convs and speaker projection stay fp32
        est.rotary_embed.inv_freq = inv_freq
        self.inv_freq = inv_freq
        self.qkv_w = [
            torch.cat([b.attn.to_q.weight, b.attn.to_k.weight, b.attn.to_v.weight]) for b in est.transformer_blocks
        ]
        self.qkv_b = [torch.cat([b.attn.to_q.bias, b.attn.to_k.bias, b.attn.to_v.bias]) for b in est.transformer_blocks]
        cp = est.input_embed.conv_pos_embed
        self.pos_conv = [(gconv_weight(c[0]), c[0].bias) for c in (cp.conv1, cp.conv2)]
        self.flow = flow
        self.hift = CausalHiFT.from_checkpoint(os.path.join(self.model_dir, "hift.pt")).to(dd)
        # --fuse-codec: one-launch history-ring writes, GELU in the FF GEMM's epilogue, and the vocoder
        # in channels-last layout with its conv biases / residual adds folded into the Snake kernels
        # (hift_layout "ncl": the same fusions in the default layout).
        self.fuse_codec = bool(self.config.get("fuse_codec"))
        self.hift.set_snake(snake_pad, fuse_bias=self.fuse_codec)
        self.hift_decode = self.hift.decode
        if self.fuse_codec and self.config.get("hift_layout", "cl") == "cl":
            self.hift_decode = functools.partial(hift_decode_cl, self.hift)
        self.frontend = CosyVoice3Frontend(self.model_dir, str(dd))
        ts = t_span()
        self.dts = [float(ts[s + 1] - ts[s]) for s in range(N_STEPS)]

        # Voice slots (prompt key/values per flow step) and per-request decoder state.
        self.n_voices = int(self.config.get("n_voices", 4))
        self.voice_frames = 2 * (int(self.config.get("voice_frames", 600)) // 2)
        self.detok_slots = int(self.config.get("detok_slots") or self.max_batch_size)
        per_chunk = N_STEPS * DEPTH * 2 * 2 * BLOCK_FRAMES * DIM * 2  # bytes per (slot, chunk)
        hist = self.config.get("history_chunks")
        if hist is None:
            budget = float(self.config.get("flow_history_gb", 24)) * (1 << 30)
            hist = int(budget // ((self.detok_slots + 1) * per_chunk))
        self.history = max(1, min(8, int(hist)))
        logger.info(
            "CosyVoice3: %d detok slots, %d-chunk flow history (%.1f GiB), %d voice slots of %d frames",
            self.detok_slots,
            self.history,
            (self.detok_slots + 1) * self.history * per_chunk / (1 << 30),
            self.n_voices,
            self.voice_frames,
        )

    # ---- LM ---------------------------------------------------------------------------------------

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        kind = arena["mask"][:rows, :1]
        tok = ids[:, 0]
        emb = self.lm.llm.model.model.embed_tokens
        text = emb(tok.clamp(0, emb.num_embeddings - 1))
        speech = self.lm.speech_embedding(tok.clamp(0, VOCAB - 1))
        return torch.where(kind == TEXT, text, speech)

    def fuse_lm_modules(self):
        return [self.lm.llm]

    def fused_stack_spec(self):
        from ...fusion.decoder_stack import neox_cos_sin

        bb = self.lm.llm.model.model
        theta = float(bb.layers[0].self_attn.rope_theta)
        return dict(
            layers=list(bb.layers),
            norm=bb.norm,
            n_q=self.num_qo_heads,
            n_kv=self.num_kv_heads,
            head_dim=self.head_dim,
            cos_sin=neox_cos_sin(self.max_context + 1, self.head_dim, theta, self.device),
        )

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        if self.fstack is not None:
            return self.fstack(hidden, pos, attn, self.kv_cache)
        return self.lm.llm(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.llm_decoder(hidden)[:, None, :]

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        n = n_req
        logits = self.lm.llm_decoder(hidden).float()  # [n, V]
        slot = arena["slot"][:n].long()
        first = arena["first"][:n] != 0
        step = arena["step"][:n]
        active = arena["active"][:n]
        temp = arena["temp"][:n]
        # Request constants arrive on the last prompt row.
        last_row = arena["last_row"][:n].long()
        n_text = torch.where(first, arena["tok"][:, 1][last_row], self.req_text[slot])
        voice = torch.where(first, arena["mask"][:, 1][last_row], self.req_voice[slot])
        self.req_text.index_copy_(0, slot, n_text)
        self.req_voice.index_copy_(0, slot, voice)

        vocab = torch.arange(VOCAB, device=logits.device)
        early = (step < 2 * n_text)[:, None] & (vocab >= SPEECH_VOCAB)[None]
        logits = logits.masked_fill(early, float("-inf"))
        # RAS nucleus: sorted probabilities kept while the mass before them is < top_p, at most top_k.
        t = torch.where(temp > 0, temp, torch.ones_like(temp))
        probs = torch.softmax(logits / t[:, None], dim=-1)
        top_p, top_i = probs.topk(RAS_TOP_K, dim=-1)
        keep = (top_p.cumsum(-1) - top_p) < RAS_TOP_P
        neg = torch.full_like(logits, float("-inf"))
        nucleus = neg.scatter(1, top_i, torch.where(keep, logits.gather(1, top_i), float("-inf")))
        k = ops()
        args = (active, temp, self.k0[:n], self.p1[:n], self.m0[:n], arena["seed"][:n], step)
        d1, d2 = self.draw[0, :n], self.draw[1, :n]
        k.sample(nucleus[:, None], 1, *args, self.LM_SALT, d1)
        # A pick seen in the last 10 tokens is redrawn from the full distribution without it.
        ras = torch.where(first[:, None], torch.full_like(self.ras[slot], -1), self.ras[slot])
        rep = (ras == d1).any(-1)
        k.sample(logits.scatter(1, d1.long(), float("-inf"))[:, None], 1, *args, self.LM_SALT + 1, d2)
        tok = torch.where(rep, d2[:, 0], d1[:, 0])
        tok = torch.where(step >= 20 * n_text, torch.full_like(tok, EOS), tok)
        ras.scatter_(1, (step % RAS_WIN).long()[:, None], tok[:, None])
        self.ras.index_copy_(0, slot, ras)
        silent = (tok[:, None] == self.silent_ids[None]).any(-1)
        run = torch.where(silent, torch.where(first, 0, self.silent_run[slot]) + 1, 0)
        self.silent_run.index_copy_(0, slot, run)
        out_tok[:, 0] = tok
        out_tok[:, 1] = torch.where(silent & (run > MAX_SILENT), torch.full_like(voice, -1), voice)

    # ---- state ------------------------------------------------------------------------------------

    def declare(self, builder: EngineSpecBuilder) -> None:
        super().declare(builder)
        lanes = [(builder.dt_inputs, self.max_detok_batch_size)]
        if builder.urgent_lane:
            lanes.append((builder.du_inputs, max(self._urgent_sizes())))
        for dinp, nb in lanes:
            dinp.add("dt_len", (nb,), torch.int32)  # valid tokens of the window
            dinp.add("dt_last", (nb,), torch.int32)  # no further chunk: emit everything

    def allocate_state(self) -> None:
        super().allocate_state()
        dev, dd = self.device, self.detok_device
        S = self.max_slots + 1
        i32 = dict(dtype=torch.int32, device=dev)
        self.req_text = torch.zeros(S, **i32)
        self.req_voice = torch.zeros(S, **i32)
        self.ras = torch.full((S, RAS_WIN), -1, **i32)
        self.silent_run = torch.zeros(S, **i32)
        self.silent_ids = torch.tensor(SILENT, **i32)
        n = self.max_batch_size
        self.draw = torch.zeros(2, n, 1, **i32)
        self.k0 = torch.zeros(n, **i32)
        self.p1 = torch.ones(n, dtype=torch.float32, device=dev)
        self.m0 = torch.zeros(n, dtype=torch.float32, device=dev)

        NV, P, D, W = self.n_voices, self.voice_frames, self.detok_slots + 1, self.history
        f16 = dict(dtype=torch.float16, device=dd)
        self.prompt_kv = torch.zeros(N_STEPS, DEPTH, 2, NV * 2, P, DIM, **f16)
        self.voice_tail = torch.zeros(N_STEPS, 2, NV * 2, CONV_TAIL, DIM, **f16)
        self.voice_tok = torch.zeros(NV, LEFT, dtype=torch.long, device=dd)
        self.voice_len = torch.zeros(NV, dtype=torch.int32, device=dd)
        self.voice_spk = torch.zeros(NV, MEL, device=dd)
        self.hist_kv = torch.zeros(N_STEPS, DEPTH, 2, D * 2 * W, BLOCK_FRAMES * DIM, **f16)
        self.slot_tail = torch.zeros(N_STEPS, 2, D * 2, CONV_TAIL, DIM, **f16)
        self.slot_chunk = torch.zeros(D, dtype=torch.long, device=dd)
        self.mel_hist = torch.zeros(D, MEL, HIFT_HIST, device=dd)
        self.src_hist = torch.zeros(D, HIFT_SRC * SAMPLES_PER_FRAME, device=dd)
        self.phase = torch.zeros(D, 9, dtype=torch.float64, device=dd)
        self.frame0 = torch.zeros(D, dtype=torch.long, device=dd)
        self.flow_noise = initial_noise(NOISE_FRAMES).T.contiguous().to(dd)  # [frames, 80]
        w = self.hift.m_source.l_linear.weight[0].float().cpu()
        self.src_noise = (noise_table(SOURCE_NOISE) * w).sum(-1).to(dd)
        self.default_voice = self.hook_encode_voice(str(download_audio_from_url(DEFAULT_REF_AUDIO)), 0)["speech_tokens"]

    # ---- voices -----------------------------------------------------------------------------------

    @torch.no_grad()
    def hook_encode_voice(self, path: str, slot: int) -> Dict[str, Any]:
        """Encodes reference audio into voice slot ``slot``: the flow prompt's key / values and conv
        tails for every step, its speaker embedding and last two tokens. Returns the speech tokens
        (the LM prompt). The caller guarantees no running request uses the slot."""
        if not 0 <= slot < self.n_voices:
            raise ValueError(f"voice slot {slot} out of range")
        dd = self.detok_device
        tok, mel, xvec = self.frontend.encode(path)
        n = tok.shape[0]
        if n < LEFT:
            raise ValueError("reference audio is too short")
        m = min(n, self.voice_frames // 2)
        if m < n:
            logger.info("voice prompt trimmed to its last %d of %d tokens for the flow", m, n)
        ftok = tok[n - m :].to(dd)
        P2 = 2 * m
        f = self.flow
        mu = f.sequence_mu(ftok[None])
        spk = f.speaker(xvec[None].to(dd))
        cond = mel[2 * (n - m) :][None].to(dd)
        noise = self.flow_noise[:P2][None]
        rows = slice(2 * slot, 2 * slot + 2)
        k = min(CONV_TAIL, P2)

        def capture(s, i, kk, vv):
            self.prompt_kv[s, i, 0, rows, :P2] = kk
            self.prompt_kv[s, i, 1, rows, :P2] = vv

        def capture_tails(s, h0, c1):
            for c, t in enumerate((h0, c1)):
                self.voice_tail[s, c, rows].zero_()
                self.voice_tail[s, c, rows, CONV_TAIL - k :] = t[:, P2 - k :]

        f.solve(mu, spk, cond, noise, block_mask=block_causal_mask(P2, device=dd), mods=self.mods,
                capture=capture, capture_tails=capture_tails)  # fmt: skip
        self.voice_tok[slot] = ftok[-LEFT:]
        self.voice_len[slot] = P2
        self.voice_spk[slot] = spk[0].float()
        # The engine streams read the slot without synchronizing with this (hook) stream.
        torch.cuda.synchronize(dd)
        return {"speech_tokens": tok.tolist()}

    # ---- detokenizer ------------------------------------------------------------------------------

    def compile_detok_targets(self):
        return [(self.hift, "decode")]

    def _flow_chunk(self, tok, valid, voice, slot, chunk, reset, c_len) -> torch.Tensor:
        """One 50-frame block per row -> mel [bd, 50, 80] (frames past ``c_len`` are meaningless)."""
        f, est = self.flow, self.flow.est
        bd = tok.shape[0]
        W = self.history
        emb = f.input_embedding(tok) * valid[..., None]
        mu = f.token_mu(emb, LEFT)  # [bd, 50, 80]
        zeros = torch.zeros_like(mu)
        spk = self.voice_spk[voice][:, None, :].expand(-1, BLOCK_FRAMES, -1)
        # rows are cfg-major: [conditional rows; unconditional rows (zero mu / speaker / prompt mel)]
        rest = torch.cat([torch.cat([zeros, mu, spk], dim=-1), torch.zeros_like(mu).repeat(1, 1, 3)])
        frames = torch.arange(BLOCK_FRAMES, device=mu.device)
        pos = self.voice_len[voice].long()[:, None] + BLOCK_FRAMES * chunk[:, None] + frames
        ang = pos.float()[..., None] * self.inv_freq
        ang = torch.stack((ang, ang), dim=-1).flatten(-2)
        cos, sin = ang.cos().repeat(2, 1, 1), ang.sin().repeat(2, 1, 1)
        x = self.flow_noise[pos % NOISE_FRAMES]  # [bd, 50, 80] fp32

        cfg = torch.arange(2, device=mu.device).repeat_interleave(bd)
        p_row = (voice.repeat(2) * 2 + cfg).int()
        p_len = self.voice_len[voice].repeat(2)
        h_row = (slot.repeat(2) * 2 + cfg).int()
        h_len = (chunk.clamp(max=W) * BLOCK_FRAMES).int().repeat(2)
        c_len2 = c_len.int().repeat(2)
        hr, pr = h_row.long(), p_row.long()
        ring = hr * W + (chunk % W).repeat(2)
        keep_tail = (~reset).repeat(2)[:, None, None]
        proj = est.input_embed.proj
        blocks = est.transformer_blocks
        for s in range(N_STEPS):
            h0 = proj(torch.cat([x.repeat(2, 1, 1), rest], dim=-1).half())
            t1, t2 = (torch.where(keep_tail, self.slot_tail[s, c][hr], self.voice_tail[s, c][pr]) for c in range(2))
            # conv position embedding (f.conv_pos), fused
            (w1, b1), (w2, b2) = self.pos_conv
            c1 = gconv_mish(torch.cat([t1, h0], 1), w1, b1)
            h = gconv_mish(torch.cat([t2, c1], 1), w2, b2, res=h0)  # fp32 residual stream
            self.slot_tail[s, 0].index_copy_(0, hr, h0[:, -CONV_TAIL:])
            self.slot_tail[s, 1].index_copy_(0, hr, c1[:, -CONV_TAIL:])
            # Each block's closing residual (gate_mlp * ff) is folded into the next normalization.
            o = gate = None
            for i in range(DEPTH):
                b = blocks[i]
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada[s, i].unbind(0)
                n = res_ln_mod(h, scale_msa, shift_msa, torch.half, o, gate)
                qkv = F.linear(n, self.qkv_w[i], self.qkv_b[i])
                rope_qk(qkv, cos, sin, DIM)
                q, k, v = qkv[..., :DIM], qkv[..., DIM : 2 * DIM], qkv[..., 2 * DIM :]
                hk = self.hist_kv[s, i, 0].view(-1, W * BLOCK_FRAMES, DIM)
                hv = self.hist_kv[s, i, 1].view(-1, W * BLOCK_FRAMES, DIM)
                o = seg_attention(
                    q, self.prompt_kv[s, i, 0], self.prompt_kv[s, i, 1], p_row, p_len,
                    hk, hv, h_row, h_len, k, v, c_len2, HEADS,
                )  # fmt: skip
                # The chunk joins the history after attending (the ring slot may hold chunk - W).
                if self.fuse_codec:
                    ring_write(self.hist_kv[s, i], qkv, ring, DIM)
                else:
                    self.hist_kv[s, i, 0].view(-1, BLOCK_FRAMES, DIM).index_copy_(0, ring, k)
                    self.hist_kv[s, i, 1].view(-1, BLOCK_FRAMES, DIM).index_copy_(0, ring, v)
                n = res_ln_mod(h, scale_mlp, shift_mlp, torch.half, b.attn.to_out[0](o), gate_msa)
                ff = b.ff.ff
                if self.fuse_codec:  # cuBLASLt GELU epilogue (tanh approximation)
                    l0 = ff[0][0]
                    g = torch._addmm_activation(l0.bias, n.view(-1, DIM), l0.weight.t(), use_gelu=True)
                    o = ff[2](g.view(*n.shape[:2], -1))
                else:
                    o = ff[2](F.gelu(ff[0][0](n), approximate="tanh"))
                gate = gate_mlp
            scale, shift = self.final[s].unbind(0)  # f.head
            vel = est.proj_out(res_ln_mod(h, scale, shift, torch.half, o, gate)).float()
            x = x + self.dts[s] * ((1 + CFG_RATE) * vel[:bd] - CFG_RATE * vel[bd:])
        return x

    def _hift_chunk(self, mel, slot, reset, c_len, last):
        """Vocodes the new block after ``HIFT_HIST`` frames of history -> (samples [bd, MAX_EMIT * 480],
        valid length [bd])."""
        hift = self.hift
        bd = mel.shape[0]
        dev = mel.device
        spf = SAMPLES_PER_FRAME
        new = mel.transpose(1, 2)
        pad = torch.zeros(bd, MEL, HIFT_HIST, device=dev)
        r3 = reset[:, None, None]
        win = torch.where(r3, torch.cat([new, pad], 2), torch.cat([self.mel_hist[slot], new], 2))
        off = torch.where(reset, 0, HIFT_HIST)
        vend = off + c_len
        fr = torch.arange(HIFT_WIN, device=dev)
        win = win * (fr[None, None, :] < vend[:, None, None])
        f0 = hift.f0_predictor(win.double()).float()
        cs = torch.cumsum(hift.frame_rad(f0), dim=1)  # [bd, 90, 9] cycles
        base = torch.where(reset[:, None], 0.0, self.phase[slot] - cs[:, HIFT_SRC - 1])
        frame0 = torch.where(reset, 0, self.frame0[slot])
        smp = torch.arange(HIFT_WIN * spf, device=dev)
        noise = self.src_noise[(frame0[:, None] * spf + smp) % SOURCE_NOISE]
        src = hift.source(f0, base[:, None] + cs, noise)
        carried = F.pad(self.src_hist[slot], (0, (HIFT_WIN - HIFT_SRC) * spf))
        src = torch.where((~reset)[:, None] & (smp < HIFT_SRC * spf), carried, src)
        audio = self.hift_decode(win, src)  # [bd, 90 * 480]

        e0 = torch.where(reset, 0, HIFT_SRC)
        e1 = torch.maximum(vend - torch.where(last, 0, HIFT_HOLD), e0)
        j = torch.arange(MAX_EMIT * spf, device=dev)
        pcm = audio.gather(1, (e0[:, None] * spf + j).clamp(max=HIFT_WIN * spf - 1))
        n = (e1 - e0) * spf
        pcm = pcm * (j < n[:, None])

        st = (vend - HIFT_HIST).clamp(min=0)
        idx = (st[:, None] + torch.arange(HIFT_HIST, device=dev))[:, None, :].expand(bd, MEL, HIFT_HIST)
        self.mel_hist.index_copy_(0, slot, win.gather(2, idx))
        self.src_hist.index_copy_(0, slot, src.gather(1, st[:, None] * spf + smp[: HIFT_SRC * spf]))
        at = cs.gather(1, (st + HIFT_SRC - 1)[:, None, None].expand(bd, 1, cs.shape[2]))[:, 0]
        self.phase.index_copy_(0, slot, (base + at) % 1)
        self.frame0.index_copy_(0, slot, frame0 + st)
        return pcm, n

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        dd = self.detok_device
        slot = arena["dt_slot"][:bd].long().clamp(0, self.detok_slots)
        reset = arena["dt_reset"][:bd] != 0
        dlen = arena["dt_len"][:bd].long()
        last = arena["dt_last"][:bd] != 0
        toks = arena["dt_tok"][:bd].long()
        voice = toks[:, :, 1].amax(1).clamp(0, self.n_voices - 1)
        tok = toks[:, :, 0].clamp(0, SPEECH_VOCAB - 1)
        tok = torch.cat([torch.where(reset[:, None], self.voice_tok[voice], tok[:, :LEFT]), tok[:, LEFT:]], 1)
        chunk = torch.where(reset, 0, self.slot_chunk[slot] + 1)
        valid = torch.arange(WINDOW, device=dd)[None] < dlen[:, None]
        c_len = 2 * (dlen.clamp(max=LEFT + CORE) - LEFT).clamp(min=0)
        mel = self._flow_chunk(tok, valid, voice, slot, chunk, reset, c_len)
        self.slot_chunk.index_copy_(0, slot, chunk)
        pcm, n = self._hift_chunk(mel, slot, reset, c_len, last)
        audio = out["audio"][:bd]
        audio[:, :-1].copy_(pcm16(pcm))
        audio[:, -1].copy_(n.to(torch.int16))

    # ---- meta -------------------------------------------------------------------------------------

    def token_kind(self, tok: torch.Tensor) -> torch.Tensor:
        # Column 1 is the voice slot, -1 for tokens that are fed back but not vocoded.
        kind = torch.where(tok[:, 1] < 0, TOKEN_OTHER, TOKEN_AUDIO)
        return torch.where(tok[:, 0] >= SPEECH_VOCAB, TOKEN_STOP, kind).to(torch.int32)

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """Rows ``[id, n_text]`` with mask ``[kind, voice slot]``: ``[sos] system_prompt ref_text text [task]
        ref_speech``, or ``[sos] system_prompt text [task]`` for a reference clip without transcript."""
        ref_text = model_kwargs.get("ref_text")
        ref_text = ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None
        if audio_path is not None:
            voice = self._voice(audio_path)
        else:
            if ref_text is not None:
                raise ValueError("ref_text needs reference audio")
            voice, ref_text = self._voices()[0][1], DEFAULT_REF_TEXT
        text = text_normalize(text)

        def with_system(t: str) -> str:
            return t if "<|endofprompt|>" in t else SYSTEM_PROMPT + t

        if ref_text is not None:
            prefix, text_ids = self._encode(with_system(ref_text)), self._encode(text)
        else:
            prefix, text_ids = [], self._encode(with_system(text))
        n_text = len(text_ids)
        tokens: List[int] = []
        mask: List[int] = []
        for tok, kind in [(SOS, SPEECH)] + [(t, TEXT) for t in prefix + text_ids] + [(TASK, SPEECH)]:
            tokens += [tok, n_text]
            mask += [kind, voice.slot]
        if ref_text is not None:
            for t in voice.speech:
                tokens += [t, n_text]
                mask += [SPEECH, voice.slot]
        return Prompt(tokens, mask, hold=_VoiceUse(voice, self._voice_lock))

    _voice_lock = threading.Lock()
    _encode_lock = threading.Lock()
    _voice_slots: Optional[List[Any]] = None

    def _voices(self) -> List[Any]:
        """Per slot: (content key, voice, last use) or None; slot 0 holds the default voice."""
        if self._voice_slots is None:
            self._voice_slots = [None] * max(1, self.n_voices)
            self._voice_slots[0] = ("", _Voice(0, list(self.default_voice)), 0)
            self._voice_clock = 0
        return self._voice_slots

    def _lookup(self, key: str) -> Optional[_Voice]:
        slots = self._voices()
        self._voice_clock += 1
        for i, e in enumerate(slots):
            if i > 0 and e is not None and e[0] == key:
                self._voice_slots[i] = (key, e[1], self._voice_clock)
                return e[1]
        return None

    def _reserve(self) -> Optional[int]:
        slots = self._voices()
        free = [i for i in range(1, len(slots)) if slots[i] is None]
        if free:
            return free[0]
        idle = [i for i in range(1, len(slots)) if slots[i][1].users == 0]
        if not idle:
            return None
        i = min(idle, key=lambda i: slots[i][2])
        slots[i] = None
        return i

    def _voice(self, path: str) -> _Voice:
        """The voice slot of reference clip ``path``, encoding it on a cache miss (an LRU of slots that no
        running request uses). Raises ``ResourceBusyError`` instead of waiting while running requests use
        every slot: the Python runtime releases slots on the thread that builds prompts (the Rust runtime's
        builder, ``rust/src/models/cosyvoice3.rs``, keeps its own slots and waits on a request thread)."""
        with open(path, "rb") as fh:
            key = hashlib.sha1(fh.read()).hexdigest()
        with self._voice_lock:
            v = self._lookup(key)
        if v is not None:
            return v
        if len(self._voices()) < 2:
            raise ValueError("reference audio needs a voice slot (n_voices >= 2)")
        with self._encode_lock:
            with self._voice_lock:
                v = self._lookup(key)
                if v is not None:
                    return v
                slot = self._reserve()
            if slot is None:
                raise ResourceBusyError(f"all {len(self._voices()) - 1} voice slots are in use by running requests")
            speech = [int(t) for t in self.hook_encode_voice(path, slot)["speech_tokens"]]
            if not speech:
                raise RuntimeError("encode_voice returned no speech tokens")
            v = _Voice(slot, speech)
            with self._voice_lock:
                self._voice_slots[slot] = (key, v, self._voice_clock)
            return v

    def meta(self) -> Dict[str, Any]:
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "cosyvoice3_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        self.tokenizer.backend_tokenizer.save(tok_path)
        return {
            "kind": "cosyvoice3",
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 24000,
            "detok_overlap": AHEAD + LEFT,
            "max_tokens": self.model.default_sampling_config.max_tokens,
            "sos": SOS,
            "task_id": TASK,
            "system_prompt": SYSTEM_PROMPT,
            "default_voice": {"ref_text": DEFAULT_REF_TEXT, "speech_tokens": self.default_voice},
            "n_voices": self.n_voices,
            "row_kinds": {"text": TEXT, "speech": SPEECH},
            "decode_mask": [SPEECH, 0],
            "preseed_frames": LEFT,
            "audio_len_in_row": True,
            "chunk_samples": CORE * 2 * SAMPLES_PER_FRAME,
            "sampling": {
                "temperature": 1.0,
                "top_k": RAS_TOP_K,
                "top_p": RAS_TOP_P,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        }
