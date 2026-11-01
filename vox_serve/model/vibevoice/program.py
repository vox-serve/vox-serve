"""VibeVoice-Realtime-0.5B graph program (``microsoft/VibeVoice-Realtime-0.5B``; modules in ``model.py``).

The reference (``VibeVoiceStreamingForConditionalGenerationInference.generate``) interleaves windows of 5
text tokens with 6 speech latents over four KV contexts: the lower 4 layers on text only (``lm``), the upper
20 layers on text hidden states and speech (``tts``), and a negative upper context that only sees speech
(``neg_tts``, the classifier-free-guidance condition). Here every LM step advances each request by one
position, so a window's text tokens go in one per step (the same sequence of rows, causally equivalent):

* text step: the next text token through the lower layers, then one ``tts`` row (its hidden state + the text
  type embedding);
* speech step: the next latent from the diffusion head (DPM-Solver++, ``ddpm_steps`` steps, CFG between the
  ``tts`` and ``neg_tts`` conditions), emitted as an audio frame, and fed back as one ``tts`` row and one
  ``neg_tts`` row (acoustic connector + the speech type embedding). The EOS classifier on the new ``tts``
  hidden state ends the request after this frame (the next step emits ``TOKEN_STOP``).

The runtime's paged KV cache holds the ``tts`` context: request positions are ``tts`` positions. The prompt is
the voice's ``tts`` rows; their K/V are copied from the voice prompt (precomputed caches, no forward). The
lower-layer and negative contexts live in per-slot buffers of this program with masked dense attention;
per-slot state (text tokens, window counters, the two diffusion conditions) is updated in the graph.

Rows have 64 columns, the latent size: an audio frame's columns are the fp32 bits of its 64-d latent, which
the detokenizer reads back (no lookup into LM-side state that the slot's next request could overwrite). Prompt
rows carry the voice prompt row in column 0 and the request's text tokens, 63 per row, in columns 1..63; the
mask holds ``[kind, n_text, voice]``.

The detokenizer is the causal acoustic decoder with each conv's history kept per detok slot (the reference's
frame-by-frame streaming decode, in chunks of ``detok_interval`` frames of 3200 samples).
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import flashinfer
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download

from ...rt.core import Arena, EngineSpecBuilder, RtAttention
from ...rt.kernels import ops
from ...rt.program import TOKEN_AUDIO, TOKEN_OTHER, TOKEN_STOP, LMProgram, pcm16
from ...rt.prompt import Prompt
from ...utils import get_logger
from .model import (
    REPO,
    VibeVoiceRealtime,
    dpm_solver_plan,
    load_voice,
    match_voice,
    resolve_voices,
    sample_latent,
    voice_path,
)

logger = get_logger(__name__)

PROMPT, DECODE = 1, 2
TEXT_WINDOW, SPEECH_WINDOW = 5, 6
NOISE_SALT = 1
TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B"  # the processor's ``language_model_pretrained_name``
DEFAULT_VOICE = "en-Carter_man"


def rope_cache(max_pos: int, head_dim: int, theta: float) -> torch.Tensor:
    """FlashInfer cos / sin cache ``[max_pos, head_dim]`` (fp32 holding bf16-rounded values, as HF Qwen2
    casts its cos / sin to the model dtype)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = torch.einsum("i,j->ij", torch.arange(max_pos, dtype=torch.float), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16).float()


class VibeVoiceProgram(LMProgram):
    uses_input_mask = True
    uses_repetition_penalty = False
    detok_stateful = True
    supports_urgent_lane = True

    def load(self) -> None:
        name = self.config.get("model", REPO)
        if "/" not in name:
            name = REPO
        self.model_name = name
        model_dir = snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])
        vv, cfg = VibeVoiceRealtime.from_pretrained(model_dir)
        decoder = vv.decoder
        vv.decoder = None
        self.vv = vv.to(self.device, torch.bfloat16).eval()
        self.decoder = decoder.to(self.detok_device, torch.bfloat16).eval()
        self.hop = decoder.hop
        self.dt_scale = vv.speech_scaling_factor.to(self.detok_device, torch.bfloat16)
        self.dt_bias = vv.speech_bias_factor.to(self.detok_device, torch.bfloat16)

        self.model = SimpleNamespace(default_sampling_config=SimpleNamespace(max_tokens=self.max_context))
        self.n_codebooks = vv.latent_dim
        self.vocab_size = vv.vocab_size
        self.num_layers = len(vv.tts_layers)
        self.num_qo_heads, self.num_kv_heads, self.head_dim = vv.n_q, vv.n_kv, vv.head_dim
        self.hidden_size = vv.hidden_size
        self.detok_interval = int(self.config.get("detok_interval") or 4)
        if not 1 <= self.detok_interval <= 10:  # the row's last int16 holds the sample count
            raise ValueError("VibeVoice detok_interval must be in [1, 10] frames")
        self.detok_samples = self.detok_interval * self.hop + 1
        self.max_text = int(self.config.get("max_text_tokens", 1024))

        self.cfg_scale = float(self.config.get("cfg_scale", 1.5))
        self.ddpm_steps = int(self.config.get("ddpm_steps", 5))
        self.plan = dpm_solver_plan(cfg["diffusion_head_config"], self.ddpm_steps)
        head = self.vv.prediction_head
        with torch.no_grad():
            t = torch.tensor([s["t"] for s in self.plan], dtype=torch.bfloat16, device=self.device)
            self.t_emb = head.t_embed(t)  # [steps, H]: the reference embeds bf16 timesteps

        self.tokenizer_json = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
        self._load_voices()
        logger.info(
            "VibeVoice-Realtime: %d voices, cfg %.2f, %d DPM steps, %d frames per chunk, max context %d",
            len(self.voice_names),
            self.cfg_scale,
            self.ddpm_steps,
            self.detok_interval,
            self.max_context,
        )

    def _load_voices(self) -> None:
        """Packs the voice prompts: ``tts`` rows of every voice back to back (prompt rows index them), the
        lower-layer and negative contexts padded per voice, and the two initial diffusion conditions."""
        names = resolve_voices(self.config.get("voices"))
        voices = [load_voice(voice_path(n, self.config.get("voices_dir"))) for n in names]
        self.voice_names = names
        dev, dt = self.device, torch.bfloat16
        self.voice_rows: Dict[str, Dict[str, int]] = {}
        off = 0
        for i, (n, v) in enumerate(zip(names, voices, strict=True)):
            L = v["tts_lm_k"].shape[1]
            self.voice_rows[n] = {"index": i, "offset": off, "len": L}
            off += L
        self.preset_tk = torch.cat([v["tts_lm_k"] for v in voices], 1).to(dev, dt)  # [layers, rows, kv, hd]
        self.preset_tv = torch.cat([v["tts_lm_v"] for v in voices], 1).to(dev, dt)

        def padded(key: str) -> torch.Tensor:
            L = max(v[key].shape[1] for v in voices)
            return torch.stack([F.pad(v[key], (0, 0, 0, 0, 0, L - v[key].shape[1])) for v in voices]).to(dev, dt)

        self.preset_bk, self.preset_bv = padded("lm_k"), padded("lm_v")  # [V, layers, L, kv, hd]
        self.preset_nk, self.preset_nv = padded("neg_tts_lm_k"), padded("neg_tts_lm_v")
        self.preset_blen = torch.tensor([v["lm_k"].shape[1] for v in voices], dtype=torch.int32, device=dev)
        self.preset_nlen = torch.tensor([v["neg_tts_lm_k"].shape[1] for v in voices], dtype=torch.int32, device=dev)
        self.preset_pos = torch.stack([v["tts_lm_h"] for v in voices]).to(dev, dt)
        self.preset_neg = torch.stack([v["neg_tts_lm_h"] for v in voices]).to(dev, dt)
        if max(r["len"] for r in self.voice_rows.values()) >= self.max_context:
            raise ValueError("a voice prompt does not fit in max_context")

    # ---- state ------------------------------------------------------------------------------------

    def declare(self, builder: EngineSpecBuilder) -> None:
        super().declare(builder)
        lanes = [(builder.dt_inputs, self.max_detok_batch_size)]
        if builder.urgent_lane:
            lanes.append((builder.du_inputs, max(self._urgent_sizes())))
        for dinp, nb in lanes:
            dinp.add("dt_len", (nb,), torch.int32)  # valid frames of the window (the last chunk may be short)
            # Unused (no look-ahead; with no overlap the chunk reaching the last frame is the last one), but
            # it makes the Python runtime fill dt_len too.
            dinp.add("dt_last", (nb,), torch.int32)

    def allocate_state(self) -> None:
        super().allocate_state()
        dev, dt = self.device, torch.bfloat16
        S1 = self.max_slots + 1
        H, kv, hd = self.hidden_size, self.num_kv_heads, self.head_dim
        i32 = dict(dtype=torch.int32, device=dev)
        self.text_buf = torch.zeros(S1, self.max_text, **i32)
        self.text_len = torch.zeros(S1, **i32)
        self.text_idx = torch.zeros(S1, **i32)
        self.tw_left = torch.zeros(S1, **i32)  # text tokens left in the current window
        self.sp_left = torch.zeros(S1, **i32)  # speech steps left in the current window
        self.stop_next = torch.zeros(S1, **i32)
        self.blen = torch.zeros(S1, **i32)
        self.nlen = torch.zeros(S1, **i32)
        self.lb = self.preset_bk.shape[2] + self.max_text
        self.ln = self.preset_nk.shape[2] + self.max_context
        n_lm = len(self.vv.lm_layers)
        self.bk = torch.zeros(n_lm, S1, self.lb, kv, hd, dtype=dt, device=dev)
        self.bv = torch.zeros_like(self.bk)
        self.nk = torch.zeros(self.num_layers, S1, self.ln, kv, hd, dtype=dt, device=dev)
        self.nv = torch.zeros_like(self.nk)
        self.pos_cond = torch.zeros(S1, H, dtype=dt, device=dev)
        self.neg_cond = torch.zeros(S1, H, dtype=dt, device=dev)
        self.cos_sin = rope_cache(max(self.max_context, self.lb, self.ln) + 1, hd, self.vv.rope_theta).to(dev)
        self.noise = torch.zeros(self.max_batch_size, self.n_codebooks, dtype=torch.float32, device=dev)
        max_rows = max(self.prefill_buckets + [self.max_batch_size])
        self.row_in = torch.zeros(max_rows + 1, H, dtype=dt, device=dev)
        self.cols = torch.arange(self.n_codebooks - 1, device=dev)
        self.ar = torch.arange(max(self.lb, self.ln), device=dev)
        self.dt_states = [
            torch.zeros(self.detok_slots + 1, c, n, dtype=dt, device=self.detok_device)
            for c, n in self.decoder.state_shapes()
        ]
        self.dt_ar = torch.arange(self.detok_interval * self.hop, device=self.detok_device)

    # ---- LM step ----------------------------------------------------------------------------------

    def _rope(self, pos: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> None:
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(pos, q, k, self.head_dim, self.cos_sin, is_neox=True)

    def _dense_attn(self, kc, vc, q, k, v, slot, w_slot, w_pos, length):
        """Attention over a per-slot cache kc / vc [S + 1, L, kv, hd]: writes k / v at (w_slot, w_pos), then
        attends to the first ``length`` positions of each row's slot."""
        n, kv, hd = q.shape[0], self.num_kv_heads, self.head_dim
        kc.index_put_((w_slot, w_pos), k.view(n, kv, hd))
        vc.index_put_((w_slot, w_pos), v.view(n, kv, hd))
        L = kc.shape[1]
        keys = kc.index_select(0, slot).transpose(1, 2)  # [n, kv, L, hd]
        vals = vc.index_select(0, slot).transpose(1, 2)
        mask = (self.ar[None, :L] < length.clamp(min=1)[:, None])[:, None, None, :]
        # The q heads of one kv head (grouped-query) are its query rows.
        o = F.scaled_dot_product_attention(q.view(n, kv, -1, hd), keys, vals, attn_mask=mask)
        return o.reshape(n, -1)

    def _sample_latent(self, cond: torch.Tensor, neg: torch.Tensor, arena: Arena, n: int) -> torch.Tensor:
        """The next latent from Philox noise keyed by (seed, step), independent of graph replays."""
        noise = self.noise[:n]
        ops().randn(noise, arena["seed"][:n], arena["step"][:n], NOISE_SALT)
        x = noise.to(torch.bfloat16)
        return sample_latent(self.vv.prediction_head, self.plan, self.t_emb, cond, neg, x, self.cfg_scale)

    def lm_step(self, arena: Arena, out: Arena, rows: int, n_req: int, attn: RtAttention) -> None:
        R, N, S = rows, n_req, self.max_slots
        vv = self.vv
        tok = arena["tok"][:R]
        mask = arena["mask"][:R]
        pos = arena["pos"][:R]
        prompt = mask[:, 0] == PROMPT
        slot = arena["slot"][:N].long()
        active = arena["active"][:N] != 0
        first = arena["first"][:N] != 0
        last_row = arena["last_row"][:N].long()
        dummy = torch.full_like(slot, S)

        if attn.kind == "prefill":
            # Prompt rows: the text tokens they carry.
            row_slot = arena["row_slot"][:R].long()
            t_idx = pos.long()[:, None] * (self.n_codebooks - 1) + self.cols[None]
            ok = prompt[:, None] & (t_idx < self.max_text)
            dest = torch.where(ok, row_slot[:, None] * self.max_text + t_idx, S * self.max_text)
            self.text_buf.view(-1).index_put_((dest.reshape(-1),), tok[:, 1:].reshape(-1))
            # New requests: the voice prompt's lower-layer and negative contexts, conditions and counters.
            s0 = torch.where(active & first, slot, dummy)
            lr = mask.index_select(0, last_row)
            n_text, voice = lr[:, 1], lr[:, 2].long().clamp(0, len(self.voice_names) - 1)
            self.text_len.index_copy_(0, s0, n_text)
            self.text_idx.index_copy_(0, s0, torch.zeros_like(n_text))
            self.tw_left.index_copy_(0, s0, n_text.clamp(max=TEXT_WINDOW))
            self.sp_left.index_copy_(0, s0, torch.full_like(n_text, SPEECH_WINDOW))
            self.stop_next.index_copy_(0, s0, torch.zeros_like(n_text))
            self.blen.index_copy_(0, s0, self.preset_blen[voice])
            self.nlen.index_copy_(0, s0, self.preset_nlen[voice])
            self.pos_cond.index_copy_(0, s0, self.preset_pos[voice])
            self.neg_cond.index_copy_(0, s0, self.preset_neg[voice])
            lb, ln = self.preset_bk.shape[2], self.preset_nk.shape[2]
            self.bk[:, :, :lb].index_copy_(1, s0, self.preset_bk[voice].transpose(0, 1))
            self.bv[:, :, :lb].index_copy_(1, s0, self.preset_bv[voice].transpose(0, 1))
            self.nk[:, :, :ln].index_copy_(1, s0, self.preset_nk[voice].transpose(0, 1))
            self.nv[:, :, :ln].index_copy_(1, s0, self.preset_nv[voice].transpose(0, 1))

        dec = active & ~first
        stop = self.stop_next[slot] != 0
        twl = self.tw_left[slot]
        is_text = dec & ~stop & (twl > 0)
        is_speech = dec & ~stop & (twl == 0)
        is_stop = dec & stop

        # Speech: the next latent and its feedback row.
        latent = self._sample_latent(self.pos_cond[slot], self.neg_cond[slot], arena, N)
        types = vv.tts_input_types.weight
        e_speech = vv.acoustic_connector(latent) + types[0]

        # Text: the next token through the lower layers.
        tidx = self.text_idx[slot]
        h = vv.embed_tokens(self.text_buf[slot, tidx.clamp(max=self.max_text - 1)].long())
        bpos = self.blen[slot]
        bw, bwpos = torch.where(is_text, slot, dummy), torch.where(is_text, bpos, 0)
        blen_new = bpos + is_text.int()
        for i, layer in enumerate(vv.lm_layers):
            q, k, v = layer.qkv(h)
            self._rope(bpos, q, k)
            h = layer.finish(h, self._dense_attn(self.bk[i], self.bv[i], q, k, v, slot, bw, bwpos, blen_new))
        e_text = h + types[1]

        # Upper layers: the requests' tts rows (in the runtime's rows) and neg_tts rows (appended).
        row_in = self.row_in[: R + 1]
        row_in.zero_()
        row_in.index_copy_(0, torch.where(dec, last_row, R), torch.where(is_text[:, None], e_text, e_speech))
        h = torch.cat([row_in[:R], e_speech])
        npos = self.nlen[slot]
        nw, nwpos = torch.where(is_speech, slot, dummy), torch.where(is_speech, npos, 0)
        nlen_new = npos + is_speech.int()
        all_pos = torch.cat([pos, npos])
        vrow = tok[:, 0].long().clamp(0, self.preset_tk.shape[1] - 1)
        kv, hd = self.num_kv_heads, self.head_dim
        for i, layer in enumerate(vv.tts_layers):
            q, k, v = layer.qkv(h)
            self._rope(all_pos, q, k)
            km = torch.where(prompt[:, None, None], self.preset_tk[i, vrow], k[:R].view(R, kv, hd))
            vm = torch.where(prompt[:, None, None], self.preset_tv[i, vrow], v[:R].view(R, kv, hd))
            attn.set_kv_cache(self.kv_cache[i], km, vm)
            om = attn.run(q[:R].view(R, self.num_qo_heads, hd), self.kv_cache[i]).reshape(R, -1)
            on = self._dense_attn(self.nk[i], self.nv[i], q[R:], k[R:], v[R:], slot, nw, nwpos, nlen_new)
            h = layer.finish(h, torch.cat([om, on]))
        new_pos = vv.tts_norm(h.index_select(0, last_row))
        new_neg = vv.tts_norm(h[R:])
        eos = torch.sigmoid(vv.eos(new_pos))[:, 0] > 0.5

        # Per-slot state.
        upd = torch.where(dec, slot, dummy)
        self.pos_cond.index_copy_(0, torch.where(is_text | is_speech, slot, dummy), new_pos)
        self.neg_cond.index_copy_(0, nw, new_neg)
        tidx_new = tidx + is_text.int()
        twl_new = twl - is_text.int()
        spl = self.sp_left[slot] - is_speech.int()
        wrap = is_speech & (spl == 0)
        twl_new = torch.where(wrap, (self.text_len[slot] - tidx_new).clamp(min=0, max=TEXT_WINDOW), twl_new)
        spl = torch.where(wrap, SPEECH_WINDOW, spl)
        self.text_idx.index_copy_(0, upd, tidx_new)
        self.tw_left.index_copy_(0, upd, twl_new)
        self.sp_left.index_copy_(0, upd, spl)
        self.blen.index_copy_(0, upd, blen_new)
        self.nlen.index_copy_(0, upd, nlen_new)
        self.stop_next.index_copy_(0, upd, (stop | (is_speech & eos)).int())

        out_tok = out["out_tok"][:N]
        out_tok.copy_(torch.where(is_speech[:, None], latent.float().view(torch.int32), 0))
        kind = torch.where(is_speech, TOKEN_AUDIO, torch.where(is_stop, TOKEN_STOP, TOKEN_OTHER))
        out["out_kind"][:N].copy_(kind.to(torch.int32))
        self.last_tok.index_copy_(0, slot, out_tok)

    # ---- detokenizer ------------------------------------------------------------------------------

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        latent = arena["dt_tok"][:bd].view(torch.float32).to(torch.bfloat16)  # [bd, frames, 64]
        scaled = latent / self.dt_scale - self.dt_bias
        slots = arena["dt_slot"][:bd].long()
        keep = (arena["dt_reset"][:bd] == 0)[:, None, None]
        states = [torch.where(keep, st.index_select(0, slots), 0) for st in self.dt_states]
        wav, states = self.decoder(scaled.transpose(1, 2), states)
        for st, s in zip(self.dt_states, states, strict=True):
            st.index_copy_(0, slots, s)
        n = arena["dt_len"][:bd].long() * self.hop
        audio = out["audio"][:bd]
        audio[:, :-1].copy_(pcm16(wav.reshape(bd, -1) * (self.dt_ar[None] < n[:, None])))
        audio[:, -1].copy_(n.to(torch.int16))

    # ---- prompt -----------------------------------------------------------------------------------

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """The voice's ``tts`` prompt rows ``[voice row, 63 text tokens]``, mask ``[PROMPT, n_text, voice]``;
        the text is the reference processor's ``text.strip() + "\\n"`` without special tokens."""
        if audio_path is not None:
            raise ValueError("VibeVoice-Realtime uses its voice prompts and does not clone reference audio")
        m = self.spec_meta
        voices = m["voices"]
        requested = model_kwargs.get("voice") or model_kwargs.get("speaker")
        name = m["default_voice"]
        if isinstance(requested, str) and requested.strip():
            name = match_voice(requested, list(voices))
            if name is None:
                raise ValueError(f"unknown VibeVoice voice {requested!r}; available: {', '.join(voices)}")
        text = text.replace("’", "'").replace("“", '"').replace("”", '"').strip()
        ids = self._encode(text + "\n", add_special_tokens=False)
        if not text:
            raise ValueError("no text provided")
        if len(ids) > m["max_text_tokens"]:
            raise ValueError(f"text of {len(ids)} tokens exceeds max_text_tokens {m['max_text_tokens']}")
        C = m["n_codebooks"]
        per = C - 1
        v = voices[name]
        tokens: List[int] = []
        mask: List[int] = []
        for p in range(v["len"]):
            chunk = ids[p * per : (p + 1) * per]
            tokens += [v["offset"] + p] + chunk + [0] * (per - len(chunk))
            mask += [PROMPT, len(ids), v["index"]] + [0] * (C - 3)
        return Prompt(tokens, mask)

    def meta(self) -> Dict[str, Any]:
        return {
            "kind": "vibevoice",
            "prompt_builder": "python",
            "model_name": self.model_name,
            "tokenizer_json": self.tokenizer_json,
            "sample_rate": 24000,
            "max_tokens": self.model.default_sampling_config.max_tokens,
            "detok_overlap": 0,
            "audio_len_in_row": True,
            "decode_mask": [DECODE] + [0] * (self.n_codebooks - 1),
            "voices": self.voice_rows,
            "default_voice": DEFAULT_VOICE if DEFAULT_VOICE in self.voice_rows else self.voice_names[0],
            "max_text_tokens": self.max_text,
            "cfg_scale": self.cfg_scale,
            "ddpm_steps": self.ddpm_steps,
            "sampling": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0},
        }
