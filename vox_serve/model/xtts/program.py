"""XTTS-v2 (``coqui/XTTS-v2``) graph program: a GPT-2 over 32 conditioning latents, the BPE text and DVAE codes,
and a HiFi-GAN vocoder that decodes the GPT's hidden states (one 1024-d *latent* per code) to 24 kHz audio.

Rows carry more than token ids, because the vocoder needs data the runtimes do not keep per request: the
latents (needed until the request's last chunk, while the LM slot and KV pages are released when generation
ends) and the 512-d speaker embedding. Every row is ``C = 771`` int32 columns::

    0 TOK      text token id (TEXT rows) or code (AUDIO rows)
    1 KIND     COND / TEXT / AUDIO
    2 POS      learned position: text position, or the code's position (start code 0, k-th sampled code k)
    3..514     LAT, 1024 fp16 values: a conditioning latent (COND rows) or, in sampled rows, the latent of the
               hidden state that sampled the code
    515..770   SPK, the speaker embedding in fp16, carried by the prompt's last row and every sampled row

The prompt is ``32 COND rows | [START] text [STOP] (TEXT, positions 0..) | start code (AUDIO, position 0)``,
as Coqui's ``GPT.compute_embeddings``; sampled rows are fed back whole (``last_tok``), so the position and
speaker embedding travel with the request. A request stops one step after sampling the stop code, whose
latent the vocoder also decodes (as Coqui's streaming and non-streaming inference do), or after
``max_gen_tokens`` codes. The repetition penalty also covers code 1 and the start code, the placeholder ids
Coqui's Hugging Face ``generate`` sees for the prompt.

Detokenization (stateless, windowed): a chunk is ``WINDOW`` latents starting every ``CORE``: ``LEFT`` of context,
``CORE`` emitted, ``AHEAD`` of look-ahead. The vocoder's two linear interpolations (latents x4, then 22.05 ->
24 kHz) run on the global mel-frame grid, so chunk boundaries fall on whole frames (256 samples) and a chunk
emits exactly the frames a whole-sequence decode puts there. The vocoder's receptive field (about 13 mel frames
each side) fits in the context, and a window holding the request's end zeroes the vocoder's activations after
it (a whole decode's zero padding), so the chunks reproduce a whole-sequence decode (~70 dB SNR). The first
chunk is pre-seeded so it emits ``first_chunk_frames`` latents. Audio rows hold up to ``cap`` samples; the last
int16 is the valid length.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...encoder.xtts import LRELU_SLOPE
from ...rt.core import Arena, EngineSpecBuilder, RtAttention
from ...rt.kernels import ops
from ...rt.program import TOKEN_AUDIO, TOKEN_STOP, LMProgram, pcm16
from ...rt.prompt import Prompt
from ...utils import get_logger
from .model import COND_LATENTS, GPT_SR, REPO, XTTSModel

logger = get_logger(__name__)

TEXT, AUDIO, COND = 1, 2, 3
TOK, KIND, POS, LAT = 0, 1, 2, 3
LAT_W = 512  # int32 columns of a 1024-d fp16 latent
SPK = LAT + LAT_W
SPK_W = 256  # int32 columns of the 512-d fp16 speaker embedding
C = SPK + SPK_W

# chunk geometry (latents)
LEFT, CORE, AHEAD = 4, 16, 4
WINDOW = LEFT + CORE + AHEAD
MAX_REF_SECONDS = 30  # Coqui's max_ref_length


def _bits(x: torch.Tensor) -> np.ndarray:
    """fp16 values -> their int32 columns (pairs of halves)."""
    return x.to(torch.float16).contiguous().view(torch.int32).numpy()


def load_wav(path: str) -> torch.Tensor:
    """Mono float32 ``[1, T]`` at 22.05 kHz, clipped to [-1, 1] (Coqui's ``load_audio``)."""
    import soundfile as sf
    import torchaudio

    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(w).mean(1)
    except Exception:  # noqa: BLE001 - formats libsndfile cannot read (e.g. mp3)
        import librosa

        w, sr = librosa.load(path, sr=None, mono=True)
        w = torch.from_numpy(w)
    if sr != GPT_SR:
        w = torchaudio.functional.resample(w, sr, GPT_SR)
    return w.clamp(-1, 1)[None]


class XTTSProgram(LMProgram):
    uses_repetition_penalty = True
    supports_urgent_lane = True

    def load(self) -> None:
        name = self.config.get("model", REPO)
        if "/" not in name:
            name = REPO
        self.model_name = name
        self.model = XTTSModel(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        m = self.model
        self.dtype = m.dtype
        self.default_sampling = m.default_sampling_config
        m.default_sampling_config.max_tokens = self.max_context
        self.n_codebooks = C
        self.vocab_size = m.num_audio_tokens
        self.num_layers = m.num_hidden_layers
        self.num_qo_heads = m.num_attention_heads
        self.num_kv_heads = m.num_key_value_heads
        self.head_dim = m.head_dim
        self.hidden_size = m.hidden_size
        if not self.config.get("prefill_buckets"):
            # Prompts are at most 32 + 404 + 1 rows; the input arena (C columns per row) is uploaded every step.
            self.prefill_buckets = [b for b in (128, 256, 512) if b <= self.max_context]

        # 22.05 kHz latents -> 24 kHz mel frames: x4 (code hop / mel hop), then x output_sr / input_sr.
        self.up = m.hop // m.mel_hop
        self.num, self.den = self.up * m.output_sr, m.input_sr
        self.scale2 = float(np.float32(1.0 / (m.output_sr / m.input_sr)))  # PyTorch's (fp32) source scale
        self.win_frames = -(-WINDOW * self.num // self.den) + 1
        self.cap = -(-(CORE + AHEAD) * self.num // self.den) * m.mel_hop
        self.first_frames = int(self.config.get("first_chunk_frames", 8))
        if not 1 <= self.first_frames <= CORE:
            raise ValueError(f"first_chunk_frames must be in [1, {CORE}]")
        self.preseed = LEFT + CORE - self.first_frames
        self.detok_interval = WINDOW
        self.detok_samples = self.cap + 1

        self.voice_names = list(m.voices)
        self.default_voice = self.voice_names[0]
        self._voice_bits = {name: (_bits(c), _bits(s)) for name, (c, s) in m.voices.items()}
        logger.info(
            "XTTS-v2: %d voices (default %s), first chunk %d latents (pre-seed %d), %d samples per audio row",
            len(self.voice_names),
            self.default_voice,
            self.first_frames,
            self.preseed,
            self.cap,
        )

    def _attention(self, kind: str, arena: Arena, batch: int, rows: int) -> RtAttention:
        return RtAttention(
            kind,
            arena,
            "lm_",
            self.float_workspace,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            batch_size=batch,
            max_rows=rows,
            dtype=self.dtype,
        )

    # ---- LM ---------------------------------------------------------------------------------------

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model
        tok, kind, pos = ids[:, TOK].long(), ids[:, KIND, None], ids[:, POS].long()
        text = m.text_embedding(tok.clamp(0, m.num_text_tokens - 1)) + m.text_pos_embedding(
            pos.clamp(0, m.max_text_positions - 1)
        )
        mel = m.mel_embedding(tok.clamp(0, m.num_audio_tokens - 1)) + m.mel_pos_embedding(
            pos.clamp(0, m.max_mel_positions - 1)
        )
        cond = ids[:, LAT : LAT + LAT_W].contiguous().view(torch.float16).to(self.dtype)
        return torch.where(kind == COND, cond, torch.where(kind == TEXT, text, mel))

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        m = self.model
        return m.final_norm(m.gpt(hidden, attn, self.kv_cache).float())  # the latents, fp32

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.mel_head(hidden)[:, None, :]

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        k = ops()
        logits = self.head(hidden)  # [N, 1, V]
        slot, active, first = arena["slot"][:n_req], arena["active"][:n_req], arena["first"][:n_req]
        k.rep_penalty_reset(self.rep_bits, 1, slot, active, first)
        # Hugging Face's repetition penalty also covers the prompt's input ids: Coqui fills them with 1 and
        # ends them with the start code. Penalize those from the first step on.
        for t in (1, self.model.start_audio_token):
            seed_tok = torch.where(first != 0, t, -1).to(torch.int32)[:, None]
            k.rep_penalty_update(self.rep_bits, 1, self.vocab_size, slot, active, seed_tok)
        k.rep_penalty_apply(logits, 1, self.rep_bits, slot, active, self.no_first[:n_req], arena["rep_pen"][:n_req])
        sampled = out_tok[:, :1]
        self.sample_into(logits, arena, sampled, n_req, self.LM_SALT)
        k.rep_penalty_update(self.rep_bits, 1, self.vocab_size, slot, active, sampled)

    def lm_step(self, arena: Arena, out: Arena, rows: int, n_req: int, attn: RtAttention) -> None:
        fb = arena["fb"][:rows]
        row_slot = arena["row_slot"][:rows].long()
        ids = torch.where(fb[:, None] != 0, self.last_tok[row_slot], arena["tok"][:rows])
        hidden = self.backbone(self.embed(ids, arena, rows), arena["pos"][:rows], attn)
        last = arena["last_row"][:n_req].long()
        h = hidden.index_select(0, last)
        prev = ids.index_select(0, last)  # the row each request's sample follows
        out_tok = out["out_tok"][:n_req]
        self.decode_tokens(h, arena, out_tok, n_req)
        out_tok[:, KIND] = AUDIO
        out_tok[:, POS] = prev[:, POS] + 1
        out_tok[:, LAT : LAT + LAT_W].copy_(h.to(torch.float16).contiguous().view(torch.int32))
        out_tok[:, SPK:].copy_(prev[:, SPK:])
        self.last_tok.index_copy_(0, arena["slot"][:n_req].long(), out_tok)
        # The row after the stop code ends the request, as does the code past Coqui's length limit.
        stop = (prev[:, TOK] == self.model.stop_audio_token) | (out_tok[:, POS] > self.model.max_gen_tokens)
        out["out_kind"][:n_req].copy_(torch.where(stop, TOKEN_STOP, TOKEN_AUDIO).to(torch.int32))

    # ---- state ------------------------------------------------------------------------------------

    def declare(self, builder: EngineSpecBuilder) -> None:
        super().declare(builder)
        lanes = [(builder.dt_inputs, self.max_detok_batch_size)]
        if builder.urgent_lane:
            lanes.append((builder.du_inputs, max(self._urgent_sizes())))
        for dinp, nb in lanes:
            dinp.add("dt_len", (nb,), torch.int32)  # valid latents of the window
            dinp.add("dt_last", (nb,), torch.int32)  # no chunk follows: emit through the end

    def allocate_state(self) -> None:
        super().allocate_state()
        dev, dd = self.device, self.detok_device
        # The penalty covers codes (column 0) only.
        self.rep_bits = torch.zeros(self.max_slots + 1, 1, (self.vocab_size + 31) // 32, dtype=torch.int32, device=dev)
        self.no_first = torch.zeros(self.max_batch_size, dtype=torch.int32, device=dev)
        # Latents -> x4 grid (PyTorch's linear interpolation, align_corners=False): a fixed pattern per window.
        q = torch.arange(self.up * WINDOW, dtype=torch.float32)
        src = ((q + 0.5) / self.up - 0.5).clamp(min=0)
        self.i1_lo = src.floor().long().to(dd)
        self.w1 = (src - src.floor()).to(dd)[None, :, None]
        self.frames = torch.arange(self.win_frames, device=dd)
        self.sample_idx = torch.arange(self.cap, device=dd)
        self.positions = torch.arange(WINDOW, device=dd)

    # ---- detokenizer ------------------------------------------------------------------------------

    def _mel_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Mel frames ending latent ``latents`` (global): the whole-sequence decode of that many latents has
        exactly this many frames."""
        return latents * self.num // self.den

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        m = self.model
        rows = arena["dt_tok"][:bd]  # [bd, WINDOW, C]
        dlen = arena["dt_len"][:bd].long().clamp(1, WINDOW)
        last = arena["dt_last"][:bd] != 0
        tail = rows.gather(1, (dlen - 1)[:, None, None].expand(-1, 1, C))[:, 0]  # the window's last row
        base = tail[:, POS].long() - dlen  # global latent of window position 0 (code k has latent k - 1)
        g = base.clamp(min=0)  # first real latent of the window
        n = (base + dlen - g).clamp(min=1)  # real latents from g
        # Window latents from g, repeating the last real one (the end, or padding past the request).
        pos = (self.positions[None].minimum(n[:, None] - 1) + (g - base)[:, None]).clamp(0, WINDOW - 1)
        lat = rows[:, :, LAT : LAT + LAT_W].contiguous().view(torch.float16).float()
        x = lat.gather(1, pos[:, :, None].expand(-1, -1, 2 * LAT_W))  # [bd, WINDOW, 1024]
        # x4 grid (relative to latent g), then the 24 kHz frames J0 + j of the global grid.
        i0 = self.i1_lo[None].minimum(n[:, None] - 1)
        i1 = (self.i1_lo[None] + 1).minimum(n[:, None] - 1)
        y = torch.lerp(
            x.gather(1, i0[:, :, None].expand(-1, -1, x.shape[2])),
            x.gather(1, i1[:, :, None].expand(-1, -1, x.shape[2])),
            self.w1,
        )
        j0 = self._mel_frames(g)
        src = ((j0[:, None] + self.frames[None]).float() + 0.5) * self.scale2 - 0.5
        src = src.clamp(min=0)
        k0 = src.floor().long()
        w2 = (src - k0)[:, :, None]
        end = self.up * (g + n)[:, None] - 1  # the grid's last point (the request's, at its end)
        k1 = (k0 + 1).minimum(end)
        k0 = k0.minimum(end)
        rel0 = (k0 - self.up * g[:, None]).clamp(0, y.shape[1] - 1)
        rel1 = (k1 - self.up * g[:, None]).clamp(0, y.shape[1] - 1)
        z = torch.lerp(
            y.gather(1, rel0[:, :, None].expand(-1, -1, y.shape[2])),
            y.gather(1, rel1[:, :, None].expand(-1, -1, y.shape[2])),
            w2,
        )  # [bd, frames, 1024]
        spk = tail[:, SPK:].contiguous().view(torch.float16).float()[:, :, None]  # [bd, 512, 1]
        # A window holding the request's end (only a request's last windows are incomplete) zero-pads after it.
        valid = torch.where(dlen < WINDOW, self._mel_frames(g + n) - j0, self.win_frames)
        wav = self._vocoder(z.transpose(1, 2), spk, valid)[:, 0]  # [bd, frames * 256]
        # Emitted frames: the core latents' (from the first latent on the first chunk), through the end on the
        # last chunk, and never past the latents the window has.
        a = base + LEFT
        hi = torch.where(last, base + dlen, torch.minimum(a + CORE, base + dlen))
        e_lo = self._mel_frames(a.clamp(min=0))
        e_hi = self._mel_frames(hi.clamp(min=0)).maximum(e_lo)
        hop = m.mel_hop
        off = (e_lo - j0) * hop
        n_out = ((e_hi - e_lo) * hop).clamp(max=self.cap)
        pcm = wav.gather(1, (off[:, None] + self.sample_idx[None]).clamp(0, wav.shape[1] - 1))
        audio = out["audio"][:bd]
        audio[:, :-1].copy_(pcm16(pcm * (self.sample_idx[None] < n_out[:, None])))
        audio[:, -1].copy_(n_out.to(torch.int16))

    def _vocoder(self, z: torch.Tensor, spk: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``HifiganGenerator.forward`` on mel frames ``z`` [bd, 1024, F] of which the first ``valid`` [bd] exist:
        every convolution sees zeros after them, as the whole-sequence decode's zero padding (in the window,
        whatever follows the request's end would leak into its last ~13 frames)."""
        dec = self.model.hifigan_decoder.waveform_decoder
        valid = valid[:, None, None]

        def masked(o: torch.Tensor, scale: int) -> torch.Tensor:
            t = torch.arange(o.shape[-1], device=o.device)
            return o * (t < valid * scale)

        o = dec.conv_pre(masked(z, 1)) + dec.cond_layer(spk)
        scale = 1
        for i, up in enumerate(dec.ups):
            o = up(masked(F.leaky_relu(o, LRELU_SLOPE), scale))
            scale *= up.stride[0]
            if dec.cond_in_each_up_layer:
                o = o + dec.conds[i](spk)
            z_sum = None
            for block in dec.resblocks[i * dec.num_kernels : (i + 1) * dec.num_kernels]:
                x = o
                for c1, c2 in zip(block.convs1, block.convs2, strict=True):
                    xt = c1(masked(F.leaky_relu(x, LRELU_SLOPE), scale))
                    x = c2(masked(F.leaky_relu(xt, LRELU_SLOPE), scale)) + x
                z_sum = x if z_sum is None else z_sum + x
            o = z_sum / dec.num_kernels
        o = dec.conv_post(masked(F.leaky_relu(o), scale))
        return torch.tanh(o)

    def compile_detok_targets(self):
        return [(self, "_vocoder")]

    # ---- prompt -----------------------------------------------------------------------------------

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """Rows ``32 COND | [START] text [STOP] | start code``; the conditioning comes from the reference clip,
        else from the named built-in voice (``voice`` or ``speaker``)."""
        m = self.model
        lang = str(model_kwargs.get("language") or "en").lower()
        if lang.split("-")[0] not in {code.split("-")[0] for code in m.languages}:
            raise ValueError(f"language {lang} is not supported (XTTS-v2: {', '.join(m.languages)})")
        bpe = m.tokenizer.encode(text, lang)
        if len(bpe) >= m.args.gpt_max_text_tokens:
            raise ValueError(f"text is too long: {len(bpe)} tokens (XTTS-v2 reads at most 401; split the text)")
        text_ids = [m.start_text_token, *bpe, m.stop_text_token]
        if audio_path is not None:
            cond, spk = self._ref_voice(audio_path)
        else:
            name = model_kwargs.get("voice") or model_kwargs.get("speaker") or self.default_voice
            if name not in self._voice_bits:
                raise ValueError(f"voice {name} is not available for this model")
            cond, spk = self._voice_bits[name]
        n_text = len(text_ids)
        rows = np.zeros((COND_LATENTS + n_text + 1, C), dtype=np.int32)
        rows[:COND_LATENTS, KIND] = COND
        rows[:COND_LATENTS, LAT : LAT + LAT_W] = cond
        t = slice(COND_LATENTS, COND_LATENTS + n_text)
        rows[t, TOK], rows[t, KIND], rows[t, POS] = text_ids, TEXT, np.arange(n_text)
        rows[-1, TOK], rows[-1, KIND], rows[-1, POS] = m.start_audio_token, AUDIO, 0
        rows[-1, SPK:] = spk
        return Prompt(rows.reshape(-1).tolist())

    _ref_cache: "Optional[OrderedDict[str, Tuple[np.ndarray, np.ndarray]]]" = None
    _ref_lock = threading.Lock()

    def _ref_voice(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Conditioning latents and speaker embedding of reference clip ``path``, as int32 columns (an LRU of 16
        clips, keyed by content)."""
        with open(path, "rb") as fh:
            key = hashlib.sha1(fh.read()).hexdigest()
        with self._ref_lock:
            if self._ref_cache is None:
                self._ref_cache = OrderedDict()
            if key in self._ref_cache:
                self._ref_cache.move_to_end(key)
                return self._ref_cache[key]
            wav = load_wav(path)[:, : GPT_SR * MAX_REF_SECONDS]
            cond, spk = self.model.conditioning(wav)
            bits = (_bits(cond), _bits(spk))
            self._ref_cache[key] = bits
            if len(self._ref_cache) > 16:
                self._ref_cache.popitem(last=False)
            return bits

    # ---- meta -------------------------------------------------------------------------------------

    def meta(self) -> Dict[str, Any]:
        sc = self.default_sampling
        return {
            "kind": "xtts",
            "prompt_builder": "python",
            "model_name": self.model_name,
            "sample_rate": self.model.output_sr,
            "detok_overlap": WINDOW - CORE,
            "max_tokens": sc.max_tokens,
            "preseed_frames": self.preseed,
            "audio_len_in_row": True,
            "chunk_samples": CORE * self.num * self.model.mel_hop // self.den,
            "voices": self.voice_names,
            "languages": self.model.languages,
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k,
                "top_p": sc.top_p,
                "min_p": sc.min_p,
                "repetition_penalty": sc.repetition_penalty,
            },
        }
