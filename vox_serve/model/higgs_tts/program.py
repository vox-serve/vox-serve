"""Higgs Audio v3 TTS (``bosonai/higgs-tts-3-4b``) graph program: a Qwen3-4B backbone over text tokens
and the 8 delayed codebooks of the Higgs Audio v2 tokenizer (25 frames/s, 24 kHz).

Rows are ``tok = [id, 0, ...]`` with mask column 0 = ``TEXT`` (the text embedding), or eight codes
with mask ``AUDIO``: the sum of their rows of the fused ``[8 x 1026, H]`` codebook table, which is also
the output head (codes 1024 / 1025 are BOC / EOC). The prompt is SGLang-Omni's
``<|tts|> [<|ref_text|> ref_text] [<|ref_audio|> delayed ref codes] <|text|> text <|audio|>``; the
reference clip is encoded in fp32 by ``hook_encode_ref`` (the RVQ cascade diverges in bf16).

Sampling follows SGLang-Omni's ``sampler.step``: at step ``s < 8`` codebooks ``c > s`` are BOC; from
step 8 the first EOC in codebook 0, at step ``e``, ends the request after row ``e + 6`` (the other
codebooks finish their delayed frames), so the output has ``e + 7`` rows, i.e. ``e`` frames. Later
steps return -1. RoPE uses bf16-rounded cos / sin, as the model was trained (SGLang-Omni does the same).

Detokenization (stateless, windowed): a chunk is ``ROWS`` delayed rows starting every ``CORE`` rows,
which un-delay into ``WINDOW`` frames: ``LEFT`` frames of context, ``CORE`` emitted frames and ``AHEAD``
frames of look-ahead (the DAC decoder is non-causal; 12 / 8 frames reproduce a whole-sequence decode
to ~50 dB SNR in fp32; the bf16 decoder limits it to ~35-40 dB). The first chunk is pre-seeded so it
emits only ``first_chunk_frames``. Frames that do not exist (pre-seed, past the end) decode from a zero
latent, which comes closest to a whole-sequence decode's zero padding; codes >= 1024 decode as 0, as in
the reference. Audio rows hold up to ``CORE`` frames; the last int16 of a row is its valid length.
"""

import hashlib
import json
import os
import threading
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import flashinfer
import torch
import torch.nn.functional as F
from torch import nn

from ...rt.core import Arena, EngineSpecBuilder, RtAttention
from ...rt.program import TOKEN_AUDIO, TOKEN_STOP, LMProgram
from ...rt.prompt import Prompt
from ...tokenizer.higgs import CODEBOOK_SIZE, HOP, SAMPLE_RATE, HiggsAudioTokenizer
from ...utils import get_logger
from ..qwen3_tts.model import Qwen3TTSMLP, Qwen3TTSRMSNorm

logger = get_logger(__name__)

REPO = "bosonai/higgs-tts-3-4b"
TEXT, AUDIO = 1, 2
N_CB, CB_VOCAB = 8, 1026
BOC, EOC = 1024, 1025
NO_EOC = 1 << 30
MAX_REF_SECONDS = 100
SPECIALS = ("<|tts|>", "<|ref_text|>", "<|ref_audio|>", "<|text|>", "<|audio|>")

# chunk geometry (frames)
LEFT, CORE, AHEAD = 12, 25, 8
WINDOW = LEFT + CORE + AHEAD
ROWS = WINDOW + N_CB - 1


def rope_cache(max_pos: int, head_dim: int, theta: float) -> torch.Tensor:
    """FlashInfer cos / sin cache ``[max_pos, head_dim]`` (fp32 holding bf16-rounded values)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = torch.einsum("i,j->ij", torch.arange(max_pos, dtype=torch.float), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16).float()


def delay_pattern(codes: torch.Tensor) -> torch.Tensor:
    """``[T, 8]`` codes -> ``[T + 7, 8]`` delayed rows (codebook ``c`` shifted by ``c``, BOC / EOC padded)."""
    T = codes.shape[0]
    out = torch.full((T + N_CB - 1, N_CB), EOC, dtype=codes.dtype, device=codes.device)
    for c in range(N_CB):
        out[:c, c] = BOC
        out[c : c + T, c] = codes[:, c]
    return out


class HiggsAttention(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        H, hd = cfg.hidden_size, cfg.head_dim
        self.nq, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, hd
        self.q_proj = nn.Linear(H, self.nq * hd, bias=False)
        self.k_proj = nn.Linear(H, self.nkv * hd, bias=False)
        self.v_proj = nn.Linear(H, self.nkv * hd, bias=False)
        self.o_proj = nn.Linear(self.nq * hd, H, bias=False)
        self.q_norm = Qwen3TTSRMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(hd, eps=cfg.rms_norm_eps)

    def forward(self, x, pos, cos_sin, attn, kv_cache):
        n, hd = x.shape[0], self.hd
        # reshape: with --fuse-lm the projections are column slices of one GEMM output
        q = self.q_norm(self.q_proj(x).reshape(-1, hd)).view(n, -1)
        k = self.k_norm(self.k_proj(x).reshape(-1, hd)).view(n, -1)
        v = self.v_proj(x).view(n, self.nkv, hd)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(pos, q, k, hd, cos_sin, is_neox=True)
        attn.set_kv_cache(kv_cache, k.view(n, self.nkv, hd), v)
        return self.o_proj(attn.run(q.view(n, self.nq, hd), kv_cache).reshape(n, -1))


class HiggsLayer(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.self_attn = HiggsAttention(cfg)
        self.mlp = Qwen3TTSMLP(cfg)
        self.input_layernorm = Qwen3TTSRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, h, pos, cos_sin, attn, kv_cache):
        h = h + self.self_attn(self.input_layernorm(h), pos, cos_sin, attn, kv_cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class HiggsLM(nn.Module):
    """Backbone and embeddings in checkpoint naming (``body.*`` -> ``layers`` / ``norm``)."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.layers = nn.ModuleList([HiggsLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = Qwen3TTSRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.text_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.codebook_embedding = nn.Embedding(N_CB * CB_VOCAB, cfg.hidden_size)  # also the head
        self.cos_sin = None

    def forward(self, h, pos, attn, kv_cache):
        for i, layer in enumerate(self.layers):
            h = layer(h, pos, self.cos_sin, attn, kv_cache[i])
        return self.norm(h)

    @classmethod
    def from_checkpoint(cls, model_dir: str, device: torch.device, max_pos: int) -> "HiggsLM":
        from safetensors import safe_open

        tc = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
        cfg = SimpleNamespace(**tc)
        cfg.rope_theta = float((tc.get("rope_parameters") or {}).get("rope_theta", tc.get("rope_theta", 1e6)))
        with torch.device("meta"):
            lm = cls(cfg)
        rename = {
            "tied.embedding.text_embedding.weight": "text_embedding.weight",
            "tied.embedding.modality_embeddings.0.embedding.weight": "codebook_embedding.weight",
        }
        state = {}
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device=str(device)) as f:
            for name in f.keys():
                key = name[len("body.") :] if name.startswith("body.") else rename.get(name)
                if key is not None:
                    state[key] = f.get_tensor(name).to(torch.bfloat16)
        lm.load_state_dict(state, strict=True, assign=True)
        lm.cos_sin = rope_cache(max_pos, cfg.head_dim, cfg.rope_theta).to(device)
        lm.config = cfg
        return lm.eval().requires_grad_(False)


def load_wav(path: str) -> torch.Tensor:
    """Mono float32 at 24 kHz."""
    import soundfile as sf
    import torchaudio

    try:
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        w = torch.from_numpy(w).mean(1)
    except Exception:  # noqa: BLE001 - formats libsndfile cannot read (e.g. mp3)
        import librosa

        w, sr = librosa.load(path, sr=None, mono=True)
        w = torch.from_numpy(w)
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w, sr, SAMPLE_RATE)
    return w


class HiggsTTSProgram(LMProgram):
    uses_input_mask = True
    uses_repetition_penalty = False
    # dt_reset marks a request's first chunk (its pre-seeded frames); the codec itself keeps no state.
    detok_stateful = True
    supports_urgent_lane = True

    def load(self) -> None:
        from huggingface_hub import snapshot_download

        name = self.config.get("model", REPO)
        if "/" not in name:
            name = REPO
        self.model_name = name
        self.model_dir = snapshot_download(name)
        if not self.config.get("max_num_pages"):
            # 147 KB of KV per token: one context per batch entry, not the default two.
            self.max_num_pages = self.max_batch_size * -(-self.max_context // self.page_size) + 1
        self.model = SimpleNamespace(default_sampling_config=SimpleNamespace(max_tokens=self.max_context))

        self.lm = HiggsLM.from_checkpoint(self.model_dir, self.device, self.max_context + 1)
        cfg = self.lm.config
        self.n_codebooks = N_CB
        self.vocab_size = CB_VOCAB
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.text_vocab = cfg.vocab_size

        self.first_frames = int(self.config.get("first_chunk_frames", 12))
        if not 1 <= self.first_frames <= CORE:
            raise ValueError(f"first_chunk_frames must be in [1, {CORE}]")
        self.preseed = LEFT + CORE - self.first_frames
        self.detok_interval = ROWS
        self.detok_samples = CORE * HOP + 1

        dd = self.detok_device
        self.codec = HiggsAudioTokenizer.from_checkpoint(self.model_dir).to(dd)
        codec_dtype = getattr(torch, str(self.config.get("codec_dtype", "bfloat16")))
        self.codec.acoustic_decoder.to(codec_dtype)
        if self.config.get("fuse_codec"):
            from ...fusion.dac import fused_dac_decode, prepare

            prepare(self.codec.acoustic_decoder)
            self.codec.fused_decode = fused_dac_decode
        self.undelay = (torch.arange(WINDOW)[:, None] + torch.arange(N_CB)[None]).to(dd)

        from tokenizers import Tokenizer

        self.tokenizer_json = os.path.join(self.model_dir, "tokenizer.json")
        vocab = Tokenizer.from_file(self.tokenizer_json).get_vocab(with_added_tokens=True)
        missing = [s for s in SPECIALS if s not in vocab]
        if missing:
            raise RuntimeError(f"Higgs tokenizer lacks {missing}")
        self.special_ids = {s.strip("<|>"): vocab[s] for s in SPECIALS}
        logger.info(
            "Higgs TTS: %d KV pages, first chunk %d frames (pre-seed %d), codec %s%s",
            self.max_num_pages,
            self.first_frames,
            self.preseed,
            codec_dtype,
            " (fused)" if self.codec.fused_decode is not None else "",
        )

    # ---- LM ---------------------------------------------------------------------------------------

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        kind = arena["mask"][:rows, :1]
        text = self.lm.text_embedding(ids[:, 0].clamp(0, self.text_vocab - 1))
        codes = F.embedding(ids.clamp(0, CB_VOCAB - 1) + self.cb_offsets, self.lm.codebook_embedding.weight).sum(1)
        return torch.where(kind == AUDIO, codes, text)

    def fuse_lm_modules(self):
        return [self.lm]

    def fused_stack_spec(self):
        return dict(
            layers=list(self.lm.layers),
            norm=self.lm.norm,
            n_q=self.num_qo_heads,
            n_kv=self.num_kv_heads,
            head_dim=self.head_dim,
            cos_sin=self.lm.cos_sin,
        )

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        if self.fstack is not None:
            return self.fstack(hidden, pos, attn, self.kv_cache)
        return self.lm(hidden, pos, attn, self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm.codebook_embedding.weight).view(-1, N_CB, CB_VOCAB)

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        n = n_req
        draw = self.draw[:n]
        self.sample_into(self.head(hidden), arena, draw, n, self.LM_SALT)
        step = arena["step"][:n]
        slot = arena["slot"][:n].long()
        first = arena["first"][:n] != 0
        codes = torch.where(self.cb_cols[None] > step[:, None], BOC, draw)
        eoc = torch.where(first, NO_EOC, self.eoc_step[slot])
        eoc = torch.where((step >= N_CB) & (codes[:, 0] == EOC) & (eoc == NO_EOC), step, eoc)
        self.eoc_step.index_copy_(0, slot, eoc)
        out_tok.copy_(torch.where((step >= eoc + N_CB - 1)[:, None], -1, codes))

    # ---- state ------------------------------------------------------------------------------------

    def declare(self, builder: EngineSpecBuilder) -> None:
        super().declare(builder)
        lanes = [(builder.dt_inputs, self.max_detok_batch_size)]
        if builder.urgent_lane:
            lanes.append((builder.du_inputs, max(self._urgent_sizes())))
        for dinp, nb in lanes:
            dinp.add("dt_len", (nb,), torch.int32)  # valid rows of the window
            # Unused here (the window length says it), but declaring it makes the runtimes finish a request
            # only after its last chunk, which emits the frames earlier chunks kept as look-ahead.
            dinp.add("dt_last", (nb,), torch.int32)

    def allocate_state(self) -> None:
        super().allocate_state()
        dev = self.device
        self.eoc_step = torch.full((self.max_slots + 1,), NO_EOC, dtype=torch.int32, device=dev)
        self.draw = torch.zeros(self.max_batch_size, N_CB, dtype=torch.int32, device=dev)
        self.cb_cols = torch.arange(N_CB, dtype=torch.int32, device=dev)
        self.cb_offsets = torch.arange(N_CB, device=dev) * CB_VOCAB

    # ---- reference audio --------------------------------------------------------------------------

    @torch.no_grad()
    def hook_encode_ref(self, path: str) -> Dict[str, Any]:
        """Reference clip -> its delayed code rows, flattened (the prompt's AUDIO rows)."""
        wav = load_wav(path)
        if wav.numel() > MAX_REF_SECONDS * SAMPLE_RATE:
            raise ValueError(f"reference audio longer than {MAX_REF_SECONDS} s")
        with _no_tf32():
            codes = self.codec.encode(wav)
        rows = delay_pattern(codes).cpu()
        return {"rows": rows.shape[0], "codes": rows.flatten().tolist()}

    # ---- detokenizer ------------------------------------------------------------------------------

    def compile_detok_targets(self):
        return [(self.codec.acoustic_decoder, "forward")]

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        dd = self.detok_device
        rows = arena["dt_tok"][:bd].long()  # [bd, ROWS, 8]
        reset = arena["dt_reset"][:bd] != 0
        dlen = arena["dt_len"][:bd].long()
        codes = rows.gather(1, self.undelay[None].expand(bd, -1, -1))  # frame i, codebook c = row i + c
        codes = torch.where(codes >= CODEBOOK_SIZE, 0, codes)
        frame = torch.arange(WINDOW, device=dd)[None]
        # A frame exists once all its rows do; the first chunk starts with pre-seeded frames.
        real = (frame + N_CB - 1 < dlen[:, None]) & (~reset[:, None] | (frame >= self.preseed))
        wav = self.codec.decode(codes, real)  # [bd, WINDOW * HOP]
        e0 = torch.where(reset, max(LEFT, self.preseed), LEFT)
        e1 = torch.maximum((dlen - (N_CB - 1)).clamp(max=LEFT + CORE), e0)
        j = torch.arange(CORE * HOP, device=dd)
        pcm = wav.gather(1, (e0[:, None] * HOP + j).clamp(max=WINDOW * HOP - 1))
        n = (e1 - e0) * HOP
        pcm = pcm.clamp(-1, 1) * (j < n[:, None])
        audio = out["audio"][:bd]
        # Truncation toward zero, matching numpy's (x * 32767).astype(np.int16).
        audio[:, :-1].copy_((pcm * 32767).to(torch.int16))
        audio[:, -1].copy_(n.to(torch.int16))

    # ---- meta -------------------------------------------------------------------------------------

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """Rows ``[id, 0, ..]`` (text) or eight delayed codes (audio): ``<|tts|> [<|ref_text|> ref_text]
        [<|ref_audio|> ref codes] <|text|> text <|audio|>``, the brackets only with reference audio."""
        ids = self.spec_meta["special_ids"]
        tokens: List[int] = []
        mask: List[int] = []

        def text_row(t: int) -> None:
            tokens.extend([t] + [0] * (N_CB - 1))
            mask.extend([TEXT] + [0] * (N_CB - 1))

        text_row(ids["tts"])
        if audio_path is not None:
            codes = self._ref_codes(audio_path)
            ref_text = model_kwargs.get("ref_text")
            # The transcript only conditions together with the clip (as in the reference).
            if isinstance(ref_text, str) and ref_text.strip():
                text_row(ids["ref_text"])
                for t in self._encode(ref_text.strip(), add_special_tokens=False):
                    text_row(t)
            text_row(ids["ref_audio"])
            tokens.extend(codes)
            mask.extend(([AUDIO] + [0] * (N_CB - 1)) * (len(codes) // N_CB))
        text_row(ids["text"])
        for t in self._encode(text, add_special_tokens=False):
            text_row(t)
        text_row(ids["audio"])
        return Prompt(tokens, mask)

    _ref_cache: "Optional[OrderedDict[str, List[int]]]" = None
    _ref_lock = threading.Lock()

    def _ref_codes(self, path: str) -> List[int]:
        """Delayed code rows of reference clip ``path`` (an LRU of 16 clips, keyed by content)."""
        with open(path, "rb") as fh:
            key = hashlib.sha1(fh.read()).hexdigest()
        with self._ref_lock:
            if self._ref_cache is None:
                self._ref_cache = OrderedDict()
            if key in self._ref_cache:
                self._ref_cache.move_to_end(key)
                return self._ref_cache[key]
            codes = [int(c) for c in self.hook_encode_ref(path)["codes"]]
            if not codes or len(codes) % N_CB:
                raise RuntimeError(f"encode_ref returned {len(codes)} codes")
            self._ref_cache[key] = codes
            if len(self._ref_cache) > 16:
                self._ref_cache.popitem(last=False)
            return codes

    def token_kind(self, tok: torch.Tensor) -> torch.Tensor:
        # decode_tokens writes -1 rows once the request has finished.
        return torch.where(tok[:, 0] < 0, TOKEN_STOP, TOKEN_AUDIO).to(torch.int32)

    def meta(self) -> Dict[str, Any]:
        return {
            "kind": "higgs_tts",
            "model_name": self.model_name,
            "tokenizer_json": self.tokenizer_json,
            "special_ids": self.special_ids,
            "sample_rate": SAMPLE_RATE,
            "detok_overlap": ROWS - CORE,
            "max_tokens": self.model.default_sampling_config.max_tokens,
            "row_kinds": {"text": TEXT, "audio": AUDIO},
            "decode_mask": [AUDIO] + [0] * (N_CB - 1),
            "preseed_frames": self.preseed,
            "audio_len_in_row": True,
            "chunk_samples": CORE * HOP,
            "sampling": {
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 1.0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        }


class _no_tf32:
    """fp32 matmuls / convolutions for the codec encoder (TF32 flips RVQ code choices)."""

    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved
