"""Higgs Audio v2 (``bosonai/higgs-audio-v2-generation-3B-base``) graph program: a Llama-3.2-3B backbone
whose layers have a second ("dual") FFN path for audio rows, over the 8 delayed codebooks of the Higgs
Audio v2 tokenizer (25 frames/s, 24 kHz).

It builds on the Higgs Audio v3 program (``higgs_tts``), which uses the same codec: rows, the delay
pattern, reference encoding and the windowed detokenizer are shared. What differs:

* backbone: Llama layers (no q / k norm, llama3-scaled RoPE). Every layer is dual-path: text rows take
  ``input_layernorm`` / ``mlp``, audio rows ``audio_input_layernorm`` / ``audio_mlp``, around the shared
  attention. Decode rows are all audio, so decode graphs run the audio path only.
* embeddings / head: the text embedding, an ``[8 x 1026, H]`` codebook table for audio rows (the sum of
  each row's eight codes) and a separate audio head of the same shape.
* prompt: the reference's Llama-3 chat format (the chat template of transformers' ``HiggsAudioV2Processor``,
  as Boson AI's ``examples/generation.py`` fills it): a system turn "Generate audio following
  instruction." with a scene description (``instruct``, default "Audio is recorded from a quiet room."),
  for voice cloning a user turn with ``ref_text`` and an assistant turn with the reference clip's delayed
  codes (with BOS / EOS frames), then the user turn with the (normalized) text and ``<|audio_out_bos|>``
  followed by one all-BOC row: the row the reference forces as its first generated one.
* sampling (transformers' ``HiggsAudioV2GenerationMixin``): temperature / top-k / top-p per codebook, then
  repetition-aware sampling (a code that already occurs ``ras_max`` times among its codebook's last
  ``ras_win`` rows is redrawn from the untempered, unfiltered distribution); codebooks ``c > step`` are BOC,
  and once any codebook has sampled EOC at step ``t``, codebooks ``c < step - t`` are forced to EOC. The
  request ends 7 rows after codebook 0's EOC (``e`` frames for an EOC at step ``e``, as in v3).

Deviations from the reference: the repetition window starts empty, where the reference's also covers the
last rows of the reference clip; and the output ends at codebook 0's EOC, where transformers' ``decode``
keeps one more frame (codebook 0's EOC clipped to code 1023; Boson AI's original decode drops it too).
"""

import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import flashinfer
import torch
import torch.nn.functional as F
from torch import nn

from ...rt.core import Arena, RtAttention
from ...rt.kernels import ops
from ...rt.prompt import Prompt
from ...tokenizer.higgs import HiggsAudioTokenizer
from ...utils import get_logger
from ..higgs_tts.program import (
    AUDIO,
    BOC,
    CB_VOCAB,
    EOC,
    N_CB,
    NO_EOC,
    SAMPLE_RATE,
    TEXT,
    HiggsTTSProgram,
    delay_pattern,
)
from ..qwen3_tts.model import Qwen3TTSMLP, Qwen3TTSRMSNorm

logger = get_logger(__name__)

REPO = "bosonai/higgs-audio-v2-generation-3B-base"
CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"
SYSTEM = "Generate audio following instruction."
DEFAULT_SCENE = "Audio is recorded from a quiet room."
SPECIALS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|audio_out_bos|>",
    "<|audio_eos|>",
    "<|scene_desc_start|>",
    "<|scene_desc_end|>",
)

# Boson AI's examples/generation.py text normalization.
_ZH_PUNCT = {
    "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")", "【": "[",
    "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'", "、": ",", "—": "-",
    "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
}  # fmt: skip
_SOUND_TAGS = (
    ("[laugh]", "<SE>[Laughter]</SE>"),
    ("[humming start]", "<SE_s>[Humming]</SE_s>"),
    ("[humming end]", "<SE_e>[Humming]</SE_e>"),
    ("[music start]", "<SE_s>[Music]</SE_s>"),
    ("[music end]", "<SE_e>[Music]</SE_e>"),
    ("[music]", "<SE>[Music]</SE>"),
    ("[sing start]", "<SE_s>[Singing]</SE_s>"),
    ("[sing end]", "<SE_e>[Singing]</SE_e>"),
    ("[applause]", "<SE>[Applause]</SE>"),
    ("[cheering]", "<SE>[Cheering]</SE>"),
    ("[cough]", "<SE>[Cough]</SE>"),
)


def normalize_text(text: str) -> str:
    """The target text as the reference example prepares it: half-width punctuation, no parentheses,
    temperature units spelled out, sound-effect tags, collapsed whitespace and a final period."""
    for zh, en in _ZH_PUNCT.items():
        text = text.replace(zh, en)
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    for tag, replacement in _SOUND_TAGS:
        text = text.replace(tag, replacement)
    text = "\n".join(" ".join(line.split()) for line in text.split("\n") if line.strip()).strip()
    if not text.endswith((".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>")):
        text += "."
    return text


def llama3_rope_cache(max_pos: int, head_dim: int, rope: Dict[str, Any]) -> torch.Tensor:
    """FlashInfer cos / sin cache ``[max_pos, head_dim]`` with llama3 frequency scaling (fp32 holding
    bf16-rounded values, as transformers' Llama computes cos / sin and casts them to the activations)."""
    inv_freq = 1.0 / (float(rope["rope_theta"]) ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    if rope.get("rope_type") == "llama3":
        factor, lo, hi = float(rope["factor"]), float(rope["low_freq_factor"]), float(rope["high_freq_factor"])
        old = float(rope["original_max_position_embeddings"])
        wavelen = 2 * math.pi / inv_freq
        scaled = torch.where(wavelen > old / lo, inv_freq / factor, inv_freq)
        smooth = (old / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = (wavelen >= old / hi) & (wavelen <= old / lo)
        inv_freq = torch.where(medium, smoothed, scaled)
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16).float()


class LlamaAttention(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        H, hd = cfg.hidden_size, cfg.head_dim
        self.nq, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, hd
        self.q_proj = nn.Linear(H, self.nq * hd, bias=False)
        self.k_proj = nn.Linear(H, self.nkv * hd, bias=False)
        self.v_proj = nn.Linear(H, self.nkv * hd, bias=False)
        self.o_proj = nn.Linear(self.nq * hd, H, bias=False)

    def forward(self, x, pos, cos_sin, attn, kv_cache):
        n, hd = x.shape[0], self.hd
        # contiguous: with --fuse-lm the projections are column slices of one GEMM output
        q = self.q_proj(x).contiguous()
        k = self.k_proj(x).contiguous()
        v = self.v_proj(x).view(n, self.nkv, hd)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(pos, q, k, hd, cos_sin, is_neox=True)
        attn.set_kv_cache(kv_cache, k.view(n, self.nkv, hd), v)
        return self.o_proj(attn.run(q.view(n, self.nq, hd), kv_cache).reshape(n, -1))


class DualFFNLayer(nn.Module):
    """A Llama layer with a second norm / FFN path for audio rows (both share the attention)."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = LlamaAttention(cfg)
        self.mlp = Qwen3TTSMLP(cfg)
        self.audio_mlp = Qwen3TTSMLP(cfg)
        self.input_layernorm = Qwen3TTSRMSNorm(H, eps=eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(H, eps=eps)
        self.audio_input_layernorm = Qwen3TTSRMSNorm(H, eps=eps)
        self.audio_post_attention_layernorm = Qwen3TTSRMSNorm(H, eps=eps)

    def forward(self, h, audio, pos, cos_sin, attn, kv_cache):
        """audio: ``[rows, 1]`` bool (rows taking the audio path), or None when every row does."""
        if audio is None:
            h = h + self.self_attn(self.audio_input_layernorm(h), pos, cos_sin, attn, kv_cache)
            return h + self.audio_mlp(self.audio_post_attention_layernorm(h))
        x = torch.where(audio, self.audio_input_layernorm(h), self.input_layernorm(h))
        h = h + self.self_attn(x, pos, cos_sin, attn, kv_cache)
        a = self.audio_mlp(self.audio_post_attention_layernorm(h))
        t = self.mlp(self.post_attention_layernorm(h))
        return h + torch.where(audio, a, t)


class HiggsV2LM(nn.Module):
    """Backbone, embeddings and audio head in checkpoint naming (``model.*`` -> ``layers`` / ``norm`` /
    ``embed_tokens`` / ``embed_audio_tokens``; ``audio_lm_head``)."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.layers = nn.ModuleList([DualFFNLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = Qwen3TTSRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_audio_tokens = nn.Embedding(N_CB * CB_VOCAB, cfg.hidden_size)
        self.audio_lm_head = nn.Linear(cfg.hidden_size, N_CB * CB_VOCAB, bias=False)
        self.cos_sin = None

    def forward(self, h, audio, pos, attn, kv_cache):
        for i, layer in enumerate(self.layers):
            h = layer(h, audio, pos, self.cos_sin, attn, kv_cache[i])
        return self.norm(h)

    @classmethod
    def from_checkpoint(cls, model_dir: str, device: torch.device, max_pos: int) -> "HiggsV2LM":
        from safetensors import safe_open

        c = json.load(open(os.path.join(model_dir, "config.json")))
        if c.get("num_codebooks", N_CB) != N_CB or c.get("codebook_size", CB_VOCAB) != CB_VOCAB:
            raise ValueError(f"expected {N_CB} codebooks of {CB_VOCAB} codes, got {c.get('num_codebooks')}")
        cfg = SimpleNamespace(**c)
        cfg.head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        rope = dict(c.get("rope_parameters") or c.get("rope_scaling") or {})
        rope.setdefault("rope_theta", c.get("rope_theta", 500000.0))
        with torch.device("meta"):
            lm = cls(cfg)
        rename = {"model.embed_audio_tokens.embed_audio_tokens.weight": "embed_audio_tokens.weight"}
        state = {}
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name == "text_lm_head.weight":
                    continue  # training only
                key = rename.get(name) or (name[len("model.") :] if name.startswith("model.") else name)
                state[key] = f.get_tensor(name).to(torch.bfloat16)
        lm.load_state_dict(state, strict=True, assign=True)
        lm.cos_sin = llama3_rope_cache(max_pos, cfg.head_dim, rope).to(device)
        lm.config = cfg
        return lm.eval().requires_grad_(False)


class HiggsAudioV2Program(HiggsTTSProgram):
    def load(self) -> None:
        from huggingface_hub import snapshot_download

        name = self.config.get("model", REPO)
        if "/" not in name:
            name = REPO
        self.model_name = name
        self.model_dir = snapshot_download(name)
        if not self.config.get("max_num_pages"):
            # 112 KB of KV per token: one context per batch entry, not the default two.
            self.max_num_pages = self.max_batch_size * -(-self.max_context // self.page_size) + 1
        self.model = SimpleNamespace(default_sampling_config=SimpleNamespace(max_tokens=self.max_context))

        self.lm = HiggsV2LM.from_checkpoint(self.model_dir, self.device, self.max_context + 1)
        cfg = self.lm.config
        self.n_codebooks = N_CB
        self.vocab_size = CB_VOCAB
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.text_vocab = cfg.vocab_size
        codec_dir = snapshot_download(self.config.get("codec_model", CODEC_REPO))
        self._init_codec(HiggsAudioTokenizer.from_checkpoint(codec_dir, prefix=""))

        gen = {}
        if os.path.exists(os.path.join(self.model_dir, "generation_config.json")):
            gen = json.load(open(os.path.join(self.model_dir, "generation_config.json")))
        self.gen_config = gen
        self.ras_win = int(self.config.get("ras_win_len", gen.get("ras_win_len") or 0))
        self.ras_max = int(self.config.get("ras_win_max_num_repeat", gen.get("ras_win_max_num_repeat") or 2))

        from tokenizers import Tokenizer

        self.tokenizer_json = os.path.join(self.model_dir, "tokenizer.json")
        vocab = Tokenizer.from_file(self.tokenizer_json).get_vocab(with_added_tokens=True)
        missing = [s for s in SPECIALS if s not in vocab]
        if missing:
            raise RuntimeError(f"Higgs Audio v2 tokenizer lacks {missing}")
        self.special_ids = {s.strip("<|>"): vocab[s] for s in SPECIALS}
        logger.info(
            "Higgs Audio v2: %d KV pages, first chunk %d frames (pre-seed %d), codec %s%s, RAS window %d",
            self.max_num_pages,
            self.first_frames,
            self.preseed,
            self.codec_dtype,
            " (fused)" if self.codec.fused_decode is not None else "",
            self.ras_win,
        )

    # ---- LM ---------------------------------------------------------------------------------------

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        self._audio_rows = arena["mask"][:rows, :1] == AUDIO  # read by backbone in the same graph
        text = self.lm.embed_tokens(ids[:, 0].clamp(0, self.text_vocab - 1))
        codes = F.embedding(ids.clamp(0, CB_VOCAB - 1) + self.cb_offsets, self.lm.embed_audio_tokens.weight).sum(1)
        return torch.where(self._audio_rows, codes, text)

    def fused_stack_spec(self):
        return None  # the fused decoder stack has no dual FFN path

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        # Decode rows are all generated audio rows.
        audio = None if getattr(attn, "kind", "prefill") == "decode" else self._audio_rows
        return self.lm(hidden, audio, pos, attn, self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.audio_lm_head(hidden).view(-1, N_CB, CB_VOCAB)

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        n = n_req
        logits = self.head(hidden)
        draw = self.draw[:n]
        self.sample_into(logits, arena, draw, n, self.LM_SALT)
        step = arena["step"][:n]
        slot = arena["slot"][:n].long()
        first = arena["first"][:n] != 0
        if self.ras_win > 0:
            hist = torch.where(first[:, None, None], BOC, self.ras_hist[slot])  # [n, ras_win, 8]
            rep = ((hist == draw[:, None]) & (hist < BOC)).sum(1)
            alt = self.alt_draw[:n]
            ops().sample(
                logits, N_CB, arena["active"][:n], self.ones[:n], self.zeros_i[:n], self.ones[:n],
                self.zeros_f[:n], arena["seed"][:n], step, self.LM_SALT + 1, alt,
            )  # fmt: skip
            draw = torch.where(rep >= self.ras_max, alt, draw)
        codes = torch.where(self.cb_cols[None] > step[:, None], BOC, draw)
        trig = torch.where(first, NO_EOC, self.trig_step[slot])
        forced = (trig != NO_EOC)[:, None] & (self.cb_cols[None] < (step - trig)[:, None])
        codes = torch.where(forced, EOC, codes)
        trig = torch.where((trig == NO_EOC) & (codes == EOC).any(-1), step, trig)
        self.trig_step.index_copy_(0, slot, trig)
        eoc = torch.where(first, NO_EOC, self.eoc_step[slot])
        eoc = torch.where((eoc == NO_EOC) & (codes[:, 0] == EOC), step, eoc)
        self.eoc_step.index_copy_(0, slot, eoc)
        if self.ras_win > 0:
            self.ras_hist.index_copy_(0, slot, torch.cat([hist[:, 1:], codes[:, None]], 1))
        out_tok.copy_(torch.where((step >= eoc + N_CB - 1)[:, None], -1, codes))

    # ---- state ------------------------------------------------------------------------------------

    def allocate_state(self) -> None:
        super().allocate_state()
        dev, n = self.device, self.max_batch_size
        self.trig_step = torch.full((self.max_slots + 1,), NO_EOC, dtype=torch.int32, device=dev)
        self.ras_hist = torch.full((self.max_slots + 1, max(1, self.ras_win), N_CB), BOC, dtype=torch.int32, device=dev)
        self.alt_draw = torch.zeros(n, N_CB, dtype=torch.int32, device=dev)
        self.ones = torch.ones(n, dtype=torch.float32, device=dev)
        self.zeros_f = torch.zeros(n, dtype=torch.float32, device=dev)
        self.zeros_i = torch.zeros(n, dtype=torch.int32, device=dev)

    def fuse_lm_modules(self):
        return [self.lm]

    # ---- prompt -----------------------------------------------------------------------------------

    def ref_rows(self, codes: torch.Tensor) -> torch.Tensor:
        """The reference clip framed by a BOS and an EOS frame, then delayed (``T + 9`` rows)."""
        bos = torch.full((1, N_CB), BOC, dtype=codes.dtype, device=codes.device)
        eos = torch.full((1, N_CB), EOC, dtype=codes.dtype, device=codes.device)
        return delay_pattern(torch.cat([bos, codes, eos]))

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """Text rows ``[id, 0, ..]`` of the chat-formatted prompt and audio rows of eight delayed codes: the
        system turn with the scene, [the ``ref_text`` / reference-audio turns], the user text, and
        ``<|audio_out_bos|>`` + an all-BOC row."""
        scene = model_kwargs.get("instruct")
        scene = scene.strip() if isinstance(scene, str) and scene.strip() else self.spec_meta["default_scene"]
        tokens: List[int] = []
        mask: List[int] = []

        def text_rows(s: str) -> None:
            for t in self._encode(s, add_special_tokens=False):
                tokens.extend([t] + [0] * (N_CB - 1))
                mask.extend([TEXT] + [0] * (N_CB - 1))

        def audio_rows(codes: List[int]) -> None:
            tokens.extend(codes)
            mask.extend(([AUDIO] + [0] * (N_CB - 1)) * (len(codes) // N_CB))

        system = f"{SYSTEM}\n\n<|scene_desc_start|>\n{scene}\n<|scene_desc_end|>"
        prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        if audio_path is not None:
            ref_text = model_kwargs.get("ref_text")
            if not (isinstance(ref_text, str) and ref_text.strip()):
                raise ValueError("voice cloning with Higgs Audio v2 needs ref_text, the reference audio's transcript")
            codes = self._ref_codes(audio_path)
            text_rows(
                prefix + f"<|start_header_id|>user<|end_header_id|>\n\n{ref_text.strip()}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"
            )
            audio_rows(codes)
            prefix = "<|audio_eos|><|eot_id|>"
        text_rows(
            prefix + f"<|start_header_id|>user<|end_header_id|>\n\n{normalize_text(text)}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"
        )
        audio_rows([BOC] * N_CB)
        return Prompt(tokens, mask)

    def meta(self) -> Dict[str, Any]:
        m = super().meta()
        g = self.gen_config
        m.update(
            kind="higgs_audio_v2",
            prompt_builder="python",
            default_scene=DEFAULT_SCENE,
            sample_rate=SAMPLE_RATE,
            sampling={
                "temperature": float(g.get("temperature", 1.0)),
                "top_k": int(g.get("top_k", 50)),
                "top_p": float(g.get("top_p", 0.95)),
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        )
        return m
