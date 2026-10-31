"""Common structure of a graph program: the static buffers and captured stages of one model.

A program owns the model weights and every persistent device buffer (KV cache, per-slot sampling
state), declares the input/output arena fields and captures one CUDA graph per stage and shape
bucket. The Rust runtime only ever sees the exported arenas, graph handles and ``meta``.

Conventions shared by every LM program (and relied on by the Rust runtime):

* rows: one row per token position processed this step. ``tok[r]`` holds the token ids (all
  codebooks) of row ``r`` unless ``fb[r] != 0``, in which case the ids are read on device from
  ``last_tok[row_slot[r]]`` (the request's previously sampled token), so the CPU never has to wait
  for the previous step's output before launching the next one.
* requests: ``N`` per-request entries per graph. ``last_row[i]`` selects the row whose hidden state
  feeds the LM head for request ``i``. Inactive / padding entries have ``active[i] = 0`` and use
  the dummy slot ``max_slots``.
* KV page 0 is a scratch page reserved for padding rows; the runtime never hands it to a request.
* ``out_kind[i]``: what request ``i``'s sampled row is (``TOKEN_AUDIO`` / ``TOKEN_STOP`` /
  ``TOKEN_OTHER``), computed in the graph by ``token_kind``. The Rust runtime reads it and never
  interprets token ids itself; its only per-model code builds prompts (``rust/src/models``).
"""

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils import get_logger
from .core import Arena, EngineSpecBuilder, RtAttention, add_attention_fields
from .kernels import ops
from .prompt import TOKEN_AUDIO, TOKEN_OTHER, TOKEN_STOP, FeaturePool, Prompt, encode  # noqa: F401

logger = get_logger(__name__)


def _pow2_up_to(n: int) -> List[int]:
    sizes = [1]
    while sizes[-1] * 2 <= n:
        sizes.append(sizes[-1] * 2)
    if sizes[-1] != n:
        sizes.append(n)
    return sizes


class Program:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        # Disaggregation: the detokenizer (audio decoder) lives on another GPU.
        dd = config.get("detok_device")
        self.detok_device = torch.device(f"cuda:{int(dd)}") if dd is not None else device

    def build(self, builder: EngineSpecBuilder) -> None:
        raise NotImplementedError


class LMProgram(Program):
    """Single-stage autoregressive LM followed by a chunked audio detokenizer.

    Subclasses provide the model (``load``), the forward pieces (``embed``, ``backbone``,
    ``head``), the detokenizer (``detokenize``) and model-specific ``meta``.
    """

    # Graph RNG salts: different stages of one request must not draw the same random numbers.
    LM_SALT = 0

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.dtype = torch.bfloat16
        self.max_batch_size = int(config.get("max_batch_size", 8))
        self.max_slots = int(config.get("max_slots", 4 * self.max_batch_size))
        self.page_size = int(config.get("page_size", 64))
        self.max_context = int(config.get("max_context", 2048))
        buckets = config.get("prefill_buckets") or [b for b in (256, 512, 1024, 2048) if b <= self.max_context]
        self.prefill_buckets = sorted({int(b) for b in buckets})
        pages_per_req = math.ceil(self.max_context / self.page_size)
        default_pages = 2 * self.max_batch_size * pages_per_req + 1
        self.max_num_pages = int(config.get("max_num_pages") or default_pages)
        self.decode_batch_sizes = _pow2_up_to(self.max_batch_size)
        self.max_detok_batch_size = int(config.get("max_detok_batch_size", self.max_batch_size))
        self.detok_batch_sizes = _pow2_up_to(self.max_detok_batch_size)
        if config.get("urgent_max_batch"):
            self.urgent_batch_sizes = tuple(_pow2_up_to(int(config["urgent_max_batch"])))
        # Explicit graph sizes (microbenchmarks); the largest must equal the corresponding maximum.
        if config.get("decode_batch_sizes"):
            self.decode_batch_sizes = sorted({int(b) for b in config["decode_batch_sizes"]})
            assert self.decode_batch_sizes[-1] == self.max_batch_size, "largest decode size != max_batch_size"
        if config.get("detok_batch_sizes"):
            self.detok_batch_sizes = sorted({int(b) for b in config["detok_batch_sizes"]})
            assert self.detok_batch_sizes[-1] == self.max_detok_batch_size, "largest detok size != max"
        self.float_workspace_mb = int(config.get("float_workspace_mb", 256))
        self.detok_slots = self.max_slots

        self.load()

    # ---- model hooks --------------------------------------------------------------------------

    def load(self) -> None:
        raise NotImplementedError

    n_codebooks: int = 1
    vocab_size: int
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    detok_interval: int
    detok_samples: int  # int16 samples produced per detokenized chunk (all channels)

    uses_input_mask: bool = False
    uses_repetition_penalty: bool = True
    # Leading token columns read from ``last_tok`` on fed-back rows (None = all); the remaining
    # columns always come from the host-written ``tok``.
    fb_columns = None
    # Detokenizer keeps per-request state across chunks (``dt_slot`` / ``dt_reset`` inputs).
    detok_stateful: bool = False
    # The LM step can be captured as two graphs, backbone + first head ("lm_decode") and the depth
    # transformer ("lm_depth"), so the runtime can schedule around the depth phase. Programs that
    # support it implement ``decode_head`` / ``decode_depth`` and set this flag.
    supports_depth_split: bool = False
    # The backbone can additionally run as layer groups (``backbone_range``), one graph each.
    supports_layer_split: bool = False
    fstack = None  # FusedDecoderStack with --fuse-layers
    # Detok can run on a second, urgent lane (own arenas; ``detok_step(..., lane=1)`` must not share
    # scratch state with lane 0). Safe for stateless detokenizers whose intermediates live in the
    # graph pool, and for programs that keep a separate work cache per lane.
    supports_urgent_lane: bool = False
    urgent_lane: bool = False  # set by build() when the spec has urgent arenas
    # Batch sizes of the urgent lane's detok graphs.
    urgent_batch_sizes = (1, 2, 4, 8)
    # Rows of the device feature pool (per-request prompt features such as encoded audio), 0 = none.
    # Rows are allocated by the Rust runtime and filled by hooks (``_stage_features`` +
    # ``hook_features_commit``); programs read them from ``self.feature_pool``.
    feature_pool_rows: int = 0
    feature_dim: int = 0

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        """ids: [rows, C] int64 -> hidden [rows, H]."""
        raise NotImplementedError

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        raise NotImplementedError

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [N, H] -> logits [N, C, V]."""
        raise NotImplementedError

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        """hidden [N, H] (last row of every request) -> sampled tokens written to out_tok [N, C]."""
        logits = self.head(hidden)  # [N, C, V]
        self.sample_into(logits, arena, out_tok, n_req, self.LM_SALT, penalty=self.uses_repetition_penalty)

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [Bd, interval, C] int64 -> float audio in [-1, 1] with Bd * detok_samples elements."""
        raise NotImplementedError

    def stop_tokens(self) -> List[int]:
        """Codebook-0 tokens that end generation."""
        return []

    def audio_tokens(self) -> Optional[Tuple[int, int]]:
        """``[lo, hi)`` of the codebook-0 tokens that are audio; the others are fed back but not
        detokenized (interleaved text). ``None``: every token that does not stop generation is audio."""
        return None

    _stop_ids: Optional[torch.Tensor] = None  # stop_tokens() on device, made before capture

    def token_kind(self, tok: torch.Tensor) -> torch.Tensor:
        """tok [N, C] int32 (the sampled rows) -> [N] int32 ``TOKEN_*`` codes. Runs inside the captured
        LM graph; the default classifies codebook 0 by ``stop_tokens`` and ``audio_tokens``."""
        t = tok[:, 0]
        kind = torch.full_like(t, TOKEN_AUDIO)
        span = self.audio_tokens()
        if span is not None:
            kind = torch.where((t >= span[0]) & (t < span[1]), kind, TOKEN_OTHER)
        if self._stop_ids is not None:
            kind = torch.where((t[:, None] == self._stop_ids).any(-1), TOKEN_STOP, kind)
        return kind

    def meta(self) -> Dict[str, Any]:
        """Model constants for the Rust runtime: the prompt builder's inputs (``kind`` names the
        builder, or ``prompt_builder``), plus ``max_tokens``, ``sampling``, ``sample_rate`` and, where
        they apply, ``decode_mask``, ``decode_row``, ``preseed_frames`` and ``text_stream``."""
        return {}

    # ---- prompt ----------------------------------------------------------------------------------

    spec_meta: Dict[str, Any] = {}  # the engine's meta (common + ``meta()``), set by ``build``

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """The prompt rows of a request. The Python runtime always builds prompts here, the Rust runtime
        when ``meta["prompt_builder"]`` is ``python``; a model's Rust prompt builder must produce the same
        rows. Default: the ``template`` builder (``meta["prompt_template"]``)."""
        if self.spec_meta.get("prompt_builder") != "template":
            raise NotImplementedError(f"{type(self).__name__} has no Python prompt builder")
        t = self.spec_meta["prompt_template"]
        if audio_path is not None:
            raise ValueError("this model does not accept audio input")
        s = t.get("text", "{text}")
        for name, k in (t.get("kwargs") or {}).items():
            v = model_kwargs.get(name)
            if isinstance(v, str):
                value = v
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                value = str(v)
            else:
                value = k.get("default", "")
            if k.get("choices") and value not in k["choices"]:
                raise ValueError(f"{name} {value} is not available for this model")
            s = s.replace("{" + name + "}", value)
        # The request text goes in last, so braces in it are never substituted.
        ids = encode(t["tokenizer_json"], s.replace("{text}", text, 1), t.get("add_special_tokens", True))
        return Prompt(list(t.get("prefix_ids", [])) + ids + list(t.get("suffix_ids", [])))

    def tokenize_stream_text(self, text: str) -> List[int]:
        """Tokens of a streamed text chunk (input streaming; no special tokens)."""
        raise ValueError("input streaming is not supported by this model")

    def preprocess_streaming(
        self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any], first: int
    ) -> Prompt:
        """Prompt of an input-streaming request whose only text token so far is ``first``."""
        raise ValueError("input streaming is not supported by this model")

    def hook_preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """``preprocess`` for the Rust runtime's ``python`` prompt builder."""
        p = self.preprocess(text, audio_path, model_kwargs)
        if p.features is not None or p.hold is not None:
            raise ValueError(f"{type(self).__name__} prompts hold resources: use its Rust prompt builder")
        return {"tokens": p.tokens, "mask": p.mask}

    _feature_pool_alloc: Optional[FeaturePool] = None

    def _features(self) -> FeaturePool:
        """Allocator of ``feature_pool`` rows for prompts built in Python."""
        if self._feature_pool_alloc is None:
            self._feature_pool_alloc = FeaturePool(self.feature_pool_rows)
        return self._feature_pool_alloc

    def _encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return encode(self.spec_meta["tokenizer_json"], text, add_special_tokens)

    # ---- feature pool hooks -------------------------------------------------------------------

    def _stage_features(self, feats: torch.Tensor) -> str:
        """Keeps [n, feature_dim] features on device until the runtime assigns them pool rows."""
        key = uuid.uuid4().hex
        self._staged[key] = feats.to(self.dtype)
        return key

    def hook_features_commit(self, key: str, offset: int) -> None:
        feats = self._staged.pop(key)
        if offset < 0 or offset + feats.shape[0] > self.feature_pool_rows:
            raise ValueError("feature pool range out of bounds")
        self.feature_pool[offset : offset + feats.shape[0]].copy_(feats)
        # The engine stream reads the pool without synchronizing with this (hook) stream.
        torch.cuda.current_stream().synchronize()

    def hook_features_discard(self, key: str) -> None:
        self._staged.pop(key, None)

    # ---- build --------------------------------------------------------------------------------

    def declare(self, builder: EngineSpecBuilder) -> None:
        inp, out = builder.inputs, builder.outputs
        C = self.n_codebooks
        max_rows = max(self.prefill_buckets + [self.max_batch_size])
        n = self.max_batch_size

        inp.add("tok", (max_rows, C), torch.int32)
        inp.add("fb", (max_rows,), torch.int32)
        inp.add("row_slot", (max_rows,), torch.int32)
        inp.add("pos", (max_rows,), torch.int32)
        if self.uses_input_mask:
            inp.add("mask", (max_rows, C), torch.int32)
        add_attention_fields(inp, "lm_", n, max_rows, self.max_num_pages)

        inp.add("last_row", (n,), torch.int32)
        inp.add("slot", (n,), torch.int32)
        inp.add("active", (n,), torch.int32)
        inp.add("first", (n,), torch.int32)
        inp.add("temp", (n,), torch.float32)
        inp.add("top_k", (n,), torch.int32)
        inp.add("top_p", (n,), torch.float32)
        inp.add("min_p", (n,), torch.float32)
        inp.add("rep_pen", (n,), torch.float32)
        inp.add("seed", (n,), torch.int64)
        inp.add("step", (n,), torch.int32)
        lanes = [(builder.dt_inputs, builder.dt_outputs, self.max_detok_batch_size)]
        if builder.urgent_lane:
            lanes.append((builder.du_inputs, builder.du_outputs, max(self._urgent_sizes())))
        for dinp, dout, nb in lanes:
            dinp.add("dt_tok", (nb, self.detok_interval, C), torch.int32)
            if self.detok_stateful:
                dinp.add("dt_slot", (nb,), torch.int32)
                dinp.add("dt_reset", (nb,), torch.int32)
            dout.add("audio", (nb, self.detok_samples), torch.int16)

        out.add("out_tok", (n, C), torch.int32)
        out.add("out_kind", (n,), torch.int32)

    def allocate_state(self) -> None:
        C = self.n_codebooks
        self.kv_cache = torch.zeros(
            self.num_layers,
            self.max_num_pages,
            2,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.float_workspace = torch.empty(self.float_workspace_mb << 20, dtype=torch.uint8, device=self.device)
        # Last sampled token of every slot (+ dummy slot), read by rows with fb != 0.
        self.last_tok = torch.zeros(self.max_slots + 1, C, dtype=torch.int32, device=self.device)
        words = (self.vocab_size + 31) // 32 if self.uses_repetition_penalty else 1
        self.rep_bits = torch.zeros(self.max_slots + 1, C, words, dtype=torch.int32, device=self.device)
        self.feature_pool_rows = int(self.config.get("feature_pool_rows", self.feature_pool_rows))
        self.feature_pool = torch.zeros(
            max(1, self.feature_pool_rows), max(1, self.feature_dim), dtype=self.dtype, device=self.device
        )
        if self.supports_depth_split:
            self.seg_hidden = torch.zeros(self.max_batch_size, self.hidden_size, dtype=self.dtype, device=self.device)
        if self.supports_layer_split:
            self.seg_rows = torch.zeros(self.max_batch_size, self.hidden_size, dtype=self.dtype, device=self.device)
        self._staged: Dict[str, torch.Tensor] = {}

    def decode_head(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        raise NotImplementedError

    def decode_depth(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        raise NotImplementedError

    def lm_step_head(self, arena: Arena, out: Arena, rows: int, n_req: int, attn: RtAttention) -> None:
        """First segment of a split LM step: backbone + first head, hidden kept for the depth graph."""
        hidden = self._lm_hidden(arena, rows, n_req, attn)
        self.seg_hidden[:n_req].copy_(hidden)
        self.decode_head(hidden, arena, out["out_tok"][:n_req], n_req)

    def backbone_range(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention, l0: int, l1: int, last: bool):
        """Layers [l0, l1) of the backbone (plus the final norm when ``last``)."""
        raise NotImplementedError

    def lm_step_layers(
        self, arena: Arena, out: Arena, rows: int, n_req: int, attn: RtAttention, l0: int, l1: int
    ) -> None:
        """One layer group of a layer-split LM step. The first group embeds the inputs; the last one
        also runs the first head (codebook 0). Rows travel between groups in ``seg_rows``."""
        last = l1 >= self.num_layers
        if l0 == 0:
            tok = arena["tok"][:rows]
            fb = arena["fb"][:rows]
            row_slot = arena["row_slot"][:rows].long()
            fed = fb[:, None] != 0
            if self.fb_columns is not None:
                fed = fed & (torch.arange(self.n_codebooks, device=fb.device) < self.fb_columns)
            ids = torch.where(fed, self.last_tok[row_slot], tok).long()
            hidden = self.embed(ids, arena, rows)
        else:
            hidden = self.seg_rows[:rows]
        hidden = self.backbone_range(hidden, arena["pos"][:rows], attn, l0, l1, last)
        if not last:
            self.seg_rows[:rows].copy_(hidden)
            return
        hidden = hidden.index_select(0, arena["last_row"][:n_req].long())
        self.seg_hidden[:n_req].copy_(hidden)
        self.decode_head(hidden, arena, out["out_tok"][:n_req], n_req)

    def lm_step_depth(self, arena: Arena, out: Arena, n_req: int) -> None:
        """Second segment of a split LM step: the depth transformer and the token feedback."""
        out_tok = out["out_tok"][:n_req]
        self.decode_depth(self.seg_hidden[:n_req], arena, out_tok, n_req)
        self.last_tok.index_copy_(0, arena["slot"][:n_req].long(), out_tok)
        out["out_kind"][:n_req].copy_(self.token_kind(out_tok))

    def _lm_hidden(self, arena: Arena, rows: int, n_req: int, attn: RtAttention) -> torch.Tensor:
        tok = arena["tok"][:rows]
        fb = arena["fb"][:rows]
        row_slot = arena["row_slot"][:rows].long()
        fed = fb[:, None] != 0
        if self.fb_columns is not None:
            fed = fed & (torch.arange(self.n_codebooks, device=fb.device) < self.fb_columns)
        ids = torch.where(fed, self.last_tok[row_slot], tok).long()
        hidden = self.backbone(self.embed(ids, arena, rows), arena["pos"][:rows], attn)
        return hidden.index_select(0, arena["last_row"][:n_req].long())

    def lm_step(self, arena: Arena, out: Arena, rows: int, n_req: int, attn: RtAttention) -> None:
        hidden = self._lm_hidden(arena, rows, n_req, attn)
        out_tok = out["out_tok"][:n_req]
        self.decode_tokens(hidden, arena, out_tok, n_req)
        self.last_tok.index_copy_(0, arena["slot"][:n_req].long(), out_tok)
        out["out_kind"][:n_req].copy_(self.token_kind(out_tok))

    @torch.compiler.disable
    def sample_into(
        self,
        logits: torch.Tensor,
        arena: Arena,
        out: torch.Tensor,
        n_req: int,
        salt: int,
        penalty: bool = False,
    ) -> None:
        """Samples logits [N, R, V] into out [N, R] (int32, row stride may exceed R)."""
        k = ops()
        R = logits.shape[1]
        slot = arena["slot"][:n_req]
        active = arena["active"][:n_req]
        first = arena["first"][:n_req]
        if penalty:
            k.rep_penalty_reset(self.rep_bits, R, slot, active, first)
            k.rep_penalty_apply(logits, R, self.rep_bits, slot, active, first, arena["rep_pen"][:n_req])
        k.sample(
            logits,
            R,
            active,
            arena["temp"][:n_req],
            arena["top_k"][:n_req],
            arena["top_p"][:n_req],
            arena["min_p"][:n_req],
            arena["seed"][:n_req],
            arena["step"][:n_req],
            salt,
            out,
        )
        if penalty:
            k.rep_penalty_update(self.rep_bits, R, self.vocab_size, slot, active, out)

    def _urgent_sizes(self):
        return [b for b in self.urgent_batch_sizes if b <= self.max_detok_batch_size] or [1]

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        audio = self.detokenize(arena["dt_tok"][:bd].long()).reshape(bd, self.detok_samples)
        # Truncation toward zero, matching numpy's (x * 32767).astype(np.int16).
        out["audio"][:bd].copy_((audio.float() * 32767).to(torch.int16))

    def fuse_lm_modules(self) -> List[torch.nn.Module]:
        """LM modules whose sibling q/k/v and gate/up projections ``--fuse-lm`` merges into single
        GEMMs (``vox_serve.fusion.projections``). Programs with Llama / Qwen-style layers list them here."""
        return []

    def fused_stack_spec(self) -> Optional[Dict[str, Any]]:
        """Arguments of ``vox_serve.fusion.decoder_stack.FusedDecoderStack`` for ``--fuse-layers`` (layers, norm,
        head counts, RoPE table), or None if the program's backbone has no fused path."""
        return None

    def _fuse_lm(self) -> None:
        from ..fusion.projections import merge_projections

        roots = self.fuse_lm_modules()
        n = sum(merge_projections(r) for r in roots)
        logger.info("fuse_lm: merged %d projection groups in %d module(s)", n, len(roots))

    def build(self, builder: EngineSpecBuilder) -> None:
        if self.config.get("max_tokens"):
            # Per-request token limit (prompt included) instead of the model's default.
            self.model.default_sampling_config.max_tokens = int(self.config["max_tokens"])
        if self.config.get("fuse_lm"):
            if self.config.get("torch_compile"):
                # Compiled frames compute their own slice and skip the shared state, so a mix of
                # compiled and eager siblings could read a stale slice: keep the two exclusive.
                logger.warning("--fuse-lm is ignored with --enable-torch-compile")
            else:
                self._fuse_lm()
        if self.config.get("fuse_layers"):
            spec = self.fused_stack_spec()
            if spec is None:
                logger.warning("%s has no fused decoder stack; ignoring fuse_layers", type(self).__name__)
            else:
                from ..fusion.decoder_stack import FusedDecoderStack

                self.fstack = FusedDecoderStack(**spec)
                logger.info("fuse_layers: fused decoder stack for %d layers", len(spec["layers"]))
        self.declare(builder)
        builder.inputs.finalize()
        builder.outputs.finalize()
        if builder.disaggregated:
            builder.dt_inputs.finalize()
            builder.dt_outputs.finalize()
        if builder.urgent_lane:
            builder.du_inputs.finalize()
            builder.du_outputs.finalize()
        self.urgent_lane = builder.urgent_lane
        self.allocate_state()
        stops = self.stop_tokens()
        self._stop_ids = torch.tensor(stops, dtype=torch.int32, device=self.device) if stops else None
        inp, out = builder.inputs, builder.outputs
        dinp, dout = builder.dt_inputs, builder.dt_outputs
        self._init_inputs_for_capture(inp, dinp, builder)
        self._size_float_workspace(inp)
        if self.config.get("torch_compile"):
            self._compile_lm()

        n = self.max_batch_size
        seg_cfg = str(self.config.get("lm_segments") or "whole")
        group = int(seg_cfg.split(":")[1]) if seg_cfg.startswith("layers:") else 0
        split = seg_cfg == "depth" or group > 0
        if split and not self.supports_depth_split:
            logger.warning("%s has no depth split; capturing whole LM steps", type(self).__name__)
            split, group = False, 0
        if group and not self.supports_layer_split:
            logger.warning("%s has no layer split; splitting at the depth transformer only", type(self).__name__)
            group = 0
        self.lm_split = split
        self.lm_groups = (
            [(l0, min(l0 + group, self.num_layers)) for l0 in range(0, self.num_layers, group)] if group else []
        )
        # Every decode / detok graph exists once per SM mode (key [size, mode]); prefill graphs only
        # for the whole GPU (mode 0), since prefill is compute-bound and runs exclusively. Layer groups
        # after the first are stage "lm_seg" with key [size, mode, group]; they share the first
        # group's attention plan (same wrapper), so only "lm_decode" carries the attention spec.
        for m, mode in enumerate(builder.modes):
            # Warm-up blocks are cached per stream and every mode warms up on its own stream: release
            # them so each mode's warm-ups reuse the same memory.
            torch.cuda.empty_cache()
            for bs in sorted(self.decode_batch_sizes, reverse=True):
                attn = self._attention("decode", inp, bs, bs)
                self._prepare_capture_rows(inp, bs, bs)
                st = mode["lm_stream"]
                if self.lm_groups:
                    for gi, (l0, l1) in enumerate(self.lm_groups):
                        fn = lambda bs=bs, a=attn, l0=l0, l1=l1: self.lm_step_layers(inp, out, bs, bs, a, l0, l1)  # noqa: E731
                        if gi == 0:
                            builder.capture("lm_decode", [bs, m], fn, attn, stream=st)
                        else:
                            builder.capture("lm_seg", [bs, m, gi], fn, stream=st)
                    builder.capture("lm_depth", [bs, m], lambda bs=bs: self.lm_step_depth(inp, out, bs), stream=st)
                elif split:
                    builder.capture(
                        "lm_decode",
                        [bs, m],
                        lambda bs=bs, a=attn: self.lm_step_head(inp, out, bs, bs, a),
                        attn,
                        stream=st,
                    )
                    builder.capture("lm_depth", [bs, m], lambda bs=bs: self.lm_step_depth(inp, out, bs), stream=st)
                else:
                    builder.capture(
                        "lm_decode", [bs, m], lambda bs=bs, a=attn: self.lm_step(inp, out, bs, bs, a), attn, stream=st
                    )
        for t in sorted(self.prefill_buckets, reverse=True):
            attn = self._attention("prefill", inp, n, t)
            self._prepare_capture_rows(inp, t, n)
            builder.capture("lm_prefill", [t, 0], lambda t=t, a=attn: self.lm_step(inp, out, t, n, a), attn)
        if self.config.get("compile_detok"):
            self._compile_detok()
        for m, mode in enumerate(builder.modes):
            # Warm-up blocks are cached per stream and every mode warms up on its own stream: release
            # them so each mode's warm-ups reuse the same memory.
            torch.cuda.empty_cache()
            for bd in sorted(self.detok_batch_sizes, reverse=True):
                builder.capture(
                    "detok",
                    [bd, m],
                    lambda bd=bd: self.detok_step(dinp, dout, bd),
                    family="detok",
                    device=self.detok_device,
                    stream=mode["dec_stream"],
                )
        if builder.urgent_lane:
            uin, uout = builder.du_inputs, builder.du_outputs
            for bd in sorted(self._urgent_sizes(), reverse=True):
                builder.capture(
                    "detok_urgent",
                    [bd, 0],
                    lambda bd=bd: self.detok_step(uin, uout, bd, lane=1),
                    family="detok_urgent",
                    device=self.detok_device,
                )
        self._init_inputs_for_capture(inp, dinp, builder)

        builder.meta.update(self._common_meta())
        builder.meta.update(self.meta())
        # Server-wide sampling overrides (--temperature, ...) replace the model's defaults.
        sampling = builder.meta.setdefault("sampling", {})
        for key in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty"):
            if self.config.get(key) is not None:
                sampling[key] = self.config[key]
        if self.config.get("repetition_penalty") is not None and not self.uses_repetition_penalty:
            logger.warning("%s does not apply a repetition penalty; ignoring it", type(self).__name__)
        self.spec_meta = builder.meta

    def _compile_lm(self) -> None:
        """Wraps the LM forward pieces in torch.compile. Each captured shape compiles during its
        warm-up runs (static shapes), and the compiled kernels are then recorded into the graph like
        any others; calls Dynamo cannot trace (FlashInfer, the sampler ops) become graph breaks."""
        import torch._dynamo

        n_shapes = len(self.decode_batch_sizes) + len(self.prefill_buckets)
        for key in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(torch._dynamo.config, key):
                setattr(torch._dynamo.config, key, max(getattr(torch._dynamo.config, key), 8 * n_shapes))
        names = ["embed", "backbone", "decode_tokens"]
        if self.supports_depth_split:
            names += ["decode_head", "decode_depth"]
        if self.supports_layer_split:
            names += ["backbone_range"]
        targets = self.config.get("torch_compile_targets")
        if targets:
            names = [n for n in names if n in targets]
        for name in names:
            setattr(self, name, torch.compile(getattr(self, name), dynamic=False))
        logger.info("torch.compile enabled for %s", ", ".join(names))

    def compile_detok_targets(self):
        """(object, attribute) pairs of detokenizer functions to wrap in torch.compile. Default: the
        flow-matching estimator if the decoder has one (the bulk of its kernels; the vocoder around it
        keeps NumPy-backed state that Dynamo cannot guard), otherwise the whole ``detokenize``."""
        dec = getattr(getattr(self, "model", None), "audio_decoder", None)
        if isinstance(dec, torch.nn.Module):
            for name, mod in dec.named_modules():
                if name.endswith("estimator"):
                    return [(mod, a) for a in ("forward", "forward_chunk") if hasattr(mod, a)]
        return [(self, "detokenize")]

    def _compile_detok(self) -> None:
        """Fuses the detokenizer's elementwise / copy work with torch.compile before its graphs are
        captured (each batch size compiles during its warm-up runs)."""
        import torch._dynamo

        for key in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(torch._dynamo.config, key):
                setattr(torch._dynamo.config, key, max(getattr(torch._dynamo.config, key), 64))
        for obj, attr in self.compile_detok_targets():
            setattr(obj, attr, torch.compile(getattr(obj, attr), dynamic=False))
            logger.info("torch.compile enabled for detokenizer %s.%s", type(obj).__name__, attr)

    def _size_float_workspace(self, inp: Arena) -> None:
        """Grows the shared FlashInfer float workspace until the largest prefill plan fits (split-kv
        scratch scales with batch entries x rows x heads). Runs before any graph is captured, since
        graphs bake in the workspace address."""
        while True:
            try:
                self._attention("prefill", inp, self.max_batch_size, max(self.prefill_buckets))
                return
            except RuntimeError as e:
                size = self.float_workspace.numel()
                if "workspace" not in str(e).lower() or size >= 8 << 30:
                    raise
            self.float_workspace = None
            torch.cuda.empty_cache()
            self.float_workspace = torch.empty(2 * size, dtype=torch.uint8, device=self.device)
            logger.info("FlashInfer float workspace grown to %d MiB", (2 * size) >> 20)

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
        )

    def _init_inputs_for_capture(self, inp: Arena, dinp: Arena, builder: Optional[EngineSpecBuilder] = None) -> None:
        """Safe inputs for warm-up runs: everything points at the dummy slot and scratch page."""
        inp.storage.zero_()
        dinp.storage.zero_()
        if builder is not None and builder.urgent_lane:
            builder.du_inputs.storage.zero_()
            if self.detok_stateful:
                builder.du_inputs["dt_slot"].fill_(self.detok_slots)
                builder.du_inputs["dt_reset"].fill_(1)
        inp["row_slot"].fill_(self.max_slots)
        inp["slot"].fill_(self.max_slots)
        inp["temp"].fill_(1.0)
        inp["top_p"].fill_(1.0)
        inp["rep_pen"].fill_(1.0)
        if self.detok_stateful:
            dinp["dt_slot"].fill_(self.detok_slots)
            dinp["dt_reset"].fill_(1)

    def _prepare_capture_rows(self, inp: Arena, rows: int, n_req: int) -> None:
        inp["lm_kv_page"].zero_()
        inp["lm_kv_off"][:rows].copy_(torch.arange(rows, device=self.device) % self.page_size)
        inp["last_row"][:n_req].copy_(torch.arange(n_req, device=self.device) * rows // n_req)
        torch.cuda.synchronize(self.device)

    def _common_meta(self) -> Dict[str, Any]:
        return {
            "engine_index": int(self.config.get("engine_index", 0)),
            "device": self.device.index or 0,
            "n_codebooks": self.n_codebooks,
            "vocab_size": self.vocab_size,
            "max_batch_size": self.max_batch_size,
            "max_slots": self.max_slots,
            "page_size": self.page_size,
            "max_num_pages": self.max_num_pages,
            "max_context": self.max_context,
            "prefill_buckets": self.prefill_buckets,
            "decode_batch_sizes": self.decode_batch_sizes,
            "detok_batch_sizes": self.detok_batch_sizes,
            "detok_interval": self.detok_interval,
            "detok_samples": self.detok_samples,
            "scratch_page": 0,
            "uses_input_mask": self.uses_input_mask,
            "detok_stateful": self.detok_stateful,
            "detok_slots": self.detok_slots,
            "feature_pool_rows": self.feature_pool_rows,
            "lm_split": bool(getattr(self, "lm_split", False)),
            "lm_groups": [list(g) for g in getattr(self, "lm_groups", [])],
        }
