"""
CUDA Graph wrapper for the acoustic flow-matching transformer in Voxtral-TTS.

Ported from vllm-omni's ``CUDAGraphAcousticTransformerWrapper``. Captures the
acoustic transformer forward pass (semantic logit + n-step Euler ODE with CFG)
into CUDA graphs for fixed batch sizes, eliminating kernel launch overhead on
every decode step.

Deviations from the vllm-omni original (per the Voxtral-TTS interface contract):
- ``capture_sizes`` is a constructor argument (gap fix #1), not a hardcoded
  ``DEFAULT_CAPTURE_SIZES`` class attribute.
- ``graph_pool`` is an injected constructor argument (gap fix), replacing
  ``vllm.platforms.current_platform.get_global_graph_pool()``.
- ``static_noise`` is filled via ``static_noise.normal_(generator=...)`` with a
  seeded ``torch.Generator`` before each replay (gap fix #8), so RNG state is
  not baked into the captured graph.
- ``vllm.logger`` is replaced with ``vox_serve.utils.get_logger``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch.cuda import CUDAGraph

from vox_serve.utils import get_logger

if TYPE_CHECKING:
    from vox_serve.tokenizer.voxtral_tts import FlowMatchingAudioTransformer

logger = get_logger(__name__)


class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads.

    Mirrors vllm-omni's ``AudioSpecialTokens``. Output audio tokens from the
    quantizer are offset by ``len(AudioSpecialTokens)`` to avoid conflicts with
    text tokens.
    """

    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [token for token in AudioSpecialTokens]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


class AcousticHeadCudaGraph:
    """
    CUDA Graph wrapper for the acoustic flow-matching transformer.

    Replaces the eager sampler path (which has Python-level branching and
    dynamic tensor allocation) with a CUDA-graph-compatible path using
    ``torch.argmax`` (equivalent to ``top_k=1`` greedy sampling) and
    pre-allocated static buffers.

    Args:
        flow_transformer: the acoustic flow-matching transformer
            (``FlowMatchingAudioTransformer``). Provides ``model_args``,
            ``acoustic_transformer_args``, ``acoustic_embeddings_levels``,
            ``semantic_codebook_output``, ``time_embedding`` and
            ``_predict_velocity``.
        graph_pool: the CUDA graph memory pool handle to capture into.
        capture_sizes: batch sizes to capture graphs for (gap fix #1).
        device: capture device.
        dtype: capture dtype.
        hidden_dim: backbone hidden size fed into the acoustic transformer.
        seed: seed for the ``torch.Generator`` driving ``static_noise`` refills.
    """

    def __init__(
        self,
        flow_transformer: "FlowMatchingAudioTransformer",
        graph_pool,
        capture_sizes: list[int],
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
        seed: int = 42,
    ):
        self.acoustic_transformer = flow_transformer
        self.graph_pool = graph_pool

        self.capture_sizes = sorted(capture_sizes)
        self.device = torch.device(device)
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.seed = seed

        # Pre-compute constants from the acoustic transformer
        self.empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
        self.end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self.semantic_mask_start = (
            len(AudioSpecialTokens) + self.acoustic_transformer.model_args.semantic_codebook_size
        )
        self.n_acoustic_codebook = self.acoustic_transformer.model_args.n_acoustic_codebook
        self.acoustic_embeddings_levels = self.acoustic_transformer.acoustic_embeddings_levels

        self.n_steps = self.acoustic_transformer.acoustic_transformer_args.n_decoding_steps

        # Seeded RNG for refilling the noise buffer before each replay (gap fix #8).
        # Kept off the captured graph so random state is not baked in.
        self._noise_generator = torch.Generator(device=self.device)
        self._noise_generator.manual_seed(self.seed)

        # Graph storage
        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_noise: dict[int, torch.Tensor] = {}
        self.static_cfg_alpha: dict[int, torch.Tensor] = {}
        self.static_fake_eos: dict[int, torch.Tensor] = {}
        self.static_audio_codes: dict[int, torch.Tensor] = {}

        self.enabled = False
        self._warmed_up = False

        # Capture eagerly at construction time.
        self._warmup_and_capture(self.device, self.dtype, self.hidden_dim)

    def _warmup_and_capture(self, device: torch.device, dtype: torch.dtype, hidden_dim: int):
        """Perform eager warmup and CUDA graph capture for all bucket sizes."""
        if self._warmed_up:
            logger.warning("AcousticHeadCudaGraph already warmed up, skipping")
            return

        logger.info(
            "AcousticHeadCudaGraph: starting warmup and capture for sizes %s",
            self.capture_sizes,
        )

        # Pre-create persistent buffers
        self.timesteps = torch.linspace(0, 1, self.n_steps + 1, device=device, dtype=dtype)
        self.fake_eos_one = torch.tensor(1.0, dtype=dtype, device=device)
        self.fake_eos_zero = torch.tensor(0.0, dtype=dtype, device=device)

        # Phase 1: Eager warmup for ALL capture sizes
        for size in self.capture_sizes:
            dummy = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
            dummy_cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)
            dummy_noise = torch.randn(size, self.n_acoustic_codebook, device=device, dtype=dtype)
            with torch.no_grad():
                self._forward_cudagraph_compatible(dummy, cfg_alpha=dummy_cfg_alpha, noise=dummy_noise)

        torch.accelerator.synchronize(device)

        # Phase 2: Capture graphs
        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype, hidden_dim)
                logger.info("  Captured CUDA Graph for batch_size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDA Graph for batch_size=%d",
                    size,
                    exc_info=True,
                )

        self.enabled = True
        self._warmed_up = True
        logger.info(
            "AcousticHeadCudaGraph warmup complete. Captured %d/%d graphs.",
            len(self.graphs),
            len(self.capture_sizes),
        )

    def _forward_cudagraph_compatible(
        self,
        hidden_states: torch.Tensor,
        cfg_alpha: torch.Tensor,
        noise: torch.Tensor,
    ):
        """
        The actual computation captured by the CUDA graph.

        This replaces the full ``compute_mm_logits -> acoustic_transformer.forward()``
        path with a graph-compatible version:
        - Uses argmax instead of an eager Sampler (equivalent for top_k=1)
        - Uses pre-created timesteps buffer instead of torch.linspace
        - Uses pre-created scalar tensors for torch.where
        - Calls ``_predict_velocity`` directly
        - Uses a pre-allocated noise buffer to avoid baking random state
          into the CUDA graph
        - Uses a pre-allocated cfg_alpha buffer for per-request CFG strength
        """
        at = self.acoustic_transformer
        B = hidden_states.shape[0]

        # --- Semantic logits via linear projection ---
        semantic_logit = at.semantic_codebook_output(hidden_states).float()
        semantic_logit[:, self.empty_audio_token_id] = -float("inf")
        semantic_logit[:, self.semantic_mask_start :] = -float("inf")

        # argmax == top_k=1 greedy sampling
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)  # (B, 1)

        # --- Flow matching: Euler ODE ---
        should_decode = semantic_code.squeeze(1) != self.end_audio_token_id

        x = noise

        # Pre-compute zero hidden states for unconditional CFG branch
        hidden_states_zero = torch.zeros_like(hidden_states)

        timesteps = self.timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            # Batch conditional + unconditional velocity in a single forward pass
            t_emb = at.time_embedding(t.view(-1, 1).repeat(B, 1)).to(hidden_states.dtype)
            x_batched = torch.cat([x, x], dim=0)  # (2B, C)
            llm_batched = torch.cat([hidden_states, hidden_states_zero], dim=0)  # (2B, D)
            t_emb_batched = t_emb.repeat(2, 1)  # (2B, D)

            v_all = at._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]

            # CFG combination (cfg_alpha is (B, 1), v_t is (B, C))
            v_t = cfg_alpha * v_t + (1 - cfg_alpha) * uncond_v_t

            x = x + v_t * dt

        # --- Quantize ---
        sampled = torch.clamp(x, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self.empty_audio_token_id
        acoustic_codes = output_codes + len(AudioSpecialTokens)

        # --- Combine semantic + acoustic ---
        audio_codes = torch.cat([semantic_code, acoustic_codes], dim=1)  # (B, 1 + n_acoustic)

        # --- Compute fake_eos ---
        fake_eos = torch.where(
            audio_codes[:, 0] == self.end_audio_token_id,
            self.fake_eos_one,
            self.fake_eos_zero,
        )

        return fake_eos, audio_codes

    def _capture_graph_for_size(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ):
        """Capture a CUDA graph for a specific batch size."""
        static_input = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
        static_noise = torch.randn(size, self.n_acoustic_codebook, device=device, dtype=dtype)
        static_cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)

        # Stabilizing eager run
        with torch.no_grad():
            _ = self._forward_cudagraph_compatible(static_input, cfg_alpha=static_cfg_alpha, noise=static_noise)

        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self.graph_pool):
                static_fake_eos, static_audio_codes = self._forward_cudagraph_compatible(
                    static_input, cfg_alpha=static_cfg_alpha, noise=static_noise
                )

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_noise[size] = static_noise
        self.static_cfg_alpha[size] = static_cfg_alpha
        self.static_fake_eos[size] = static_fake_eos
        self.static_audio_codes[size] = static_audio_codes

    def _get_padded_size(self, actual_size: int) -> int | None:
        """Round up to the nearest captured bucket size."""
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def __call__(
        self,
        backbone_hidden_states: torch.Tensor,
        cfg_alpha_per_req: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the acoustic head via a captured CUDA graph.

        Args:
            backbone_hidden_states: ``(B, hidden_dim)`` backbone hidden states.
            cfg_alpha_per_req: ``(B,)`` per-request CFG strength.

        Returns:
            ``(audio_codes, fake_eos)`` where ``audio_codes`` is ``(B, 37)`` int
            and ``fake_eos`` is ``(B,)`` bool.

        Raises:
            RuntimeError: if no captured graph covers the requested batch size,
            or if the wrapper has not been warmed up.
        """
        actual_size = backbone_hidden_states.shape[0]

        if not self.enabled or not self._warmed_up:
            raise RuntimeError("AcousticHeadCudaGraph is not warmed up; no captured graphs available.")

        padded_size = self._get_padded_size(actual_size)
        if padded_size is None or padded_size not in self.graphs:
            raise RuntimeError(
                f"AcousticHeadCudaGraph has no captured graph for batch size {actual_size} "
                f"(capture sizes: {self.capture_sizes})."
            )

        # Zero static input, then copy actual data
        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:actual_size] = backbone_hidden_states

        # Copy per-request cfg_alpha into static buffer (pad with 1.2 default).
        # The wrapper writes per-request cfg_alpha into static_cfg_alpha[:B, 0].
        self.static_cfg_alpha[padded_size].fill_(1.2)
        self.static_cfg_alpha[padded_size][:actual_size, 0] = cfg_alpha_per_req

        # Fill noise buffer with fresh random values before replay so the
        # flow-matching ODE starts from different initial noise each time.
        # Uses a seeded torch.Generator so RNG is not baked into the graph
        # (gap fix #8).
        self.static_noise[padded_size].normal_(generator=self._noise_generator)

        # Replay captured graph
        self.graphs[padded_size].replay()

        # Clone and slice outputs for actual batch size
        fake_eos = self.static_fake_eos[padded_size][:actual_size].clone().bool()
        audio_codes = self.static_audio_codes[padded_size][:actual_size].clone()

        return audio_codes, fake_eos


def _smoke_test() -> None:
    """
    GPU-less shape smoke test (gap fix #12).

    Uses a mock graph pool and a tiny stub ``flow_transformer`` so the buffer
    allocation / shape logic can be exercised on CPU with NO CUDA and NO
    dependency on agent P1's ``FlowMatchingAudioTransformer``.

    NOTE: actual ``torch.cuda.graph`` capture cannot run on CPU, so this test
    exercises only the static-buffer allocation and the
    ``_forward_cudagraph_compatible`` shape/dtype logic. The CUDA capture /
    replay path is not covered here.
    """
    import types

    hidden_dim = 16
    n_acoustic_codebook = 36
    semantic_codebook_size = 64
    n_decoding_steps = 3
    acoustic_embeddings_levels = 21
    n_codebooks = 1 + n_acoustic_codebook  # 37
    # vocab dim of the semantic head: specials + semantic codebook + a margin
    semantic_vocab = len(AudioSpecialTokens) + semantic_codebook_size + 8

    class _StubFlowTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_args = types.SimpleNamespace(
                semantic_codebook_size=semantic_codebook_size,
                n_acoustic_codebook=n_acoustic_codebook,
            )
            self.acoustic_transformer_args = types.SimpleNamespace(n_decoding_steps=n_decoding_steps)
            self.acoustic_embeddings_levels = acoustic_embeddings_levels
            self._sem = torch.nn.Linear(hidden_dim, semantic_vocab)
            self._time = torch.nn.Linear(1, hidden_dim)
            self._vel = torch.nn.Linear(n_acoustic_codebook + hidden_dim + hidden_dim, n_acoustic_codebook)

        def semantic_codebook_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self._sem(hidden_states)

        def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
            return self._time(t)

        def _predict_velocity(
            self, x_t: torch.Tensor, llm_output: torch.Tensor, t_emb: torch.Tensor
        ) -> torch.Tensor:
            return self._vel(torch.cat([x_t, llm_output, t_emb], dim=-1))

    device = torch.device("cpu")
    dtype = torch.float32
    stub = _StubFlowTransformer().to(device=device, dtype=dtype)
    mock_graph_pool = object()  # never actually used on CPU
    capture_sizes = [1, 2, 4]

    # Build the wrapper but skip the CUDA-only warmup/capture: instead exercise
    # the constants + buffer allocation + the captured-body shape logic directly.
    obj = AcousticHeadCudaGraph.__new__(AcousticHeadCudaGraph)
    obj.acoustic_transformer = stub
    obj.graph_pool = mock_graph_pool
    obj.capture_sizes = sorted(capture_sizes)
    obj.device = device
    obj.dtype = dtype
    obj.hidden_dim = hidden_dim
    obj.seed = 42
    obj.empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
    obj.end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    obj.semantic_mask_start = len(AudioSpecialTokens) + stub.model_args.semantic_codebook_size
    obj.n_acoustic_codebook = stub.model_args.n_acoustic_codebook
    obj.acoustic_embeddings_levels = stub.acoustic_embeddings_levels
    obj.n_steps = stub.acoustic_transformer_args.n_decoding_steps
    obj._noise_generator = torch.Generator(device=device)
    obj._noise_generator.manual_seed(obj.seed)
    obj.timesteps = torch.linspace(0, 1, obj.n_steps + 1, device=device, dtype=dtype)
    obj.fake_eos_one = torch.tensor(1.0, dtype=dtype, device=device)
    obj.fake_eos_zero = torch.tensor(0.0, dtype=dtype, device=device)

    # Exercise the captured-body shape logic for each capture size.
    for size in obj.capture_sizes:
        hidden_states = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
        cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)
        noise = torch.randn(size, n_acoustic_codebook, device=device, dtype=dtype)
        with torch.no_grad():
            fake_eos, audio_codes = obj._forward_cudagraph_compatible(
                hidden_states, cfg_alpha=cfg_alpha, noise=noise
            )
        assert audio_codes.shape == (size, n_codebooks), (
            f"audio_codes shape {tuple(audio_codes.shape)} != {(size, n_codebooks)}"
        )
        assert audio_codes.dtype == torch.long, f"audio_codes dtype {audio_codes.dtype} != torch.long"
        assert fake_eos.shape == (size,), f"fake_eos shape {tuple(fake_eos.shape)} != {(size,)}"

    # Exercise the static-buffer allocation path (the non-graph parts of
    # _capture_graph_for_size).
    for size in obj.capture_sizes:
        static_input = torch.zeros(size, hidden_dim, device=device, dtype=dtype)
        static_noise = torch.randn(size, n_acoustic_codebook, device=device, dtype=dtype)
        static_cfg_alpha = torch.full((size, 1), 1.2, device=device, dtype=dtype)
        with torch.no_grad():
            static_fake_eos, static_audio_codes = obj._forward_cudagraph_compatible(
                static_input, cfg_alpha=static_cfg_alpha, noise=static_noise
            )
        # Allocate the dicts lazily for the test.
        for attr in ("static_inputs", "static_noise", "static_cfg_alpha", "static_fake_eos", "static_audio_codes"):
            if not hasattr(obj, attr):
                setattr(obj, attr, {})
        obj.static_inputs[size] = static_input
        obj.static_noise[size] = static_noise
        obj.static_cfg_alpha[size] = static_cfg_alpha
        obj.static_fake_eos[size] = static_fake_eos
        obj.static_audio_codes[size] = static_audio_codes
        assert static_audio_codes.shape == (size, n_codebooks)

    # Exercise _get_padded_size rounding logic.
    assert obj._get_padded_size(1) == 1
    assert obj._get_padded_size(3) == 4
    assert obj._get_padded_size(4) == 4
    assert obj._get_padded_size(5) is None

    # Exercise the __call__ buffer-prep + slicing logic without an actual
    # captured graph: install a fake graph object whose replay() is a no-op.
    obj.enabled = True
    obj._warmed_up = True

    class _FakeGraph:
        def replay(self):  # noqa: D401 - no-op for CPU smoke test
            pass

    obj.graphs = {size: _FakeGraph() for size in obj.capture_sizes}
    bsz = 3
    backbone_hidden_states = torch.randn(bsz, hidden_dim, device=device, dtype=dtype)
    cfg_alpha_per_req = torch.full((bsz,), 1.2, device=device, dtype=dtype)
    audio_codes, fake_eos = obj(backbone_hidden_states, cfg_alpha_per_req)
    # padded to bucket 4, then sliced back to bsz
    assert audio_codes.shape == (bsz, n_codebooks), f"__call__ audio_codes shape {tuple(audio_codes.shape)}"
    assert audio_codes.dtype == torch.long, f"__call__ audio_codes dtype {audio_codes.dtype}"
    assert fake_eos.shape == (bsz,), f"__call__ fake_eos shape {tuple(fake_eos.shape)}"
    assert fake_eos.dtype == torch.bool, f"__call__ fake_eos dtype {fake_eos.dtype}"

    print("AcousticHeadCudaGraph smoke test passed (CPU-only: buffer + shape logic; CUDA capture not exercised).")


if __name__ == "__main__":
    _smoke_test()
