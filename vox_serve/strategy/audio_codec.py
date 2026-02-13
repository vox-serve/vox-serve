"""
Audio Codec Strategy for detokenization.

This strategy handles converting LM output tokens to audio waveforms
using audio codecs like SNAC, Mimi, HiFiGAN, DAC, etc.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..model.base import BaseLM
from ..requests import Request
from .base import (
    AllocatedResources,
    CacheSpec,
    CacheType,
    GenerationStrategy,
    ResourceSpec,
    StrategyPhase,
    StrategyType,
)


class AudioCodecStrategy(GenerationStrategy):
    """
    Strategy for audio codec decoding (detokenization).

    Converts discrete audio tokens from the LLM into continuous audio waveforms.
    Some codecs maintain state across chunks (decoder cache), making this
    a stateful strategy.

    Features:
    - Interval-based decoding with overlap for smooth streaming
    - Optional watermarking
    - Per-request decoder cache management
    """

    def __init__(
        self,
        model: BaseLM,
        detokenize_interval: Optional[int] = None,
        detokenize_overlap: Optional[int] = None,
        needs_watermarking: bool = False,
        watermarker_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        # Use model defaults if not specified
        self._detokenize_interval = detokenize_interval or model.detokenize_interval
        self._detokenize_overlap = detokenize_overlap or model.detokenize_overlap
        self._needs_watermarking = needs_watermarking or model.needs_watermarking
        self._watermarker_type = watermarker_type or model.watermarker_type

        self._phases = [
            StrategyPhase(
                name="decode",
                is_stateful=True,  # Some codecs have decoder cache
                requires_cache=True,
                batch_size_limits=None,
            ),
        ]

    @property
    def name(self) -> str:
        return "audio_codec"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.AUDIO_CODEC

    @property
    def phases(self) -> List[StrategyPhase]:
        return self._phases

    @property
    def detokenize_interval(self) -> int:
        """Number of tokens to process per decoding call."""
        return self._detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        """Overlap between consecutive decoding windows."""
        return self._detokenize_overlap

    @property
    def detokenize_step(self) -> int:
        """Step size between consecutive decoding windows."""
        return self._detokenize_interval - self._detokenize_overlap

    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """
        Specify decoder cache requirements.

        Note: Some audio codecs don't need a cache (stateless decoders),
        but we include it for consistency. The cache_shape may be None
        for stateless codecs.
        """
        # Check if model needs decoder cache
        initial_cache = self.model.audio_decoder_initial_cache(batch_size=1)
        has_decoder_cache = initial_cache is not None

        cache_specs = []
        if has_decoder_cache:
            # Decoder cache is model-specific, so we don't specify shape here
            # Pool will call model.audio_decoder_initial_cache() to get it
            cache_specs.append(
                CacheSpec(
                    cache_type=CacheType.DECODER_CACHE,
                    shape=None,  # Shape is dynamic, managed per-request
                    dtype=torch.float32,
                )
            )

        return ResourceSpec(
            cache_specs=cache_specs,
            requires_cuda_graph=True,
            requires_flashinfer=False,  # Audio codec doesn't use attention
        )

    def prepare_inputs(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Prepare token windows for decoding.

        Collects token chunks from each request based on their
        audio_decode_idx positions.
        """
        if len(requests) == 0:
            return None

        # Collect token chunks and track which request each belongs to
        token_ids = []
        request_chunk_mapping = []  # (request_idx, chunk_idx)
        decoder_caches = []

        for req_idx, req in enumerate(requests):
            # Process multiple chunks from the same request if available
            for chunk_idx in range(len(req.audio_decode_idx)):
                decode_idx = req.audio_decode_idx[chunk_idx]
                new_tokens = req.lm_output_audio_tokens[decode_idx : decode_idx + self._detokenize_interval]

                # Pad if needed
                if len(new_tokens) < self._detokenize_interval:
                    new_tokens = list(new_tokens)
                    new_tokens.extend([new_tokens[-1]] * (self._detokenize_interval - len(new_tokens)))

                token_ids.append(torch.cat(new_tokens, dim=0))
                request_chunk_mapping.append((req_idx, chunk_idx))

                # Collect decoder cache if available
                if req.decoder_cache is not None:
                    decoder_caches.append(req.decoder_cache)

        if not token_ids:
            return None

        return {
            "token_ids": token_ids,
            "request_chunk_mapping": request_chunk_mapping,
            "decoder_caches": decoder_caches if decoder_caches else None,
        }

    async def execute(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute audio decoding.

        Converts token chunks to audio and puts results in request queues.

        Returns:
            Dictionary with audio tensors and metadata
        """
        if len(requests) == 0:
            return None

        if prepared_inputs is None:
            prepared_inputs = self.prepare_inputs(requests, resources, phase)

        if prepared_inputs is None:
            return None

        token_ids_list = prepared_inputs["token_ids"]
        request_chunk_mapping = prepared_inputs["request_chunk_mapping"]

        if not token_ids_list:
            return None

        # Stack tokens into batch
        token_ids = torch.stack(token_ids_list, dim=0)

        # Transfer to decoder device if needed
        device = resources.device
        if token_ids.device != torch.device(device):
            token_ids = token_ids.to(device, non_blocking=True)
            torch.cuda.synchronize(device=device)

        # Run audio decoding
        audio_tensors = self.model.postprocess(token_ids)

        return {
            "audio_tensors": audio_tensors,
            "request_chunk_mapping": request_chunk_mapping,
            "requests": requests,
        }

    def post_process(
        self,
        requests: List[Request],
        outputs: Any,
        resources: AllocatedResources,
    ) -> None:
        """
        Convert audio tensors to bytes and put in request output queues.
        """
        if outputs is None:
            return

        audio_tensors = outputs["audio_tensors"]
        request_chunk_mapping = outputs["request_chunk_mapping"]
        original_requests = outputs["requests"]

        # Process each chunk and assign to corresponding request
        for i, (req_idx, chunk_idx) in enumerate(request_chunk_mapping):
            req = original_requests[req_idx]
            decode_idx = req.audio_decode_idx[chunk_idx]

            # Convert to int16 audio bytes
            audio = audio_tensors[i].detach().cpu().numpy()
            audio_int16 = (audio * 32767).astype(np.int16)

            # Handle padding - trim if this was a partial chunk
            last_chunk_len = len(req.lm_output_audio_tokens[decode_idx : decode_idx + self._detokenize_interval])
            if last_chunk_len < self._detokenize_interval:
                trim_len = int(audio_int16.shape[1] * (last_chunk_len - 0.5) / self._detokenize_interval)
                audio_int16 = audio_int16[:, :trim_len]

            audio_bytes = audio_int16.tobytes()
            req.output_audio.put(audio_bytes)

        # Check if any request is completely done
        for req in original_requests:
            if req.done_lm_generation and (
                req.audio_decode_idx[-1] + self._detokenize_interval > len(req.lm_output_audio_tokens)
            ):
                req.done_all = True

    def requires_cuda_graph_for_phase(self, phase: str) -> bool:
        """Audio codec can use CUDA graphs for consistent performance."""
        return True

    def can_decode_chunk(self, request: Request) -> bool:
        """Check if request has enough tokens for another decode chunk."""
        if not request.next_audio_decode_idx:
            return False
        next_idx = request.next_audio_decode_idx[-1] + self.detokenize_step
        return next_idx + self._detokenize_interval <= len(request.lm_output_audio_tokens)

    def schedule_next_chunk(self, request: Request) -> bool:
        """
        Schedule the next decode chunk for a request.

        Returns True if a chunk was scheduled, False otherwise.
        """
        if not self.can_decode_chunk(request):
            return False

        if not request.next_audio_decode_idx:
            # First chunk
            request.next_audio_decode_idx = [0]
        else:
            # Add next chunk index
            next_idx = request.next_audio_decode_idx[-1] + self.detokenize_step
            request.next_audio_decode_idx.append(next_idx)

        return True
