"""
Encoder Strategy for stateless encoding operations.

This strategy handles text and audio encoding, which are stateless
operations that don't require cache management.
"""

from typing import Any, Dict, List, Optional

from ..model.base import BaseLM
from ..requests import Request
from .base import (
    AllocatedResources,
    GenerationStrategy,
    ResourceSpec,
    StrategyPhase,
    StrategyType,
)


class EncoderStrategy(GenerationStrategy):
    """
    Strategy for stateless encoder operations.

    Handles preprocessing of inputs (text tokenization, audio encoding)
    without maintaining any state between calls. This is typically the
    first step in the generation pipeline.

    Features:
    - Text tokenization
    - Audio feature extraction (for speech-to-speech models)
    - Input mask generation
    - Repetition cache initialization
    """

    def __init__(self, model: BaseLM, **kwargs):
        super().__init__(model, **kwargs)
        self._phases = [
            StrategyPhase(
                name="encode",
                is_stateful=False,  # Encoders are stateless
                requires_cache=False,
                batch_size_limits=None,  # Can process any batch size
            ),
        ]

    @property
    def name(self) -> str:
        return "encoder"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.ENCODER

    @property
    def phases(self) -> List[StrategyPhase]:
        return self._phases

    def resource_spec(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
    ) -> ResourceSpec:
        """
        Encoders don't require caches or CUDA graphs.
        """
        return ResourceSpec(
            cache_specs=[],  # No cache needed
            requires_cuda_graph=False,  # Dynamic input sizes
            requires_flashinfer=False,  # No attention
        )

    def prepare_inputs(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
    ) -> Dict[str, Any]:
        """
        Prepare requests for encoding.

        Simply returns the requests as-is since encoding operates
        on request.prompt and request.audio_path directly.
        """
        if len(requests) == 0:
            return None

        return {
            "requests": requests,
        }

    async def execute(
        self,
        requests: List[Request],
        resources: AllocatedResources,
        phase: str,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Execute encoding for all requests.

        Calls model.preprocess() for each request and stores results
        in the request objects.

        Returns:
            List of PreprocessOutput for each request
        """
        if len(requests) == 0:
            return []

        outputs = []
        for req in requests:
            preprocess_output = self.model.preprocess(
                prompt=req.prompt,
                audio_path=req.audio_path,
            )

            # Store results in request
            req.input_tokens = preprocess_output.input_tokens
            if req.input_tokens is not None:
                req.input_length = req.input_tokens.shape[0]

            if preprocess_output.input_features is not None:
                req.input_features = preprocess_output.input_features
            if preprocess_output.input_masks is not None:
                req.input_masks = preprocess_output.input_masks
            if preprocess_output.repetition_cache is not None:
                req.repetition_cache = preprocess_output.repetition_cache
            if getattr(preprocess_output, "decoder_cache", None) is not None:
                req.decoder_cache = preprocess_output.decoder_cache

            outputs.append(preprocess_output)

        return outputs

    def post_process(
        self,
        requests: List[Request],
        outputs: Any,
        resources: AllocatedResources,
    ) -> None:
        """
        No post-processing needed for encoding.
        Results are stored in requests during execute().
        """
        pass

    def requires_cuda_graph_for_phase(self, phase: str) -> bool:
        """Encoders don't use CUDA graphs due to variable input sizes."""
        return False

    @property
    def supports_audio_input(self) -> bool:
        """Check if the model supports audio input encoding."""
        return self.model.supports_audio_input
