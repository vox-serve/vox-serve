from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List

import numpy as np

from .sampling import SamplingConfig


@dataclass
class Request:
    """
    Base class for all requests.
    """

    request_id: str
    prompt: str = None
    audio_path: str = None
    sampling_config: SamplingConfig = None
    # Model-specific kwargs (e.g., language, speaker, ref_text for Qwen3-TTS)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # position of the next token the LM processes (the prompt length after prefill)
    next_position_id: int = None

    # KV cache pages of the request
    kv_pages: List[int] = None

    # number of prompt rows (known once the program has built the prompt)
    input_length: int = None
    # sampled rows of every LM step, each of shape (n_codebooks,)
    lm_output_tokens: List[np.ndarray] = field(default_factory=list)
    # audio frames to be given to the detokenizer, each of shape (n_codebooks,)
    lm_output_audio_tokens: List[np.ndarray] = field(default_factory=list)
    output_audio: Queue = field(default_factory=Queue)

    # program state (ProgramWorker): the prompt built by the program (``vox_serve.rt.prompt.Prompt``),
    # the request's LM slot (last sampled token, repetition-penalty state) and detokenizer slot, and its
    # sampling seed / steps
    lm_prompt: Any = None
    slot: int = None
    detok_slot: int = None
    seed: int = None
    lm_steps: int = 0

    # progress status
    done_lm_prefill: bool = False
    audio_decode_idx: List[int] = field(default_factory=list)
    next_audio_decode_idx: List[int] = field(default_factory=list)
    done_lm_generation: bool = False
    done_all: bool = False
    # the API server has been told that the prompt is built (``ACCEPTED``: a streaming response starts)
    accepted_sent: bool = False
    # the completion message has been sent (async scheduling may select a request once more after that)
    completion_sent: bool = False
    finish_reason: str = None

    is_pressing: bool = False
    is_streaming: bool = False

    # Input streaming support - allows incremental text input during generation
    is_input_streaming: bool = False
    input_text_buffer: str = ""  # Buffer for accumulating text before prefill
    pending_text_tokens: Queue = field(default_factory=Queue)  # Queue of text token IDs for decode phase
    text_token_cursor: int = 0  # Index of next text token to consume
    total_text_tokens: int = 0  # Total text tokens received so far
    text_complete: bool = False  # True when TEXT_COMPLETE received
    waiting_for_text: bool = False  # True when generation paused waiting for text
    prefill_ready: bool = False  # True when enough text buffered to start prefill
    eos_injected: bool = False  # True after EOS token has been injected (for input streaming)
    first_text_token: int = None  # The only text token in the prefill prompt (for input streaming)

    # timestamp tracking for online scheduling
    chunk_send_timestamps: List[float] = field(default_factory=list)  # when each chunk was sent
    chunk_durations: List[float] = field(default_factory=list)  # duration of each chunk in seconds

