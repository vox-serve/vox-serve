from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List, Optional, TypedDict

import torch

from .sampling import SamplingConfig
from .tokenizer.base import DecoderCache


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

    # next_position_id == len(input_tokens) + len(lm_output_tokens) + 1
    next_position_id: int = None

    # KV cache data
    kv_pages: List[int] = None
    kv_token_len: int = None
    kv_last_page_len: int = None
    # kv_token_len == (len(kv_pages) - 1) * page_size + kv_last_page_len

    # input prompt tokens. shape: (seq_len, n_codebooks)
    input_tokens: torch.Tensor = None
    # length of input prompt tokens
    input_length: int = None
    # raw output tokens from LM to be given to the next step of LM inference. shape: (seq_len, n_codebooks)
    lm_output_tokens: List[torch.Tensor] = field(default_factory=list)
    # audio tokens to be given to the detokenizer, after filtering or reverting delay patterns.
    # shape: (seq_len, n_codebooks)
    lm_output_audio_tokens: List[torch.Tensor] = field(default_factory=list)
    output_audio: Queue = field(default_factory=Queue)

    # optional inputs for inference or sampling
    input_features: torch.Tensor = None
    input_masks: torch.Tensor = None
    repetition_cache: torch.Tensor = None  # Shape: (window_size, n_codebooks, vocab_size)

    # optional per-request decoder cache for models whose postprocess requires it
    decoder_cache: DecoderCache = None

    # progress status
    done_lm_prefill: bool = False
    audio_decode_idx: List[int] = field(default_factory=list)
    next_audio_decode_idx: List[int] = field(default_factory=list)
    done_lm_generation: bool = False
    done_all: bool = False
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

    # timestamp tracking for online scheduling
    chunk_send_timestamps: List[float] = field(default_factory=list)  # when each chunk was sent
    chunk_durations: List[float] = field(default_factory=list)  # duration of each chunk in seconds

    # VibeVoice windowed generation state
    text_window_index: int = 0  # Current text window index
    speech_step_in_window: int = 0  # Current speech step within the window
    finished_tags: torch.Tensor = None  # [batch] bool tensor tracking which samples hit EOS
    eos_threshold: float = 0.5  # Threshold for EOS detection (sigmoid)
    max_speech_tokens: int = 1024  # Maximum speech tokens to generate

    # TTS-specific KV cache for VibeVoice (separate from LM KV cache)
    tts_kv_pages: List[int] = field(default_factory=list)
    tts_kv_token_len: int = 0
    tts_kv_last_page_len: int = 0

    # Text tokens for windowed processing (tokenized prompt)
    text_tokens: List[int] = field(default_factory=list)
    total_text_windows: int = 0  # Total number of text windows

    # Current hidden states for bridging LM → TTS LM
    lm_last_hidden_state: torch.Tensor = None
    tts_hidden_state: torch.Tensor = None
    eos_logits: torch.Tensor = None


class LMInputs(TypedDict):
    """Typed container for scheduler-prepared inputs for LM steps."""
    qo_indptr: List[int]
    paged_kv_indptr: List[int]
    paged_kv_indices: List[int]
    paged_kv_last_page_len: List[int]
    # Pre-allocated tensors for GPU computation
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    input_features: Optional[torch.Tensor]
    input_masks: Optional[torch.Tensor]
    repetition_cache: Optional[torch.Tensor]
    is_prefill: bool
