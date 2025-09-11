from dataclasses import dataclass, field
from queue import Queue
from typing import List, Optional, TypedDict

import torch

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

    # progress status
    done_lm_prefill: bool = False
    audio_decode_idx: int = 0
    next_audio_decode_idx: int = 0
    done_lm_generation: bool = False
    done_all: bool = False
    finish_reason: str = None

    is_pressing: bool = False
    is_streaming: bool = False

    # timestamp tracking for online scheduling
    chunk_send_timestamps: List[float] = field(default_factory=list)  # when each chunk was sent
    chunk_durations: List[float] = field(default_factory=list)  # duration of each chunk in seconds


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
