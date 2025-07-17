from dataclasses import dataclass, field
from typing import Optional, Union, List

import numpy as np 

from .sampling import SamplingConfig

@dataclass
class Request:
    """
    Base class for all requests.
    """
    request_id: str 
    prompt: str
    sampling_config: SamplingConfig = None

    # next_position_id == len(input_tokens) + len(lm_output_tokens) + 1
    next_position_id: int = None

    # KV cache data 
    kv_pages: List[int] = None
    kv_token_len: int = None
    kv_last_page_len: int = None
    # kv_token_len == (len(kv_pages) - 1) * page_size + kv_last_page_len

    input_tokens: List[int] = None
    lm_output_tokens: List[List[int]] | List[int] = field(default_factory=list)
    output_audio: List[bytes] = field(default_factory=list)

    # progress status
    done_lm_prefill: bool = False
    next_audio_decode_idx: int = 0
    is_audio_available: bool = False 
    done_all: bool = False

    # future optimization 
    is_pressing: bool = False 
    
    # cache for calculating repetition penalty, boolean list of length vocab_size
    repetition_cache: List[bool] = None
    tokens_mask: List[bool] = None
