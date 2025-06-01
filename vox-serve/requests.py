from dataclasses import dataclass
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
    sampling_config: SamplingConfig

    input_tokens: List[int]
    lm_output_tokens: List[List[int]] | List[int] = []
    output_audio: List[np.ndarray] = []

    # next_position_id == len(input_tokens) + len(lm_output_tokens) + 1
    next_position_id: int

    # KV cache data 
    kv_pages: List[int] 
    kv_token_len: int 
    kv_last_page_len: int
    # kv_token_len == (len(kv_pages) - 1) * page_size + kv_last_page_len

    # progress status
    done_lm_prefill: bool = False
    is_ready_detokenize: bool = False 
    is_audio_available: bool = False 
    done_all: bool = False

    # future optimization 
    is_pressing: bool = False 
    