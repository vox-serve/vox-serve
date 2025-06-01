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
    lm_output_tokens: List[List[int]] | List[int]
    output_audio: List[np.ndarray]

    # progress status
    done_lm_prefill: bool = False
    is_ready_detokenize: bool = False 
    is_audio_available: bool = False 
    done_all: bool = False

    # future optimization 
    is_pressing: bool = False 
    