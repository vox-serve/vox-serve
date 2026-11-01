from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingConfig:
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    repetition_window: Optional[int] = None  # -1 for global window
    cfg_scale: Optional[float] = None
    greedy: bool = False

