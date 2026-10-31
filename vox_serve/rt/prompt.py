"""Prompts and the CPU-side model constants, shared by the Python runtime and the programs.

Every program builds its prompts in Python (``Program.preprocess``): the Python runtime uses them, and
the Rust runtime too when ``meta["prompt_builder"]`` is ``python``. Models served at high request rates
also have a Rust prompt builder (``rust/src/models``) that must produce the same rows.
"""

import struct
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

TOKEN_AUDIO, TOKEN_STOP, TOKEN_OTHER = 0, 1, 2


@dataclass
class Prompt:
    """Prompt rows, flattened ``[L, n_codebooks]``, with an optional per-column input mask."""

    tokens: List[int]
    mask: Optional[List[int]] = None
    # Feature-pool rows referenced by the mask; released once the prompt has been prefilled.
    features: Optional["FeatureLease"] = None
    # Model resources the request uses until it finishes (e.g. a voice slot of the detokenizer).
    hold: Any = None


class FeaturePool:
    """Allocator of the program's device feature-pool rows (``LMProgram.feature_pool``): a request's
    prompt features occupy a contiguous range from preprocessing until its prefill has run."""

    def __init__(self, rows: int):
        self.rows = rows
        self._free: List[Tuple[int, int]] = [(0, rows)] if rows > 0 else []  # sorted, coalesced
        self._cv = threading.Condition()

    def alloc(self, n: int, timeout: float = 60.0) -> "FeatureLease":
        """First fit, waiting up to ``timeout`` seconds for other requests to release rows."""
        if n <= 0 or n > self.rows:
            raise ValueError(f"feature rows {n} out of range (pool has {self.rows})")
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                for i, (start, size) in enumerate(self._free):
                    if size >= n:
                        if size == n:
                            del self._free[i]
                        else:
                            self._free[i] = (start + n, size - n)
                        return FeatureLease(self, start, n)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError(f"feature pool exhausted ({n} rows requested)")
                self._cv.wait(left)

    def _release(self, start: int, n: int) -> None:
        with self._cv:
            self._free.append((start, n))
            self._free.sort()
            merged: List[Tuple[int, int]] = []
            for s, k in self._free:
                if merged and merged[-1][0] + merged[-1][1] == s:
                    merged[-1] = (merged[-1][0], merged[-1][1] + k)
                else:
                    merged.append((s, k))
            self._free = merged
            self._cv.notify_all()


@dataclass
class FeatureLease:
    pool: FeaturePool
    offset: int
    len: int
    _released: bool = field(default=False, repr=False)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.pool._release(self.offset, self.len)

    def __del__(self):
        self.release()


@lru_cache(maxsize=None)
def tokenizer(path: str):
    """A tokenizer.json, loaded with the same library as the Rust prompt builders."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def encode(path: str, text: str, add_special_tokens: bool = True) -> List[int]:
    return tokenizer(path).encode(text, add_special_tokens=add_special_tokens).ids


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    @classmethod
    def from_meta(cls, v: Dict[str, Any]) -> "SamplingParams":
        v = v or {}
        return cls(
            temperature=float(v.get("temperature", 1.0)),
            top_k=int(v.get("top_k", 0) or 0),
            top_p=float(v.get("top_p", 1.0)),
            min_p=float(v.get("min_p", 0.0)),
            repetition_penalty=float(v.get("repetition_penalty", 1.0)),
        )


@dataclass
class ModelSpec:
    """The constants of a program's ``meta`` that scheduling needs (the Rust runtime's ``Model``)."""

    n_codebooks: int
    max_tokens: int
    sampling: SamplingParams
    sample_rate: int = 24000
    decode_mask: Optional[List[int]] = None
    decode_row: Optional[List[int]] = None
    preseed: int = 0
    # Input streaming: (text column of decode rows, pad token, eos token).
    text_stream: Optional[Tuple[int, int, int]] = None

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "ModelSpec":
        ts = meta.get("text_stream")
        return cls(
            n_codebooks=int(meta["n_codebooks"]),
            max_tokens=int(meta["max_tokens"]),
            sampling=SamplingParams.from_meta(meta.get("sampling")),
            sample_rate=int(meta.get("sample_rate") or 24000),
            decode_mask=meta.get("decode_mask"),
            decode_row=meta.get("decode_row"),
            preseed=int(meta.get("preseed_frames") or 0),
            text_stream=(int(ts["column"]), int(ts["pad"]), int(ts["eos"])) if ts else None,
        )


def wav_header(sample_rate: int, data_len: int) -> bytes:
    """44-byte header of 16-bit mono PCM."""
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_len)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_len)
    )
