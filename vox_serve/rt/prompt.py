"""Prompts and the CPU-side model constants, shared by the Python runtime and the programs.

Every program builds its prompts in Python (``Program.preprocess``): the Python runtime uses them, and
the Rust runtime too when ``meta["prompt_builder"]`` is ``python``. Models served at high request rates
also have a Rust prompt builder (``rust/src/models``) that must produce the same rows.
"""

import hashlib
import struct
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class ResourceBusyError(RuntimeError):
    """A resource a prompt needs (a voice slot, feature-pool rows) is held by other requests right now.

    Prompts built in Python must not wait for one: the Python runtime builds them on its scheduler
    thread, the same thread that releases these resources when requests finish. Its worker retries
    the request on a later step instead."""


class FeaturePool:
    """Allocator of the program's device feature-pool rows (``LMProgram.feature_pool``) for prompts
    built in Python: a request's prompt features occupy a contiguous range from preprocessing until its
    prefill has run. (The Rust runtime allocates rows itself, ``rust/src/models/features.rs``.)"""

    def __init__(self, rows: int):
        self.rows = rows
        self._free: List[Tuple[int, int]] = [(0, rows)] if rows > 0 else []  # sorted, coalesced
        self._lock = threading.RLock()  # reentrant: a lease's ``__del__`` may run inside ``alloc``

    def alloc(self, n: int) -> "FeatureLease":
        """First fit; ``ResourceBusyError`` while no free range is large enough (other requests hold the rows)."""
        if n <= 0 or n > self.rows:
            raise ValueError(f"feature rows {n} out of range (pool has {self.rows})")
        with self._lock:
            for i, (start, size) in enumerate(self._free):
                if size >= n:
                    if size == n:
                        del self._free[i]
                    else:
                        self._free[i] = (start + n, size - n)
                    return FeatureLease(self, start, n)
        raise ResourceBusyError(f"feature pool exhausted ({n} rows requested)")

    def _release(self, start: int, n: int) -> None:
        with self._lock:
            self._free.append((start, n))
            self._free.sort()
            merged: List[Tuple[int, int]] = []
            for s, k in self._free:
                if merged and merged[-1][0] + merged[-1][1] == s:
                    merged[-1] = (merged[-1][0], merged[-1][1] + k)
                else:
                    merged.append((s, k))
            self._free = merged


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


class VoiceSlots:
    """An LRU of reference voices encoded into a program's voice slots, for prompts built in Python; slot 0
    holds the default voice. A request holds its voice (``hold``, kept as ``Prompt.hold``) until it finishes,
    and a slot is only re-encoded once nothing holds its voice. ``get`` never waits for one: it raises
    ``ResourceBusyError`` while every slot is held. (Rust prompt builders keep slots of their own.)"""

    class Voice:
        def __init__(self, slot: int, data: Any = None):
            self.slot, self.data, self.users = slot, data, 0

    class _Use:
        def __init__(self, voice: "VoiceSlots.Voice", lock: threading.Lock):
            self.voice, self._lock = voice, lock
            with lock:
                voice.users += 1

        def __del__(self):
            with self._lock:
                self.voice.users -= 1

    def __init__(self, n: int, default: Any = None):
        # Per slot: (content key, voice, last use) or None.
        self._slots: List[Optional[Tuple[str, "VoiceSlots.Voice", int]]] = [None] * max(1, n)
        self._slots[0] = ("", VoiceSlots.Voice(0, default), 0)
        self._clock = 0
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def default(self) -> "VoiceSlots.Voice":
        return self._slots[0][1]

    def hold(self, voice: "VoiceSlots.Voice") -> Any:
        return VoiceSlots._Use(voice, self._lock)

    def _lookup(self, key: str) -> Optional["VoiceSlots.Voice"]:
        self._clock += 1
        for i, e in enumerate(self._slots):
            if i > 0 and e is not None and e[0] == key:
                self._slots[i] = (key, e[1], self._clock)
                return e[1]
        return None

    def _reserve(self) -> Optional[int]:
        free = [i for i in range(1, len(self._slots)) if self._slots[i] is None]
        if free:
            return free[0]
        idle = [i for i in range(1, len(self._slots)) if self._slots[i][1].users == 0]
        if not idle:
            return None
        i = min(idle, key=lambda i: self._slots[i][2])
        self._slots[i] = None
        return i

    def get(self, path: str, encode: Callable[[int], Any]) -> "VoiceSlots.Voice":
        """The voice of reference clip ``path``; on a miss, ``encode(slot)`` writes it into a free slot."""
        with open(path, "rb") as fh:
            key = hashlib.sha1(fh.read()).hexdigest()
        with self._lock:
            v = self._lookup(key)
        if v is not None:
            return v
        if len(self._slots) < 2:
            raise ValueError("reference audio needs a voice slot (n_voices >= 2)")
        with self._encode_lock:
            with self._lock:
                v = self._lookup(key)
                if v is not None:
                    return v
                slot = self._reserve()
            if slot is None:
                raise ResourceBusyError(f"all {len(self._slots) - 1} voice slots are in use by running requests")
            v = VoiceSlots.Voice(slot, encode(slot))
            with self._lock:
                self._slots[slot] = (key, v, self._clock)
            return v


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
