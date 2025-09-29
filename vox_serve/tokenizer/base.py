from dataclasses import dataclass, fields
from typing import Any, List

import torch


@dataclass
class DecoderCache:
    """Base class for decoder caches.

    This is an empty dataclass intended to be inherited by
    model-specific decoder cache containers.
    """
    def __getitem__(self, index: Any):
        """Return a sliced view of the cache along the batch dimension.

        Applies the given `index` (e.g., `:n`, a tensor of indices, etc.)
        to the first dimension of all member tensors recursively. Non-tensor
        members are left unchanged.
        """

        def _slice(obj: Any):
            if torch.is_tensor(obj):
                return obj[index]
            if isinstance(obj, DecoderCache):
                return obj[index]
            if isinstance(obj, list):
                return [
                    _slice(x)
                    for x in obj
                ]
            if isinstance(obj, tuple):
                return tuple(_slice(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _slice(v) for k, v in obj.items()}
            return obj

        sliced_values = {f.name: _slice(getattr(self, f.name)) for f in fields(self)}
        return type(self)(**sliced_values)

    @classmethod
    def cat(cls, caches: List["DecoderCache"]) -> "DecoderCache":
        """Concatenate a list of caches along the batch (dim=0).

        All caches must be of the same subclass and have compatible
        member shapes. Tensor members are concatenated on dim 0.
        Nested DecoderCache and container members (list/tuple/dict)
        are merged recursively. Non-tensor members must be identical
        (or all None) across caches.
        """
        if not caches:
            raise ValueError("caches must be a non-empty list")

        cache_type = type(caches[0])
        if not all(isinstance(c, cache_type) for c in caches):
            raise TypeError("All caches must be instances of the same cache class")

        def _merge(values: List[Any]) -> Any:
            first = values[0]

            if torch.is_tensor(first):
                return torch.cat(values, dim=0)

            if isinstance(first, DecoderCache):
                return type(first).cat(values)  # type: ignore[arg-type]

            if isinstance(first, list):
                if not all(isinstance(v, list) and len(v) == len(first) for v in values):
                    raise ValueError("List fields must have the same length to merge")
                return [
                    _merge([v[i] for v in values])
                    for i in range(len(first))
                ]

            if isinstance(first, tuple):
                if not all(isinstance(v, tuple) and len(v) == len(first) for v in values):
                    raise ValueError("Tuple fields must have the same length to merge")
                return tuple(_merge([v[i] for v in values]) for i in range(len(first)))

            if isinstance(first, dict):
                keys = first.keys()
                if not all(isinstance(v, dict) and v.keys() == keys for v in values):
                    raise ValueError("Dict fields must have the same keys to merge")
                return {k: _merge([v[k] for v in values]) for k in keys}

            if all(v is None for v in values):
                return None

            if all(v == first for v in values):
                return first

            raise TypeError(f"Unsupported or mismatched member type for merging: {type(first)}")

        merged_values = {f.name: _merge([getattr(c, f.name) for c in caches]) for f in fields(cache_type)}
        return cache_type(**merged_values)
