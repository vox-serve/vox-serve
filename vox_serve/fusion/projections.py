"""Model-agnostic LM fusion: sibling projections that read the same input become one GEMM.

``merge_projections(root)`` finds every module under ``root`` with ``q_proj`` / ``k_proj`` / ``v_proj``
(and ``gate_proj`` / ``up_proj``) linears and replaces them with members of a merged group: the first
member computes all outputs with one GEMM over the concatenated weights, and its siblings return their
slice when called with the same input tensor. Module forwards stay untouched, so this applies to any
Llama / Qwen-style layer. A sibling called with a different input (or out of order) falls back to its
own slice of the merged weight, so results never change, only the kernel count.

Runs before graph capture: the Python-level caching happens only while graphs are recorded.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

GROUPS: Tuple[Tuple[str, ...], ...] = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))


class _Group:
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], splits: List[int]):
        self.weight = weight
        self.bias = bias
        self.splits = splits
        self.key: Optional[torch.Tensor] = None
        self.outs: Optional[Sequence[torch.Tensor]] = None


class MergedLinearMember(nn.Module):
    def __init__(self, group: _Group, idx: int, start: int, size: int):
        super().__init__()
        self.group = group
        self.idx = idx
        self.in_features = group.weight.shape[1]
        self.out_features = size
        # Views into the merged weight: code that reads ``.weight`` keeps working.
        self.weight = nn.Parameter(group.weight[start : start + size], requires_grad=False)
        self.bias = None if group.bias is None else nn.Parameter(group.bias[start : start + size], requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.group
        if torch.compiler.is_compiling():
            return F.linear(x, self.weight, self.bias)
        if self.idx == 0:
            g.outs = F.linear(x, g.weight, g.bias).split(g.splits, dim=-1)
            g.key = x
        if g.key is x:
            out = g.outs[self.idx]
            if self.idx == len(g.splits) - 1:
                g.key = g.outs = None
            return out
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, member {self.idx} of {len(self.group.splits)}"


def merge_projections(root: nn.Module, groups: Sequence[Sequence[str]] = GROUPS) -> int:
    """Merges each group of sibling linears under ``root``; returns the number of groups merged."""
    n = 0
    for mod in list(root.modules()):
        for names in groups:
            lins = [getattr(mod, a, None) for a in names]
            if not all(type(lin) is nn.Linear for lin in lins):
                continue
            if len({(lin.in_features, lin.weight.dtype, lin.weight.device) for lin in lins}) != 1:
                continue
            if len({lin.bias is None for lin in lins}) != 1:
                continue
            with torch.no_grad():
                weight = torch.cat([lin.weight for lin in lins], 0).contiguous()
                bias = None if lins[0].bias is None else torch.cat([lin.bias for lin in lins], 0).contiguous()
            splits = [lin.out_features for lin in lins]
            group = _Group(weight, bias, splits)
            start = 0
            for i, (a, size) in enumerate(zip(names, splits, strict=True)):
                setattr(mod, a, MergedLinearMember(group, i, start, size))
                start += size
            n += 1
    return n
