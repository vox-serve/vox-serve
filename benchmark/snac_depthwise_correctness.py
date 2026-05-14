"""Strict correctness test of the Triton depthwise Conv1d against torch.nn.functional.conv1d.

Same weights, same inputs, both paths in the same process. Tests the kernel in isolation
(not embedded in the SNAC decoder), so cuDNN algorithm variance can't muddy the comparison.

Covers all (C, T, K, dilation) combinations actually used by the SNAC 24kHz decoder.
"""

import torch
import torch.nn.functional as F

from vox_serve.tokenizer.snac import _TritonDepthwiseConv1d


def case(C, T, K, dilation, padding, dtype, bs_list=(1, 8, 64), atol_rel=2.0):
    """Test one (C, T, K, dilation) shape. Returns (passed, worst_max_abs_diff, worst_max_rel)."""
    torch.manual_seed(0)
    weight = (torch.randn(C, 1, K, device="cuda") * 0.3).to(dtype)
    bias = (torch.randn(C, device="cuda") * 0.1).to(dtype)

    # Build the Triton module with these weights.
    base = torch.nn.Conv1d(C, C, kernel_size=K, padding=padding, dilation=dilation,
                           groups=C, bias=True).cuda().to(dtype)
    base.weight.data = weight.clone()
    base.bias.data = bias.clone()
    tri = _TritonDepthwiseConv1d(base).cuda()

    worst_abs, worst_rel = 0.0, 0.0
    for bs in bs_list:
        x = torch.randn(bs, C, T, device="cuda", dtype=dtype)

        ref = F.conv1d(x, weight, bias, stride=1, padding=padding,
                       dilation=dilation, groups=C)
        out = tri(x)

        abs_diff = (ref - out).abs()
        rel_diff = abs_diff / ref.abs().clamp(min=1e-3)
        worst_abs = max(worst_abs, abs_diff.max().item())
        worst_rel = max(worst_rel, rel_diff.max().item())

    return worst_abs, worst_rel


def main():
    torch.set_grad_enabled(False)

    # Shapes from the SNAC 24kHz decoder. After the initial WNConv1d(1536,1536,k=7,groups=1536)
    # plus 12 ResidualUnit depthwise convs, the (channels, kernel, dilation) combinations are:
    #
    #   initial:  C=1536, k=7, dilation=1, padding=3
    #   residual units (k=7, paddings = (k-1)*d // 2):
    #     DecoderBlock 0 (C=768):  dilations 1, 3, 9
    #     DecoderBlock 1 (C=384):  dilations 1, 3, 9
    #     DecoderBlock 2 (C=192):  dilations 1, 3, 9
    #     DecoderBlock 3 (C=96):   dilations 1, 3, 9
    # T grows by the upsampling rates [7, 7, 3, 3] starting from T_in≈16.
    K = 7
    shapes = [
        # (C, T, K, dilation, padding)
        (1536,   16, K, 1,  3),
        (768,   112, K, 1,  3),  (768,   112, K, 3, 9),  (768,   112, K, 9, 27),
        (384,   784, K, 1,  3),  (384,   784, K, 3, 9),  (384,   784, K, 9, 27),
        (192,  2352, K, 1,  3),  (192,  2352, K, 3, 9),  (192,  2352, K, 9, 27),
        (96,   7056, K, 1,  3),  (96,   7056, K, 3, 9),  (96,   7056, K, 9, 27),
    ]

    print(f"{'C':>5} {'T':>5} {'k':>2} {'dil':>3} {'pad':>3} | {'dtype':>6} | {'max|Δ|':>10} {'max rel':>10}")
    print("-" * 70)
    failed = []
    for dtype in (torch.bfloat16, torch.float16):
        for C, T, k, d, p in shapes:
            abs_diff, rel_diff = case(C, T, k, d, p, dtype)
            tag = "" if abs_diff < 0.05 else "  ← LARGE"
            print(f"{C:>5} {T:>5} {k:>2} {d:>3} {p:>3} | {str(dtype).split('.')[-1]:>6} | "
                  f"{abs_diff:>10.2e} {rel_diff:>10.2e}{tag}")
            if abs_diff > 0.05:
                failed.append((C, T, k, d, dtype, abs_diff))

    print()
    if failed:
        print(f"FAIL: {len(failed)} shape(s) exceeded |Δ| > 0.05")
    else:
        print("PASS: all shapes within tolerance")


if __name__ == "__main__":
    main()
