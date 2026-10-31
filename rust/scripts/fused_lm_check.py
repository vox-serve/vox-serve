"""--fuse-layers check: the fused decoder stack against the modules' forward on the same inputs.

Builds the engine with fuse_layers, then for a decode batch and a prefill runs the backbone both ways
(fstack on / off) on the embeddings of random token ids and compares the hidden states and the KV entries
written for those rows. Also times both paths eagerly.

python rust/scripts/fused_lm_check.py --model higgs-tts
"""

import argparse
import json

import torch

from vox_serve.rt import core
from vox_serve.rt.core import build_engine


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def eager_ms(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-batch-size", type=int, default=8)
    a = p.parse_args()
    config = {"model": a.model, "max_batch_size": a.max_batch_size, "page_size": 64, "max_context": 2048}
    config.update({"device": 0, "lm_segments": "whole", "fuse_layers": True, "prefill_buckets": [256]})
    build_engine(json.dumps(config))
    program, builder = core._ENGINE_KEEPALIVE[-1]
    assert program.fstack is not None
    inp = builder.inputs
    torch.manual_seed(0)
    n = a.max_batch_size
    for kind, rows in (("decode", n), ("prefill", 256)):
        attn = program._attention(kind, inp, n, rows)
        program._prepare_capture_rows(inp, rows, n)
        # Distinct KV slots per row (capture rows all write page 0); attention still reads page 0 only.
        r = torch.arange(rows, device=program.device)
        inp["lm_kv_page"][:rows].copy_(1 + r // program.page_size)
        inp["pos"][:rows].copy_(100 + r)  # non-trivial RoPE angles
        torch.cuda.synchronize()
        pos = inp["pos"][:rows]
        ids = torch.randint(0, 1000, (rows, program.n_codebooks), device=program.device)
        with torch.no_grad():
            h = program.embed(ids, inp, rows)
            page, off = attn.kv_page[:rows].long(), attn.kv_off[:rows].long()
            fs = program.fstack
            outs = {}
            for name, stack in (("module", None), ("fused", fs)):
                program.fstack = stack
                o = program.backbone(h, pos, attn)
                kv = torch.stack([c[page, :, off] for c in program.kv_cache])  # [L, rows, 2, nkv, hd]
                t = eager_ms(lambda h=h, pos=pos, attn=attn: program.backbone(h, pos, attn))
                outs[name] = (o.clone(), kv.clone(), t)
            # Sensitivity of the module path itself: the same inputs perturbed by ~1 bf16 ulp.
            program.fstack = None
            hp = h * (1 + torch.randn_like(h, dtype=torch.float32) * 2**-9).to(h.dtype)
            op = program.backbone(hp, pos, attn).clone()
            program.fstack = fs
        (o0, k0, t0), (o1, k1, t1) = outs["module"], outs["fused"]
        print(
            f"{a.model} {kind} rows={rows}: hidden rel err {rel(o1, o0):.2e}"
            f" (max abs {(o1.float() - o0.float()).abs().max():.3g}),"
            f" kv rel err {rel(k1, k0):.2e}; layer-0 kv rel err {rel(k1[0], k0[0]):.2e};"
            f" module vs module on 1-ulp-perturbed input {rel(op, o0):.2e}; eager {t0:.3f} -> {t1:.3f} ms"
        )


if __name__ == "__main__":
    main()
