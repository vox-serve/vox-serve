"""Ladder of SNAC optimizations on top of `torch.compile + CUDA graph + weight_norm bake`.

We hold `weight_norm bake = ON` as the floor, compile + graph as the harness, and apply
optimizations one at a time, measuring at bs=1, 8, 64.
"""

import argparse
import statistics

import torch

import vox_serve.tokenizer.snac as snac_mod
from vox_serve.tokenizer.snac import NoiseBlock


def make_codes(bs, device, vocab=4096):
    g = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, vocab, (bs, 4), generator=g).to(device).to(torch.int32),
        torch.randint(0, vocab, (bs, 8), generator=g).to(device).to(torch.int32),
        torch.randint(0, vocab, (bs, 16), generator=g).to(device).to(torch.int32),
    ]


def load_model(dtype=torch.bfloat16, compile_mode="default", disable_noise=True, bs_list=(1, 8, 64)):
    m = snac_mod.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda").to(dtype)
    if disable_noise:
        for mod in m.modules():
            if isinstance(mod, NoiseBlock):
                mod.forward = lambda x: x
    kwargs = {"dynamic": True}
    if compile_mode != "default":
        kwargs["mode"] = compile_mode
    m.decoder = torch.compile(m.decoder, **kwargs)
    m.quantizer.from_codes = torch.compile(m.quantizer.from_codes, **kwargs)
    for bs in bs_list:
        _ = m.decode(make_codes(bs, "cuda"))
    torch.cuda.synchronize()
    return m


def time_graph(model, codes, n_iter=100):
    for _ in range(5):
        _ = model.decode(codes)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = model.decode(codes)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = model.decode(codes)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return out, times


def measure(label, mode, dtype, compose_on=False, depthwise_on=False, bs_list=(1, 8, 64), n_iter=100):
    snac_mod.FUSE_REMOVE_WEIGHT_NORM = True
    snac_mod.FUSE_COMPOSE_QUANTIZER_TABLE = compose_on
    snac_mod.FUSE_TRITON_DEPTHWISE_CONV = depthwise_on
    m = load_model(dtype=dtype, compile_mode=mode, bs_list=bs_list)
    row = {}
    for bs in bs_list:
        _, t = time_graph(m, make_codes(bs, "cuda"), n_iter)
        row[bs] = statistics.mean(t)
    del m
    torch.cuda.empty_cache()
    return label, row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=100)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    rows = []
    rows.append(measure("baseline: compile+graph+wnbake", "default", torch.bfloat16, n_iter=args.n_iter))
    rows.append(measure("+ max-autotune-no-cudagraphs",   "max-autotune-no-cudagraphs", torch.bfloat16, n_iter=args.n_iter))
    rows.append(measure("+ fp16 instead of bf16",         "default", torch.float16, n_iter=args.n_iter))
    rows.append(measure("+ composed quantizer tables",    "default", torch.bfloat16, compose_on=True, n_iter=args.n_iter))
    rows.append(measure("stacked: fp16 + composed",       "default", torch.float16, compose_on=True, n_iter=args.n_iter))
    rows.append(measure("+ triton depthwise (bf16)",      "default", torch.bfloat16, compose_on=True, depthwise_on=True, n_iter=args.n_iter))
    rows.append(measure("+ triton depthwise (fp16)",      "default", torch.float16, compose_on=True, depthwise_on=True, n_iter=args.n_iter))

    print(f"\n{'config':>40} | {'bs=1':>10} | {'bs=8':>10} | {'bs=64':>10}")
    print("-" * 80)
    base = rows[0][1]
    for label, row in rows:
        cells = " | ".join(f"{row[bs]:>6.3f}ms ({base[bs]/row[bs]:.2f}x)" for bs in (1, 8, 64))
        print(f"{label:>40} | {cells}")
