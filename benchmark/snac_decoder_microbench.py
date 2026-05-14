"""Microbenchmark: SNAC decoder (24 kHz) eager vs CUDA-graph replay across batch sizes.

For each batch size we time `model.decode(codes)` over a fixed number of iterations
in three modes:
  - eager:        plain forward passes, torch.cuda.synchronize() around each call
  - eager_warm:   same as eager but after warmup (no first-call CUDA init overhead)
  - cuda_graph:   captured graph replay (what vox-serve's CudaGraphWorker uses)

Usage:
    python benchmark/snac_decoder_microbench.py

The codes mirror Orpheus's detokenization batching: codebook frame counts (4, 8, 16)
for one detokenization step (28 audio tokens -> 8192 samples).
"""

import argparse
import statistics
import time

import torch

from vox_serve.tokenizer.snac import SNAC


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (p / 100.0) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def make_codes(batch_size: int, device: str, vocab: int) -> list[torch.Tensor]:
    # Orpheus's SNAC stride pattern: codebook frame counts per detokenize step.
    g = torch.Generator(device="cpu").manual_seed(0)
    return [
        torch.randint(0, vocab, (batch_size, 4), generator=g, device="cpu").to(device).to(torch.int32),
        torch.randint(0, vocab, (batch_size, 8), generator=g, device="cpu").to(device).to(torch.int32),
        torch.randint(0, vocab, (batch_size, 16), generator=g, device="cpu").to(device).to(torch.int32),
    ]


def time_eager(model, codes, n_iter):
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.decode(codes)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def capture_graph(model, codes):
    # Warm streams first (required by torch.cuda.graph), then capture replay-able graph.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = model.decode(codes)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        output = model.decode(codes)
    return graph, output


def time_graph_replay(graph, n_iter):
    # Warmup replays
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--compile", action="store_true",
                   help="Also time torch.compile (dynamic=True) on decoder + quantizer.from_codes, "
                        "mirroring the reference orpheus-trtllm-local recipe.")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.set_grad_enabled(False)

    print(f"Loading SNAC (24kHz) -> {args.device}, dtype={args.dtype}")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(args.device).to(dtype)
    vocab = model.codebook_size

    if args.compile:
        # Mirror /mnt/storage/keisuke/orpheus-trtllm-local/model.py: compile decoder + quantizer
        # with dynamic=True, then trigger compilation across the batch-size range by running once
        # at each size before timing. We keep the same dtype throughout (the reference forces
        # fp32 on the decoder; we honour --dtype to keep comparisons apples-to-apples).
        print("torch.compile: wrapping decoder + quantizer.from_codes (dynamic=True)")
        model.decoder = torch.compile(model.decoder, dynamic=True)
        compiled_from_codes = torch.compile(model.quantizer.from_codes, dynamic=True)
        # Swap into the model so model.decode() picks up the compiled paths.
        model.quantizer.from_codes = compiled_from_codes

        import time as _t
        t0 = _t.time()
        for bs in args.batch_sizes:
            codes = make_codes(bs, args.device, vocab)
            _ = model.decode(codes)
        torch.cuda.synchronize()
        print(f"torch.compile warmup done in {_t.time() - t0:.1f}s")

    print(f"{'bs':>4} | {'eager mean':>11} | {'eager p50':>10} | {'eager p95':>10} | "
          f"{'graph mean':>11} | {'graph p50':>10} | {'graph p95':>10} | {'speedup':>7}")
    print("-" * 100)

    for bs in args.batch_sizes:
        codes = make_codes(bs, args.device, vocab)

        # 5 untimed warmup iterations to amortize first-call cost.
        for _ in range(5):
            _ = model.decode(codes)
        torch.cuda.synchronize()

        eager = time_eager(model, codes, args.iters)

        try:
            graph, _ = capture_graph(model, codes)
        except RuntimeError as e:
            print(f"{bs:>4} | capture failed: {e}")
            continue
        graph_times = time_graph_replay(graph, args.iters)

        e_mean, e_p50, e_p95 = statistics.mean(eager), percentile(eager, 50), percentile(eager, 95)
        g_mean, g_p50, g_p95 = statistics.mean(graph_times), percentile(graph_times, 50), percentile(graph_times, 95)
        speedup = e_mean / g_mean if g_mean > 0 else float("nan")

        print(f"{bs:>4} | {e_mean:>9.3f}ms | {e_p50:>8.3f}ms | {e_p95:>8.3f}ms | "
              f"{g_mean:>9.3f}ms | {g_p50:>8.3f}ms | {g_p95:>8.3f}ms | {speedup:>6.2f}x")


if __name__ == "__main__":
    main()
