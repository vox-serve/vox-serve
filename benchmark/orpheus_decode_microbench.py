"""Microbenchmark the Orpheus LLM decode path (CUDA-graph replay) at varying batch sizes.

Loads the same CudaGraphWorker the server uses, then times the captured decode graphs.
Also runs a torch.profiler trace at one batch size to identify hotspots.

Usage:
    python benchmark/orpheus_decode_microbench.py
    python benchmark/orpheus_decode_microbench.py --profile-bs 64
"""

import argparse
import statistics
import time

import torch

import vox_serve.model.orpheus as orph_mod
from vox_serve.worker.cuda_graph_worker import CudaGraphWorker


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (p / 100.0) * (len(xs) - 1)
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def time_graph(graph, n_iter):
    # Warm replays
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-bs", type=int, default=256)
    p.add_argument("--max-num-pages", type=int, default=2048)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--profile-bs", type=int, default=64,
                   help="Run torch.profiler at this batch size; 0 to skip")
    p.add_argument("--compile", action="store_true",
                   help="After capture, torch.compile the LM forward and re-capture the decode graphs.")
    p.add_argument("--fuse-qkv", type=int, default=1, help="1=on, 0=off (default 1)")
    p.add_argument("--fuse-mlp", type=int, default=1, help="1=on, 0=off (default 1)")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    orph_mod.FUSE_ORPHEUS_QKV = bool(args.fuse_qkv)
    orph_mod.FUSE_ORPHEUS_MLP_GATE_UP = bool(args.fuse_mlp)
    print(f"Toggles: FUSE_ORPHEUS_QKV={orph_mod.FUSE_ORPHEUS_QKV} "
          f"FUSE_ORPHEUS_MLP_GATE_UP={orph_mod.FUSE_ORPHEUS_MLP_GATE_UP}")
    print("Building CudaGraphWorker for orpheus...")
    t0 = time.time()
    worker = CudaGraphWorker(
        model_name="orpheus",
        max_batch_size=args.max_bs,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
    )
    print(f"Worker ready in {time.time() - t0:.1f}s; "
          f"captured decode bs={sorted(worker.cuda_graphs_lm_decode.keys(), reverse=True)}")

    def collect_rows(label):
        rs = []
        for bs in sorted(worker.cuda_graphs_lm_decode.keys()):
            graph = worker.cuda_graphs_lm_decode[bs]
            times = time_graph(graph, args.n_iter)
            rs.append((bs, statistics.mean(times), percentile(times, 50), percentile(times, 95)))
        return label, rs

    sections = [collect_rows("no compile")]

    if args.compile:
        print("\nApplying torch.compile to LM forward and re-capturing decode graphs ...")
        # Dynamo can't trace through FlashInfer's custom C++ ops (no fake-tensor impls).
        # Mark them as opaque so the model can compile around them (graph breaks at
        # flashinfer call sites only).
        import vox_serve.flashinfer_utils as fi
        fi.rms_norm = torch._dynamo.disable(fi.rms_norm)
        fi.apply_rope_pos_ids = torch._dynamo.disable(fi.apply_rope_pos_ids)
        from vox_serve.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
        for cls in (FlashInferDecodeWrapper, FlashInferPrefillWrapper):
            cls.set_kv_cache = torch._dynamo.disable(cls.set_kv_cache)
            cls.run = torch._dynamo.disable(cls.run)
        # NOTE: the model code imports these symbols at module load time, so patch the
        # bindings on the orpheus module too.
        import vox_serve.model.orpheus as orph
        orph.rms_norm = fi.rms_norm
        orph.apply_rope_pos_ids = fi.apply_rope_pos_ids

        t0 = time.time()
        worker.model.model = torch.compile(
            worker.model.model, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
        worker.cuda_graphs_lm_decode.clear()
        worker._initialize_decode_cuda_graphs()
        print(f"Recompile + recapture in {time.time() - t0:.1f}s")
        sections.append(collect_rows("torch.compile"))

    print()
    for label, rows in sections:
        print(f"=== {label} ===")
        print(f"{'bs':>5} | {'mean':>10} | {'p50':>10} | {'p95':>10} | {'per-req μs':>12}")
        print("-" * 70)
        for bs, m, p50, p95 in rows:
            per_tok = (m * 1000) / bs
            print(f"{bs:>5} | {m:>8.3f}ms | {p50:>8.3f}ms | {p95:>8.3f}ms | {per_tok:>10.2f}us")
        print()

    if len(sections) == 2:
        base = {bs: m for bs, m, _, _ in sections[0][1]}
        comp = {bs: m for bs, m, _, _ in sections[1][1]}
        print("=== speedup (compile / no compile) ===")
        print(f"{'bs':>5} | {'no comp':>10} | {'compile':>10} | {'speedup':>8}")
        for bs in sorted(base):
            print(f"{bs:>5} | {base[bs]:>8.3f}ms | {comp[bs]:>8.3f}ms | {base[bs]/comp[bs]:>7.3f}x")

    if args.profile_bs and args.profile_bs in worker.cuda_graphs_lm_decode:
        print(f"\nProfiling bs={args.profile_bs} ...")
        g = worker.cuda_graphs_lm_decode[args.profile_bs]
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(50):
                g.replay()
            torch.cuda.synchronize()
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
