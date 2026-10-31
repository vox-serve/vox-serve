"""Summarize a v2 runtime trace (rust/src/engine.rs tracer): per-stage GPU durations, LM step time,
how much detok work overlapped LM work, and the GPU-busy union.

python rust/scripts/v2_trace_report.py rust/logs/<run>/trace_dir/trace_0.jsonl [--skip-ms 500]
"""

import argparse
import json
import statistics
from collections import defaultdict


def load(path):
    gpu, notes = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (gpu if "lane" in r else notes).append(r)
    return gpu, notes


def overlap(a, b):
    """Total length of the intersection of two sorted interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def union(iv):
    iv = sorted(iv)
    tot, cur = 0.0, None
    for lo, hi in iv:
        if cur is None or lo > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [lo, hi]
        else:
            cur[1] = max(cur[1], hi)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def summarize(path, skip_ms=0.0):
    gpu, notes = load(path)
    gpu = [g for g in gpu if g["t0"] >= skip_ms]
    lm = sorted((g["t0"], g["t1"]) for g in gpu if g["stage"].startswith("lm_"))
    dt = sorted((g["t0"], g["t1"]) for g in gpu if g["stage"] == "detok")
    by = defaultdict(list)
    for g in gpu:
        by[(g["stage"], tuple(g["key"]))].append(g["t1"] - g["t0"])
    # detok batches that overlapped LM work vs ran alone
    dt_over, dt_alone = [], []
    for g in gpu:
        if g["stage"] != "detok":
            continue
        o = overlap([(g["t0"], g["t1"])], lm)
        (dt_over if o > 0.5 * (g["t1"] - g["t0"]) else dt_alone).append((g["key"][0], g["t1"] - g["t0"]))
    span = (max(g["t1"] for g in gpu) - min(g["t0"] for g in gpu)) if gpu else 0.0
    # Straggler-robust throughput: time from the first launch until 90% of all codec chunks are decoded
    # (the last request running alone no longer dominates the bench).
    chunks = sorted((g["t1"], g["n"]) for g in gpu if g["stage"].startswith("detok"))
    t90 = None
    if chunks:
        total, cum = sum(n for _, n in chunks), 0
        for t1, n in chunks:
            cum += n
            if cum >= 0.9 * total:
                t90 = t1 - min(g["t0"] for g in gpu)
                break
    out = {
        "span_ms": span,
        "chunks_t90_ms": t90,
        "lm_busy_ms": union(lm),
        "detok_busy_ms": union(dt),
        "gpu_busy_ms": union(lm + dt),
        "overlap_ms": overlap(union_list(lm), union_list(dt)),
        "stages": {
            f"{k[0]}{list(k[1])}": {"n": len(v), "median_ms": statistics.median(v), "mean_ms": statistics.fmean(v)}
            for k, v in sorted(by.items())
        },
        "detok_overlapped": summarize_pairs(dt_over),
        "detok_alone": summarize_pairs(dt_alone),
        "notes": len(notes),
    }
    steps = [n for n in notes if n.get("ev") == "lm"]
    if steps:
        out["lm_rows_mean"] = statistics.fmean(n["rows"] for n in steps)
        out["modes"] = dict(
            sorted(
                defaultdict(
                    int, {m: sum(1 for n in steps if n["mode"] == m) for m in {n["mode"] for n in steps}}
                ).items()
            )
        )
    dts = [n for n in notes if n.get("ev") == "detok"]
    if dts:
        out["detok_batch_mean"] = statistics.fmean(n["n"] for n in dts)
        out["detok_gates"] = {g: sum(1 for n in dts if n["gate"] == g) for g in {n["gate"] for n in dts}}
    return out


def union_list(iv):
    iv = sorted(iv)
    res = []
    for lo, hi in iv:
        if res and lo <= res[-1][1]:
            res[-1][1] = max(res[-1][1], hi)
        else:
            res.append([lo, hi])
    return res


def summarize_pairs(pairs):
    d = defaultdict(list)
    for b, ms in pairs:
        d[b].append(ms)
    return {b: {"n": len(v), "median_ms": statistics.median(v)} for b, v in sorted(d.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip-ms", type=float, default=0.0)
    a = ap.parse_args()
    print(json.dumps(summarize(a.trace, a.skip_ms), indent=1))
