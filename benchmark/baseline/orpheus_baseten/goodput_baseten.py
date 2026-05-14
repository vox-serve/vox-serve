#!/usr/bin/env python3
"""
Goodput benchmark for the Baseten Orpheus baseline (locally hosted by server.py).

Mirrors vox-serve/benchmark/goodput.py byte-for-byte on:
  - Arrival distribution (Gamma(shape=burstiness, scale=1/(burstiness*rate)))
  - Streaming-viability formula (per-chunk and all-chunks variants)
  - TTFA percentiles (p50/p90/p95/p99)
  - Dataset loader (efficient-speech/tts-serving-benchmark, libritts by default)

Three transport-level deltas vs. goodput.py (and only these):
  1. POST /predict with JSON body (vs. /generate form-data).
  2. First streamed chunk is raw PCM (no WAV header) — TTFA fires on chunk #1.
  3. PCM duration formula = len(bytes) / (24000 * 2) for mono 16-bit @ 24kHz.

Adds (since paper resubmission wants the whole goodput picture for Orpheus):
  - End-to-end latency p50/p90/p95/p99
  - Audio duration mean
  - Real-time factor (RTF) = sum(audio) / sum(latency)
  - Chunks per request mean

Usage:
  python goodput_baseten.py \\
    --host localhost --port 8000 \\
    --rate 0.5 1.0 2.0 4.0 6.0 8.0 10.0 \\
    --duration 60 \\
    --data-source libritts \\
    --output-dir results_baseten
"""

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
import numpy as np
from datasets import DatasetDict, load_dataset

SAMPLE_RATE_HZ = 24000
BYTES_PER_SAMPLE = 2  # int16
CHANNELS = 1


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    ttfa: Optional[float] = None
    end_time: Optional[float] = None
    latency: Optional[float] = None
    audio_duration: Optional[float] = None
    rtf: Optional[float] = None
    streaming_viability: Optional[float] = None
    streaming_viability_per_chunk: Optional[float] = None
    num_chunks: int = 0
    success: bool = False
    error_message: Optional[str] = None
    chunk_arrival_times: List[float] = field(default_factory=list)
    chunk_durations: List[float] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    rate: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    # TTFA
    ttfa_mean: float = 0.0
    ttfa_p50: float = 0.0
    ttfa_p90: float = 0.0
    ttfa_p95: float = 0.0
    ttfa_p99: float = 0.0
    ttfa_min: float = 0.0
    ttfa_max: float = 0.0
    # E2E latency
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    # Streaming viability
    streaming_viability_mean: float = 0.0
    streaming_viability_per_chunk_mean: float = 0.0
    streaming_viability_all_chunks_mean: float = 0.0
    # Audio + throughput
    audio_duration_mean: float = 0.0
    rtf_mean: float = 0.0
    chunks_per_request_mean: float = 0.0


def pcm_duration_bytes(nbytes: int) -> float:
    return nbytes / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHANNELS)


def _percentile(sorted_values: List[float], p: int) -> float:
    if not sorted_values:
        return 0.0
    idx = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    w = idx - lo
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


def calculate_streaming_viability(m: RequestMetrics):
    """Identical formula to vox-serve goodput.py:186-215."""
    if len(m.chunk_arrival_times) < 2 or len(m.chunk_durations) < 2:
        return None
    satisfied = 0
    total = 0
    for i in range(1, min(len(m.chunk_arrival_times), len(m.chunk_durations))):
        cumulative_audio = sum(m.chunk_durations[:i])
        latency_to_chunk = m.chunk_arrival_times[i] - m.chunk_arrival_times[0]
        if cumulative_audio > latency_to_chunk:
            satisfied += 1
        total += 1
    if total == 0:
        return None
    per_chunk = (satisfied / total) * 100.0
    all_chunks = (satisfied == total) * 100.0
    return per_chunk, all_chunks


class BasetenOrpheusBenchmark:
    def __init__(
        self,
        host: str,
        port: int,
        voice: str = "tara",
        max_tokens: int = 4096,
        save_audio: bool = False,
        output_dir: str = "results_baseten",
        data_source: str = "libritts",
    ):
        self.base_url = f"http://{host}:{port}"
        self.voice = voice
        self.max_tokens = max_tokens
        self.save_audio = save_audio
        self.output_dir = output_dir
        self.data_source = data_source
        self.dataset = None
        self.dataset_size = 0
        self.text_column = None
        self.metrics: List[RequestMetrics] = []

        os.makedirs(self.output_dir, exist_ok=True)
        if data_source != "fixed":
            self._load_dataset(data_source)

        self.sample_texts = ["Hello world! How are you?"]

    def _load_dataset(self, data_source: str):
        repo_id = "efficient-speech/tts-serving-benchmark"
        if data_source == "hifi":
            ds = load_dataset(repo_id, data_dir="hifi-tts_clean")
            self.dataset = ds["test"]
            self.text_column = "text"
        elif data_source == "libritts":
            ds = load_dataset(repo_id, data_dir="libritts_clean")
            self.dataset = ds["test"]
            self.text_column = "text_normalized"
        elif data_source == "lj-speech":
            ds = load_dataset(repo_id, data_dir="lj-speech_default")
            self.dataset = ds["train"]
            self.text_column = "normalized_text"
        else:
            raise ValueError(f"Unsupported data source for Orpheus: {data_source}")
        self.dataset_size = len(self.dataset)
        print(f"Loaded '{data_source}': {self.dataset_size} samples, column '{self.text_column}'")

    def _next_text(self) -> str:
        if self.data_source == "fixed":
            return random.choice(self.sample_texts)
        i = random.randint(0, self.dataset_size - 1)
        return self.dataset[i][self.text_column]

    async def _one_request(self, session: aiohttp.ClientSession, req_id: str) -> RequestMetrics:
        m = RequestMetrics(request_id=req_id, start_time=time.time())
        try:
            text = self._next_text()
            payload = {
                "prompt": text,
                "voice": self.voice,
                "max_tokens": self.max_tokens,
                "stop_token_ids": [128258, 128009],
                "request_id": req_id,
            }
            timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
            async with session.post(f"{self.base_url}/predict", json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    m.error_message = f"HTTP {resp.status}: {await resp.text()}"
                    m.end_time = time.time()
                    return m

                audio_bytes = bytearray()
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        break
                    now = time.time()
                    m.num_chunks += 1
                    # Baseten streams raw PCM from chunk #1 (no WAV header preamble).
                    dur = pcm_duration_bytes(len(chunk))
                    if m.ttfa is None:
                        m.ttfa = now - m.start_time
                    m.chunk_arrival_times.append(now)
                    m.chunk_durations.append(dur)
                    audio_bytes.extend(chunk)

                m.end_time = time.time()
                m.latency = m.end_time - m.start_time
                m.audio_duration = pcm_duration_bytes(len(audio_bytes))
                if m.latency and m.latency > 0:
                    m.rtf = m.audio_duration / m.latency

                viability = calculate_streaming_viability(m)
                if viability is not None:
                    m.streaming_viability_per_chunk, m.streaming_viability = viability

                if self.save_audio and len(audio_bytes) > 0:
                    fp = os.path.join(self.output_dir, f"{req_id}.pcm")
                    with open(fp, "wb") as f:
                        f.write(audio_bytes)

                m.success = True
        except asyncio.TimeoutError:
            m.error_message = "Request timeout"
        except Exception as e:
            m.error_message = repr(e)
        finally:
            if not m.end_time:
                m.end_time = time.time()
                m.latency = m.end_time - m.start_time
        return m

    async def run_at_rate(self, rate: float, duration: float, burstiness: float = 1.0) -> BenchmarkResults:
        print(f"[bench] rate={rate} req/s, duration={duration}s, burstiness={burstiness}")
        self.metrics = []
        end_time = time.time() + duration
        next_t = time.time()
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            count = 0
            while next_t < end_time:
                cur = time.time()
                if next_t > cur:
                    await asyncio.sleep(next_t - cur)
                count += 1
                req_id = f"req_{count:06d}_{uuid.uuid4().hex[:6]}"
                tasks.append(asyncio.create_task(self._one_request(session, req_id)))
                if rate > 0:
                    inter = np.random.gamma(burstiness, 1.0 / (burstiness * rate))
                else:
                    inter = float("inf")
                next_t += inter
            print(f"[bench] scheduled {len(tasks)} requests; awaiting completion...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, RequestMetrics):
                    self.metrics.append(r)
                    status = "OK" if r.success else "ERR"
                    ttfa_s = f"{r.ttfa:.3f}s" if r.ttfa else "N/A"
                    sv = (f"{r.streaming_viability:.1f}%"
                          if r.streaming_viability is not None else "N/A")
                    print(f"  [{status}] {r.request_id}: TTFA={ttfa_s}, viability={sv}, chunks={r.num_chunks}")
        return self._aggregate(rate)

    def _aggregate(self, rate: float) -> BenchmarkResults:
        out = BenchmarkResults(rate=rate)
        out.total_requests = len(self.metrics)
        ok = [m for m in self.metrics if m.success]
        out.successful_requests = len(ok)
        out.failed_requests = out.total_requests - out.successful_requests
        if not ok:
            return out

        ttfa = sorted([m.ttfa for m in ok if m.ttfa is not None])
        lat = sorted([m.latency for m in ok if m.latency is not None])
        sv_pc = [m.streaming_viability_per_chunk for m in ok if m.streaming_viability_per_chunk is not None]
        sv_all = [m.streaming_viability for m in ok if m.streaming_viability is not None]
        adur = [m.audio_duration for m in ok if m.audio_duration is not None]
        rtfs = [m.rtf for m in ok if m.rtf is not None]
        nch = [m.num_chunks for m in ok]

        if ttfa:
            out.ttfa_mean = statistics.mean(ttfa)
            out.ttfa_p50 = _percentile(ttfa, 50)
            out.ttfa_p90 = _percentile(ttfa, 90)
            out.ttfa_p95 = _percentile(ttfa, 95)
            out.ttfa_p99 = _percentile(ttfa, 99)
            out.ttfa_min = min(ttfa)
            out.ttfa_max = max(ttfa)
        if lat:
            out.latency_mean = statistics.mean(lat)
            out.latency_p50 = _percentile(lat, 50)
            out.latency_p90 = _percentile(lat, 90)
            out.latency_p95 = _percentile(lat, 95)
            out.latency_p99 = _percentile(lat, 99)
        if sv_pc:
            out.streaming_viability_per_chunk_mean = statistics.mean(sv_pc)
        if sv_all:
            out.streaming_viability_all_chunks_mean = statistics.mean(sv_all)
            out.streaming_viability_mean = statistics.mean(sv_all)
        if adur:
            out.audio_duration_mean = statistics.mean(adur)
        if rtfs:
            out.rtf_mean = statistics.mean(rtfs)
        if nch:
            out.chunks_per_request_mean = statistics.mean(nch)
        return out

    def dump_per_request_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "request_id", "success", "ttfa_s", "latency_s",
                "audio_duration_s", "rtf", "streaming_viability_per_chunk",
                "streaming_viability_all_chunks", "num_chunks", "error",
            ])
            for m in self.metrics:
                w.writerow([
                    m.request_id, int(m.success),
                    f"{m.ttfa:.4f}" if m.ttfa is not None else "",
                    f"{m.latency:.4f}" if m.latency is not None else "",
                    f"{m.audio_duration:.4f}" if m.audio_duration is not None else "",
                    f"{m.rtf:.4f}" if m.rtf is not None else "",
                    f"{m.streaming_viability_per_chunk:.2f}" if m.streaming_viability_per_chunk is not None else "",
                    f"{m.streaming_viability:.2f}" if m.streaming_viability is not None else "",
                    m.num_chunks, m.error_message or "",
                ])


def print_table(all_results: List[BenchmarkResults]) -> None:
    print("\n" + "=" * 80)
    print("BASETEN ORPHEUS — GOODPUT COMPARISON")
    print("=" * 80)
    rates = [r.rate for r in all_results]
    print("\n## Requests\n")
    print("| Metric | " + " | ".join(f"{r:.1f} req/s" for r in rates) + " |")
    print("|--------|" + "|".join("-" * 12 for _ in rates) + "|")
    print("| Total | " + " | ".join(str(r.total_requests) for r in all_results) + " |")
    print("| OK | " + " | ".join(str(r.successful_requests) for r in all_results) + " |")
    print("| Failed | " + " | ".join(str(r.failed_requests) for r in all_results) + " |")
    for name, attrs in [
        ("TTFA (s)", ["ttfa_mean", "ttfa_p50", "ttfa_p90", "ttfa_p95", "ttfa_p99"]),
        ("E2E latency (s)", ["latency_mean", "latency_p50", "latency_p90", "latency_p95", "latency_p99"]),
    ]:
        print(f"\n## {name}\n")
        print("| Stat | " + " | ".join(f"{r:.1f} req/s" for r in rates) + " |")
        print("|------|" + "|".join("-" * 12 for _ in rates) + "|")
        for a in attrs:
            label = a.split("_", 1)[1].upper().replace("_", " ")
            print(f"| {label} | " + " | ".join(f"{getattr(r, a):.3f}" for r in all_results) + " |")
    print("\n## Streaming viability (%)\n")
    print("| Variant | " + " | ".join(f"{r:.1f} req/s" for r in rates) + " |")
    print("|---------|" + "|".join("-" * 12 for _ in rates) + "|")
    print("| Per-chunk mean | " + " | ".join(f"{r.streaming_viability_per_chunk_mean:.1f}" for r in all_results) + " |")
    print("| All-chunks mean | " + " | ".join(f"{r.streaming_viability_all_chunks_mean:.1f}" for r in all_results) + " |")
    print("\n## Audio + throughput\n")
    print("| Stat | " + " | ".join(f"{r:.1f} req/s" for r in rates) + " |")
    print("|------|" + "|".join("-" * 12 for _ in rates) + "|")
    print("| Mean audio (s) | " + " | ".join(f"{r.audio_duration_mean:.2f}" for r in all_results) + " |")
    print("| Mean RTF | " + " | ".join(f"{r.rtf_mean:.2f}" for r in all_results) + " |")
    print("| Chunks/req mean | " + " | ".join(f"{r.chunks_per_request_mean:.1f}" for r in all_results) + " |")


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--voice", default="tara")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--rate", type=float, nargs="+", default=[1.0])
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--burstiness", type=float, default=1.0)
    ap.add_argument("--data-source", choices=["fixed", "hifi", "libritts", "lj-speech"], default="libritts")
    ap.add_argument("--output-dir", default="results_baseten")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=3,
                    help="Number of warmup requests (results not recorded)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bench = BasetenOrpheusBenchmark(
        host=args.host, port=args.port, voice=args.voice,
        max_tokens=args.max_tokens, save_audio=args.save_audio,
        output_dir=args.output_dir, data_source=args.data_source,
    )

    # Warmup (no metrics kept).
    if args.warmup > 0:
        print(f"[bench] warmup: {args.warmup} requests (sequential)")
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(args.warmup):
                _ = await bench._one_request(session, f"warmup_{i}")

    all_results: List[BenchmarkResults] = []
    for rate in args.rate:
        print(f"\n{'=' * 80}\nRate {rate} req/s\n{'=' * 80}")
        result = await bench.run_at_rate(rate, args.duration, args.burstiness)
        all_results.append(result)
        # Per-rate per-request CSV
        bench.dump_per_request_csv(
            os.path.join(args.output_dir, f"per_request_rate{rate}.csv")
        )

    print_table(all_results)
    # Aggregated JSON
    out_json = {"system": "baseten_orpheus_local", "results": [asdict(r) for r in all_results]}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[bench] wrote {args.output_dir}/summary.json + per-request CSVs")


if __name__ == "__main__":
    asyncio.run(main_async())
