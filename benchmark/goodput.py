#!/usr/bin/env python3
"""
Benchmarking goodput in online serving scenario.

Measures performance metrics:
- TTFA (Time to First Audio): Latency from request start to first audio chunk

Usage:
    python goodput.py --host localhost --port 8000 --rate 10 --duration 60
"""

import argparse
import asyncio
import io
import os
import random
import statistics
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import numpy as np

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    start_time: float
    ttfa: Optional[float] = None  # Time to first audio
    end_time: Optional[float] = None
    audio_duration: Optional[float] = None  # Duration of generated audio
    streaming_viability: Optional[float] = None  # Streaming viability percentage
    streaming_viability_per_chunk: Optional[float] = None  # Per-chunk streaming viability percentage
    success: bool = False
    error_message: Optional[str] = None

    # Chunk-level timing data
    chunk_arrival_times: List[float] = None
    chunk_durations: List[float] = None

    def __post_init__(self):
        if self.chunk_arrival_times is None:
            self.chunk_arrival_times = []
        if self.chunk_durations is None:
            self.chunk_durations = []


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    rate: float = 0.0  # Request rate for this benchmark
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # TTFA metrics (seconds)
    ttfa_mean: float = 0.0
    ttfa_p50: float = 0.0
    ttfa_p90: float = 0.0
    ttfa_p95: float = 0.0
    ttfa_p99: float = 0.0
    ttfa_min: float = 0.0
    ttfa_max: float = 0.0

    # Streaming viability metrics (percentage)
    streaming_viability_mean: float = 0.0
    streaming_viability_per_chunk_mean: float = 0.0
    streaming_viability_all_chunks_mean: float = 0.0


class BenchmarkClient:
    """Client for benchmarking vox-serve TTS server."""

    def __init__(self, host: str, port: int, save_audio: bool = False):
        self.base_url = f"http://{host}:{port}"
        self.save_audio = save_audio
        self.output_dir = "benchmark_output"
        self.metrics: List[RequestMetrics] = []

        if self.save_audio:
            os.makedirs(self.output_dir, exist_ok=True)

        # Sample texts for generation
        self.sample_texts = [
            "Hello world!",
            # "Hello world, this is a test message for benchmarking the performance of the text-to-speech server. "
            # "We want to measure the time it takes to receive the first audio and the overall latency.",
        ]

    def generate_random_text(self, min_words: int = 5, max_words: int = 20) -> str:
        """Generate random text for testing."""
        return random.choice(self.sample_texts)

    def get_audio_duration(self, audio_data: bytes) -> float:
        """Calculate audio duration from WAV data."""
        try:
            with io.BytesIO(audio_data) as audio_io:
                with wave.open(audio_io, "rb") as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    return duration
        except Exception:
            # Fallback estimation: assume 16kHz sample rate
            # WAV header is typically 44 bytes
            audio_samples = len(audio_data) - 44
            sample_rate = 24000
            bytes_per_sample = 2  # 16-bit audio
            return audio_samples / (sample_rate * bytes_per_sample)

    def pcm_duration_bytes(self, nbytes: int) -> float:
        # raw PCM duration in seconds
        sample_rate = 24000
        bytes_per_sample = 2  # 16-bit audio
        channels = 1  # Mono audio
        return nbytes / (sample_rate * bytes_per_sample * channels)

    def calculate_streaming_viability(self, metrics: RequestMetrics) -> Optional[tuple[float, float]]:
        """Calculate streaming viability (percentage of chunks satisfying real-time requirement)."""
        if len(metrics.chunk_arrival_times) < 2 or len(metrics.chunk_durations) < 2:
            return None

        real_time_satisfied = 0
        total_chunks = 0

        # Start from the second chunk (i > 1, or index 1)
        for i in range(1, min(len(metrics.chunk_arrival_times), len(metrics.chunk_durations))):
            # Calculate cumulative audio duration from 1st to i-th chunk
            cumulative_audio_duration = sum(metrics.chunk_durations[:i])

            # Calculate latency from arrival of 1st chunk to i-th chunk
            latency_to_chunk = metrics.chunk_arrival_times[i] - metrics.chunk_arrival_times[0]

            # Check if cumulative audio duration is longer than latency
            if cumulative_audio_duration > latency_to_chunk:
                real_time_satisfied += 1

            total_chunks += 1

        if total_chunks == 0:
            return None

        # Calculate both metrics
        per_chunk_viability = (real_time_satisfied / total_chunks) * 100.0
        all_chunks_viability = (real_time_satisfied == total_chunks) * 100.0

        return per_chunk_viability, all_chunks_viability

    async def make_request(self, session: aiohttp.ClientSession, request_id: str) -> RequestMetrics:
        """Make a single request and measure metrics."""
        metrics = RequestMetrics(request_id=request_id, start_time=time.time())

        try:
            text = self.generate_random_text()

            # Prepare form data
            form_data = aiohttp.FormData()
            form_data.add_field("text", text)
            form_data.add_field("streaming", "true")

            # print(f"new request {request_id=}")
            # Make streaming request
            async with session.post(
                f"{self.base_url}/generate", data=form_data, timeout=aiohttp.ClientTimeout(total=None, sock_read=30)
            ) as response:
                if response.status != 200:
                    metrics.error_message = f"HTTP {response.status}: {await response.text()}"
                    return metrics

                # Read streaming response
                audio_chunks = []
                chunk_count = 0

                # Use iter_any() instead of iter_chunked() to properly detect end of stream
                async for chunk in response.content.iter_any():
                    if not chunk:  # Empty chunk indicates end of stream
                        break

                    current_time = time.time()
                    chunk_count += 1

                    if chunk_count == 1:
                        # first chunk is the WAV header; do not use it for TTFA or durations
                        header = chunk
                        audio_chunks.append(chunk)
                        continue

                    # Calculate chunk duration first to detect header-only chunks (no audio)
                    # chunk_duration = self.get_audio_duration(chunk)
                    chunk_duration = self.pcm_duration_bytes(len(chunk))

                    # Set TTFA when the first audio chunk arrives
                    if metrics.ttfa is None:
                        metrics.ttfa = current_time - metrics.start_time

                    metrics.chunk_arrival_times.append(current_time)
                    metrics.chunk_durations.append(chunk_duration)

                    audio_chunks.append(chunk)

                # Calculate final metrics
                metrics.end_time = time.time()

                # Combine audio chunks and calculate duration
                full_audio = b"".join(audio_chunks)
                metrics.audio_duration = self.get_audio_duration(full_audio)

                # Save audio if enabled
                if self.save_audio and full_audio:
                    output_path = os.path.join(self.output_dir, f"{request_id}.wav")
                    with open(output_path, "wb") as f:
                        f.write(full_audio)

                # Calculate streaming viability (real-time requirement satisfaction)
                viability_result = self.calculate_streaming_viability(metrics)
                if viability_result:
                    metrics.streaming_viability_per_chunk, metrics.streaming_viability = viability_result
                else:
                    metrics.streaming_viability_per_chunk = None
                    metrics.streaming_viability = None

                metrics.success = True

        except asyncio.TimeoutError:
            metrics.error_message = "Request timeout"
        except Exception as e:
            metrics.error_message = str(e)
        finally:
            if not metrics.end_time:
                metrics.end_time = time.time()

        return metrics

    async def run_benchmark(self, rate: float, duration: float) -> BenchmarkResults:
        """Run benchmark with specified request rate for given duration."""
        print(f"Starting benchmark: {rate} req/s for {duration}s")
        print(f"Target server: {self.base_url}")
        print("=" * 60)

        # Setup Poisson arrival process
        # For Poisson process, inter-arrival times follow exponential distribution
        # with parameter lambda = rate (average rate)
        end_time = time.time() + duration
        request_count = 0
        next_request_time = time.time()

        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []

            # Schedule requests using Poisson process
            while next_request_time < end_time:
                # Wait until it's time for the next request
                current_time = time.time()
                if next_request_time > current_time:
                    await asyncio.sleep(next_request_time - current_time)

                request_count += 1
                request_id = f"req_{request_count:06d}"

                # Create request task
                task = asyncio.create_task(self.make_request(session, request_id))
                tasks.append(task)

                # Generate next inter-arrival time using exponential distribution
                # Mean inter-arrival time = 1/rate
                inter_arrival_time = np.random.exponential(1.0 / rate) if rate > 0 else float('inf')
                next_request_time += inter_arrival_time

            print(f"Scheduled {len(tasks)} requests. Waiting for completion...")

            # Wait for all requests to complete
            completed_metrics = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and collect metrics
            for result in completed_metrics:
                if isinstance(result, RequestMetrics):
                    self.metrics.append(result)

                    # Print real-time progress
                    status = "✓" if result.success else "✗"
                    ttfa_str = f"{result.ttfa:.3f}s" if result.ttfa else "N/A"
                    streaming_viability_str = (
                        f"{result.streaming_viability:.1f}%"
                        if result.streaming_viability is not None
                        else "N/A"
                    )
                    print(
                        f"{status} {result.request_id}: "
                        f"TTFA={ttfa_str}, "
                        f"Streaming_viability={streaming_viability_str}"
                    )

        return self.calculate_results(rate)

    def calculate_results(self, rate: float) -> BenchmarkResults:
        """Calculate aggregated benchmark results."""
        results = BenchmarkResults(rate=rate)

        if not self.metrics:
            return results

        results.total_requests = len(self.metrics)
        successful_metrics = [m for m in self.metrics if m.success]
        results.successful_requests = len(successful_metrics)
        results.failed_requests = results.total_requests - results.successful_requests

        if not successful_metrics:
            return results

        # Extract metrics for successful requests
        ttfa_values = [m.ttfa for m in successful_metrics if m.ttfa is not None]
        streaming_viability_values = [
            m.streaming_viability for m in successful_metrics
            if m.streaming_viability is not None
        ]
        streaming_viability_per_chunk_values = [
            m.streaming_viability_per_chunk for m in successful_metrics
            if m.streaming_viability_per_chunk is not None
        ]
        streaming_viability_all_chunks_values = [
            m.streaming_viability for m in successful_metrics
            if m.streaming_viability is not None
        ]

        # Calculate TTFA statistics
        if ttfa_values:
            ttfa_sorted = sorted(ttfa_values)
            results.ttfa_mean = statistics.mean(ttfa_values)
            results.ttfa_p50 = self._percentile(ttfa_sorted, 50)
            results.ttfa_p90 = self._percentile(ttfa_sorted, 90)
            results.ttfa_p95 = self._percentile(ttfa_sorted, 95)
            results.ttfa_p99 = self._percentile(ttfa_sorted, 99)
            results.ttfa_min = min(ttfa_values)
            results.ttfa_max = max(ttfa_values)

        # Calculate streaming viability statistics
        if streaming_viability_values:
            results.streaming_viability_mean = statistics.mean(streaming_viability_values)
        if streaming_viability_per_chunk_values:
            results.streaming_viability_per_chunk_mean = statistics.mean(streaming_viability_per_chunk_values)
        if streaming_viability_all_chunks_values:
            results.streaming_viability_all_chunks_mean = statistics.mean(streaming_viability_all_chunks_values)


        return results

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight



    def print_comparison_table(self, all_results: List[BenchmarkResults]):
        """Print comparison table for multiple request rates."""
        print("\n" + "="*80)
        print("MULTIPLE RATE BENCHMARK COMPARISON")
        print("="*80)

        if not all_results:
            print("No results to display.")
            return

        # Extract rates for column headers
        rates = [result.rate for result in all_results]

        # Request Summary Table
        print("\n## Request Summary\n")
        print("| Metric | " + " | ".join([f"{rate:.1f} req/s" for rate in rates]) + " |")
        print("|--------|" + "|".join(["-"*12 for _ in rates]) + "|")

        # Total requests row
        total_row = "| Total | " + " | ".join([str(result.total_requests) for result in all_results]) + " |"
        print(total_row)

        # Successful requests row
        success_row = "| Successful | " + " | ".join([str(result.successful_requests) for result in all_results]) + " |"
        print(success_row)

        # Failed requests row
        failed_row = "| Failed | " + " | ".join([str(result.failed_requests) for result in all_results]) + " |"
        print(failed_row)

        # Success rate row
        success_rates = []
        for result in all_results:
            rate_pct = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
            success_rates.append(f"{rate_pct:.1f}%")
        success_rate_row = "| Success Rate | " + " | ".join(success_rates) + " |"
        print(success_rate_row)

        # TTFA Metrics Table
        print("\n## Time to First Audio (TTFA) - seconds\n")
        print("| Statistic | " + " | ".join([f"{rate:.1f} req/s" for rate in rates]) + " |")
        print("|-----------|" + "|".join(["-"*12 for _ in rates]) + "|")

        # Mean TTFA
        mean_ttfa_row = "| Mean | " + " | ".join([f"{result.ttfa_mean:.3f}" for result in all_results]) + " |"
        print(mean_ttfa_row)

        # P50 TTFA
        p50_ttfa_row = "| P50 | " + " | ".join([f"{result.ttfa_p50:.3f}" for result in all_results]) + " |"
        print(p50_ttfa_row)

        # P90 TTFA
        p90_ttfa_row = "| P90 | " + " | ".join([f"{result.ttfa_p90:.3f}" for result in all_results]) + " |"
        print(p90_ttfa_row)

        # P95 TTFA
        p95_ttfa_row = "| P95 | " + " | ".join([f"{result.ttfa_p95:.3f}" for result in all_results]) + " |"
        print(p95_ttfa_row)

        # P99 TTFA
        p99_ttfa_row = "| P99 | " + " | ".join([f"{result.ttfa_p99:.3f}" for result in all_results]) + " |"
        print(p99_ttfa_row)

        # Min TTFA
        min_ttfa_row = "| Min | " + " | ".join([f"{result.ttfa_min:.3f}" for result in all_results]) + " |"
        print(min_ttfa_row)

        # Max TTFA
        max_ttfa_row = "| Max | " + " | ".join([f"{result.ttfa_max:.3f}" for result in all_results]) + " |"
        print(max_ttfa_row)

        # Streaming Viability Table (Per-Chunk Metric)
        print("\n## Streaming Viability (Per-Chunk Real-time Requirement) - percentage\n")
        print("| Statistic | " + " | ".join([f"{rate:.1f} req/s" for rate in rates]) + " |")
        print("|-----------|" + "|".join(["-"*12 for _ in rates]) + "|")

        # Mean streaming viability (per-chunk)
        streaming_per_chunk_row = "| Mean | " + " | ".join([
            f"{result.streaming_viability_per_chunk_mean:.1f}" for result in all_results]
        ) + " |"
        print(streaming_per_chunk_row)

        # Streaming Viability Table (All-Chunks Metric)
        print("\n## Streaming Viability (All-Chunks Real-time Requirement) - percentage\n")
        print("| Statistic | " + " | ".join([f"{rate:.1f} req/s" for rate in rates]) + " |")
        print("|-----------|" + "|".join(["-"*12 for _ in rates]) + "|")

        # Mean streaming viability (all-chunks)
        streaming_all_chunks_row = "| Mean | " + " | ".join(
            [f"{result.streaming_viability_all_chunks_mean:.1f}" for result in all_results]
        ) + " |"
        print(streaming_all_chunks_row)
        print()


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vox-serve TTS server")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--rate", type=float, nargs='+', default=[1.0],
                       help="Request rate(s) in req/s (single value or list, default: [1.0])")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration (seconds, default: 10.0)")
    parser.add_argument("--save-audio", action="store_true", help="Save generated audio files")

    args = parser.parse_args()

    # Validate arguments
    for rate in args.rate:
        if rate <= 0:
            print(f"Error: Request rate must be positive, got {rate}")
            return 1

    if args.duration <= 0:
        print("Error: Duration must be positive")
        return 1

    # Create and run benchmark
    client = BenchmarkClient(args.host, args.port, args.save_audio)

    # Always use multiple benchmark approach
    print(f"Running benchmarks at rates: {args.rate}")
    all_results = []

    for rate in args.rate:
        print(f"\n{'='*80}")
        print(f"Running benchmark at {rate} req/s")
        print(f"{'='*80}")

        # Clear previous metrics
        client.metrics = []

        # Run benchmark for this rate
        results = await client.run_benchmark(rate, args.duration)
        all_results.append(results)

    client.print_comparison_table(all_results)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
