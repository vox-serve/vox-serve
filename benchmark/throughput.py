#!/usr/bin/env python3
"""
Throughput benchmark for vox-serve TTS server.

Sends a specified number of requests simultaneously and measures end-to-end latency.

Usage:
    python throughput.py --host localhost --port 8000 --num-requests 10
"""

import argparse
import asyncio
import io
import os
import random
import struct
import time
import wave
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from datasets import load_dataset


@dataclass
class ThroughputMetrics:
    """Metrics for a single request in throughput benchmark."""

    request_id: str
    start_time: float
    end_time: Optional[float] = None
    latency: Optional[float] = None  # End-to-end latency
    audio_duration: Optional[float] = None  # Duration of generated audio
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class ThroughputResults:
    """Aggregated throughput benchmark results."""

    num_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Timing metrics
    total_time: float = 0.0  # Time from first request start to last request completion

    # Throughput metrics
    throughput_req_per_sec: float = 0.0
    rtf: float = 0.0  # Real Time Factor: (sum of generated audio duration) / (end-to-end latency)


class ThroughputBenchmark:
    """Throughput benchmark client for vox-serve TTS server."""

    def __init__(self, host: str, port: int, save_audio: bool = False, data_source: str = "fixed"):
        self.base_url = f"http://{host}:{port}"
        self.save_audio = save_audio
        self.output_dir = "benchmark_output"
        self.metrics: List[ThroughputMetrics] = []
        self.data_source = data_source
        self.dataset = None
        self.dataset_size = 0
        self.text_column = None

        if self.save_audio:
            os.makedirs(self.output_dir, exist_ok=True)

        # Sample texts for generation (used when data_source is "fixed")
        self.sample_texts = [
            "Hello world!",
            "This is a test message for benchmarking throughput.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing the performance of the text-to-speech server.",
            "Measuring end-to-end latency and throughput metrics.",
        ]

        # Load dataset if specified
        if data_source != "fixed":
            self._load_dataset(data_source)

    def _load_dataset(self, data_source: str):
        """Load dataset based on data source specification."""
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
            self.dataset = ds["train"]  # Only train split
            self.text_column = "normalized_text"
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        self.dataset_size = len(self.dataset)
        print(f"Loaded dataset '{data_source}': {self.dataset_size} samples, column '{self.text_column}'")

    def get_text(self, index: int) -> str:
        """Get text for a given index."""
        if self.data_source == "fixed":
            return self.sample_texts[index % len(self.sample_texts)]
        else:
            return self.dataset[index % self.dataset_size][self.text_column]

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
            # Fallback estimation: assume 24kHz sample rate
            # WAV header is typically 44 bytes
            audio_samples = len(audio_data) - 44
            sample_rate = 24000
            bytes_per_sample = 2  # 16-bit audio
            return audio_samples / (sample_rate * bytes_per_sample)

    async def make_request(self, session: aiohttp.ClientSession, request_id: str, text: str) -> ThroughputMetrics:
        """Make a single request and measure metrics with retry mechanism for 429 errors."""
        metrics = ThroughputMetrics(request_id=request_id, start_time=time.time())

        max_retries = 3
        base_delay = 1.0  # Base delay in seconds

        for attempt in range(max_retries + 1):
            try:
                # Prepare form data
                form_data = aiohttp.FormData()
                form_data.add_field("text", text)
                form_data.add_field("streaming", "true")

                # Make request
                async with session.post(
                    f"{self.base_url}/generate",
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout
                ) as response:
                    if response.status == 429:  # Rate limit exceeded
                        if attempt < max_retries:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** attempt) + (time.time() % 1)  # Add jitter
                            await asyncio.sleep(delay)
                            continue
                        else:
                            metrics.error_message = f"HTTP 429: Rate limit exceeded after {max_retries} retries"
                            return metrics

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

                        chunk_count += 1
                        audio_chunks.append(chunk)

                    metrics.end_time = time.time()
                    metrics.latency = metrics.end_time - metrics.start_time

                    # Combine audio chunks and calculate duration
                    if len(audio_chunks) > 1:
                        # Skip the first chunk (WAV header) and combine only audio data
                        audio_data = b"".join(audio_chunks[1:])
                        # Create a proper WAV file with correct header
                        data_size = len(audio_data)
                        header = struct.pack(
                            '<4sI4s4sIHHIIHH4sI',
                            b'RIFF',
                            36 + data_size,
                            b'WAVE',
                            b'fmt ',
                            16,
                            1,  # PCM
                            1,  # channels
                            24000,  # sample_rate
                            48000,  # byte_rate (24000 * 1 * 16 / 8)
                            2,  # block_align (1 * 16 / 8)
                            16,  # bits_per_sample
                            b'data',
                            data_size
                        )
                        full_audio = header + audio_data
                    else:
                        full_audio = b"".join(audio_chunks)

                    metrics.audio_duration = self.get_audio_duration(full_audio)

                    # Save audio if enabled
                    if self.save_audio and full_audio:
                        output_path = os.path.join(self.output_dir, f"{request_id}.wav")
                        with open(output_path, "wb") as f:
                            f.write(full_audio)

                    metrics.success = True
                    break  # Success, exit retry loop

            except asyncio.TimeoutError:
                metrics.error_message = "Request timeout"
                break  # Don't retry on timeout
            except Exception as e:
                metrics.error_message = str(e)
                break  # Don't retry on other exceptions

        if not metrics.success:
            print(f"Request {request_id} failed: {metrics.error_message}")
        if not metrics.end_time:
            metrics.end_time = time.time()
            metrics.latency = metrics.end_time - metrics.start_time

        return metrics

    async def run_throughput_benchmark(self, num_requests: int) -> ThroughputResults:
        """Run throughput benchmark by sending all requests simultaneously."""
        print(f"Starting throughput benchmark: {num_requests} simultaneous requests")
        print(f"Target server: {self.base_url}")
        print("=" * 60)

        # Create HTTP session with unlimited connections
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        timeout = aiohttp.ClientTimeout(total=600)

        benchmark_start_time = time.time()

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all request tasks
            tasks = []
            for i in range(num_requests):
                request_id = f"req_{i+1:06d}"
                text = self.get_text(i)
                task = asyncio.create_task(self.make_request(session, request_id, text))
                tasks.append(task)

            print(f"Sent {len(tasks)} requests simultaneously. Waiting for completion...")

            # Wait for all requests to complete
            completed_metrics = await asyncio.gather(*tasks, return_exceptions=True)

            benchmark_end_time = time.time()

            # Collect metrics
            for result in completed_metrics:
                if isinstance(result, ThroughputMetrics):
                    self.metrics.append(result)

        total_time = benchmark_end_time - benchmark_start_time
        return self.calculate_results(num_requests, total_time)

    def calculate_results(self, num_requests: int, total_time: float) -> ThroughputResults:
        """Calculate aggregated throughput benchmark results."""
        results = ThroughputResults(num_requests=num_requests, total_time=total_time)

        if not self.metrics:
            return results

        successful_metrics = [m for m in self.metrics if m.success]
        results.successful_requests = len(successful_metrics)
        results.failed_requests = len(self.metrics) - results.successful_requests

        if not successful_metrics:
            return results

        # Extract latency values
        audio_duration_values = [m.audio_duration for m in successful_metrics if m.audio_duration is not None]

        # Calculate throughput metrics
        if total_time > 0:
            results.throughput_req_per_sec = results.successful_requests / total_time

            # Calculate RTF: (sum of generated audio duration) / (wallclock time)
            if audio_duration_values:
                total_audio_duration = sum(audio_duration_values)
                results.rtf = total_audio_duration / total_time

        return results

    def print_results(self, results: ThroughputResults):
        """Print detailed benchmark results."""
        print("\n" + "="*80)
        print("THROUGHPUT BENCHMARK RESULTS")
        print("="*80)

        # Request Summary
        print("\n## Request Summary")
        print(f"Total requests: {results.num_requests}")
        print(f"Successful requests: {results.successful_requests}")
        print(f"Failed requests: {results.failed_requests}")
        success_rate = (results.successful_requests / results.num_requests * 100) if results.num_requests > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")

        # Timing Summary
        print("\n## Timing Summary")
        print(f"Total benchmark time: {results.total_time:.3f}s")
        print(f"Throughput: {results.throughput_req_per_sec:.2f} req/s")
        print(f"Real Time Factor (RTF): {results.rtf:.2f}x")

        print()


async def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for vox-serve TTS server")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--num-requests", type=int, default=10,
                       help="Number of simultaneous requests to send (default: 10)")
    parser.add_argument("--save-audio", action="store_true", help="Save generated audio files")
    parser.add_argument("--data-source", type=str, default="fixed",
                       choices=["fixed", "hifi", "libritts", "lj-speech"],
                       help="Input data source: 'fixed' for fixed text, or dataset name (default: fixed)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible experiments (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducible results
    random.seed(args.seed)

    # Validate arguments
    if args.num_requests <= 0:
        print("Error: Number of requests must be positive")
        return 1

    # Create and run benchmark
    benchmark = ThroughputBenchmark(args.host, args.port, args.save_audio, args.data_source)

    # Run throughput benchmark
    results = await benchmark.run_throughput_benchmark(args.num_requests)

    # Print results
    benchmark.print_results(results)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
