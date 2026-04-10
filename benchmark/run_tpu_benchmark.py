#!/usr/bin/env python3
"""Run the goodput benchmark with TPU-appropriate timeouts.

Usage:
    python benchmark/run_tpu_benchmark.py --host localhost --port 8765 \
        --rate 0.05 0.1 0.15 0.2 --duration 60
"""
import asyncio
import sys

import aiohttp

sys.path.insert(0, ".")
import benchmark.goodput as bg

# Increase timeouts for TPU (decode steps take ~200ms each, need 10+ steps)
_orig_run_benchmark = bg.BenchmarkClient.run_benchmark


async def _patched_run_benchmark(self, rate, duration, burstiness=1.0):
    """Override timeout to 120s sock_read for TPU."""
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Re-implement the scheduling loop from the original
        import time

        import numpy as np

        print(f"Starting benchmark: {rate} req/s for {duration}s (burstiness={burstiness})")
        print(f"Target server: {self.base_url}")
        print("=" * 60)

        end_time = time.time() + duration
        request_count = 0
        next_request_time = time.time()
        tasks = []

        while next_request_time < end_time:
            current_time = time.time()
            if next_request_time > current_time:
                await asyncio.sleep(next_request_time - current_time)

            request_count += 1
            request_id = f"req_{request_count:06d}"
            task = asyncio.create_task(self.make_request(session, request_id))
            tasks.append(task)

            if rate > 0:
                shape_k = burstiness
                scale_theta = 1.0 / (burstiness * rate)
                inter_arrival_time = np.random.gamma(shape_k, scale_theta)
            else:
                inter_arrival_time = float("inf")
            next_request_time += inter_arrival_time

        print(f"Scheduled {len(tasks)} requests. Waiting for completion...")
        completed_metrics = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed_metrics:
            if isinstance(result, bg.RequestMetrics):
                self.metrics.append(result)
                status = "\u2713" if result.success else "\u2717"
                ttfa_str = f"{result.ttfa:.3f}s" if result.ttfa else "N/A"
                sv_str = f"{result.streaming_viability:.1f}%" if result.streaming_viability is not None else "N/A"
                print(f"{status} {result.request_id}: TTFA={ttfa_str}, Streaming_viability={sv_str}")

    return self.calculate_results(rate)


bg.BenchmarkClient.run_benchmark = _patched_run_benchmark

# Also patch make_request to use longer per-request timeout
_orig_make_request = bg.BenchmarkClient.make_request


async def _patched_make_request(self, session, request_id):
    """Wrap original make_request with longer per-request timeout."""
    import time as _time

    metrics = bg.RequestMetrics(request_id=request_id, start_time=_time.time())
    try:
        text = self.generate_random_text()
        form_data = aiohttp.FormData()
        form_data.add_field("text", text)
        form_data.add_field("streaming", "true")
        async with session.post(
            f"{self.base_url}/generate",
            data=form_data,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=120),
        ) as response:
            if response.status != 200:
                metrics.error_message = f"HTTP {response.status}: {await response.text()}"
                return metrics
            audio_chunks = []
            chunk_count = 0
            async for chunk in response.content.iter_any():
                if not chunk:
                    break
                current_time = _time.time()
                chunk_count += 1
                if chunk_count == 1:
                    audio_chunks.append(chunk)
                    continue
                chunk_duration = self.pcm_duration_bytes(len(chunk))
                if metrics.ttfa is None:
                    metrics.ttfa = current_time - metrics.start_time
                metrics.chunk_arrival_times.append(current_time)
                metrics.chunk_durations.append(chunk_duration)
                audio_chunks.append(chunk)
            metrics.end_time = _time.time()
            full_audio = b"".join(audio_chunks)
            metrics.audio_duration = self.get_audio_duration(full_audio)
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
            metrics.end_time = _time.time()
    return metrics


bg.BenchmarkClient.make_request = _patched_make_request

if __name__ == "__main__":
    sys.exit(asyncio.run(bg.main()))
