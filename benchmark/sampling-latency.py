#!/usr/bin/env python3
"""
Benchmarking latency of full sampling pipeline for TTS inference.

Profiles the complete sampling pipeline including:
- Repetition penalty application
- Core sampling methods
- Repetition cache updates
- Input feature/mask updates
- Output token appending to request state

Does not require loading actual models - only tests the sampling algorithms.

Usage:
    python sampling-latency.py --vocab-size 32000 --batch-size 8 --num-iterations 1000
"""

import argparse
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List
from queue import Queue

import torch

# Import sampling components from vox-serve
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vox-serve'))

from sampling import SamplingConfig, Sampler

# Mock Request class for benchmarking
@dataclass
class MockRequest:
    """Mock request class for benchmarking sampling pipeline."""
    request_id: str
    lm_output_tokens: List[List[int]] = field(default_factory=list)
    lm_output_audio_tokens: List[List[int]] = field(default_factory=list)
    input_features: torch.Tensor = None
    input_masks: torch.Tensor = None
    repetition_cache: torch.Tensor = None


@dataclass
class SamplingBenchmarkConfig:
    """Configuration for sampling benchmark."""
    vocab_size: int = 32000
    batch_size: int = 8
    num_iterations: int = 1000
    n_codebooks: int = 8  # Number of codebooks for multi-codebook models
    hidden_size: int = 2048  # Hidden size for input features
    repetition_window: int = 10  # Repetition penalty window size
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SamplingLatencyResults:
    """Results for a single sampling method."""
    method_name: str
    mean_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    
    # Component breakdown
    repetition_penalty_latency: float = 0.0
    sampling_latency: float = 0.0
    cache_update_latency: float = 0.0
    state_update_latency: float = 0.0


class SamplingLatencyBenchmark:
    """Benchmark class for profiling sampling method latencies."""

    def __init__(self, config: SamplingBenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Pre-generate synthetic logits for consistent benchmarking
        torch.manual_seed(42)
        self.logits = torch.randn(
            config.batch_size, config.n_codebooks, config.vocab_size, 
            device=self.device, dtype=torch.float16
        )
        
        # Create mock requests for full pipeline testing
        self.mock_requests = self._create_mock_requests()
        
        # Constants for masking (similar to actual models)
        self.masked_token_id = config.vocab_size - 1
    
    def _create_mock_requests(self) -> List[MockRequest]:
        """Create mock requests with initialized state."""
        requests = []
        for i in range(self.config.batch_size):
            # Create repetition cache
            repetition_cache = torch.zeros(
                self.config.repetition_window, 
                self.config.n_codebooks,
                self.config.vocab_size,
                dtype=torch.bool,
                device=self.device
            )
            
            # Create input features and masks
            input_features = torch.zeros(
                1, self.config.hidden_size,
                device=self.device, 
                dtype=torch.bfloat16
            )
            input_masks = torch.zeros(
                1, self.config.n_codebooks,
                dtype=torch.bool,
                device=self.device
            )
            
            req = MockRequest(
                request_id=f"req_{i}",
                repetition_cache=repetition_cache,
                input_features=input_features,
                input_masks=input_masks
            )
            
            # Pre-populate with some tokens (simulate ongoing generation)
            for _ in range(3):  # Add 3 initial tokens
                token = [torch.randint(0, self.config.vocab_size, (1,)).item() 
                        for _ in range(self.config.n_codebooks)]
                req.lm_output_tokens.append(token)
                req.lm_output_audio_tokens.append(token)
            
            requests.append(req)
        
        return requests


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

    def run(self, sampling_config: SamplingConfig, method_name: str) -> SamplingLatencyResults:
        """Benchmark the complete sampling pipeline including all components."""
        
        # Warmup runs
        for warmup_i in range(10):
            requests = deepcopy(self.mock_requests)
            logits = self.logits.clone()
            
            with torch.no_grad():
                # Run the same pipeline as actual benchmark for warmup
                for j, req in enumerate(requests):
                    if req.repetition_cache is not None and sampling_config.repetition_penalty:
                        logits[j] = Sampler.apply_repetition_penalty(
                            logits[j], req.repetition_cache, sampling_config.repetition_penalty
                        )
                
                output_ids = Sampler.run_sampling(
                    logits.view(-1, self.config.vocab_size), 
                    config=sampling_config
                )
                output_ids = output_ids.view(logits.shape[0], logits.shape[1])
                
                for j, req in enumerate(requests):
                    if req.repetition_cache is not None and sampling_config.repetition_window:
                        Sampler.update_repetition_penalty_cache(
                            req.repetition_cache,
                            output_ids[j],
                            sampling_config.repetition_window,
                        )
            
            # Synchronize after warmup iteration
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        
        latencies = []
        rep_penalty_times = []
        sampling_times = []
        cache_update_times = []
        state_update_times = []
        
        for i in range(self.config.num_iterations):
            # Create fresh copies of requests for each iteration
            requests = deepcopy(self.mock_requests)
            logits = self.logits.clone()
            
            # Synchronize before starting benchmark
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # 1. Apply repetition penalty
                rep_start = time.perf_counter()
                for j, req in enumerate(requests):
                    if req.repetition_cache is not None and sampling_config.repetition_penalty:
                        logits[j] = Sampler.apply_repetition_penalty(
                            logits[j], req.repetition_cache, sampling_config.repetition_penalty
                        )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                rep_end = time.perf_counter()
                
                # 2. Run sampling
                sampling_start = time.perf_counter()
                output_ids = Sampler.run_sampling(
                    logits.view(-1, self.config.vocab_size), 
                    config=sampling_config
                )
                output_ids = output_ids.view(logits.shape[0], logits.shape[1])
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                sampling_end = time.perf_counter()
                
                # 3. Update repetition cache
                cache_start = time.perf_counter()
                for j, req in enumerate(requests):
                    if req.repetition_cache is not None and sampling_config.repetition_window:
                        Sampler.update_repetition_penalty_cache(
                            req.repetition_cache,
                            output_ids[j],
                            sampling_config.repetition_window,
                        )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                cache_end = time.perf_counter()
                
                # 4. Update request state (masking, features, tokens)
                state_start = time.perf_counter()
                for j, req in enumerate(requests):
                    # Mask tokens for multi-codebook models
                    if len(req.lm_output_tokens) + 1 < self.config.n_codebooks:
                        for k in range(len(req.lm_output_tokens) + 1, self.config.n_codebooks):
                            output_ids[j, k] = self.masked_token_id
                    
                    # Update input features and masks (decode phase)
                    req.input_features = torch.zeros(
                        1, self.config.hidden_size, 
                        device=self.device, 
                        dtype=torch.bfloat16
                    )
                    req.input_masks = torch.zeros(
                        1, self.config.n_codebooks, 
                        dtype=torch.bool, 
                        device=self.device
                    )
                    
                    # Append output tokens
                    req.lm_output_tokens.append(output_ids[j].tolist())
                    # Simulate stop condition check (not adding EOS to audio tokens)
                    if not self._is_stop_token(output_ids[j].tolist()):
                        req.lm_output_audio_tokens.append(output_ids[j].tolist())
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                state_end = time.perf_counter()
            
            end_time = time.perf_counter()
            
            # Record timings
            total_latency = (end_time - start_time) * 1000
            latencies.append(total_latency)
            rep_penalty_times.append((rep_end - rep_start) * 1000)
            sampling_times.append((sampling_end - sampling_start) * 1000)
            cache_update_times.append((cache_end - cache_start) * 1000)
            state_update_times.append((state_end - state_start) * 1000)
        
        # Calculate statistics
        sorted_latencies = sorted(latencies)
        mean_latency = statistics.mean(latencies)
        
        return SamplingLatencyResults(
            method_name=method_name,
            mean_latency=mean_latency,
            p50_latency=self._percentile(sorted_latencies, 50),
            p90_latency=self._percentile(sorted_latencies, 90),
            p95_latency=self._percentile(sorted_latencies, 95),
            p99_latency=self._percentile(sorted_latencies, 99),
            min_latency=min(latencies),
            max_latency=max(latencies),
            repetition_penalty_latency=statistics.mean(rep_penalty_times),
            sampling_latency=statistics.mean(sampling_times),
            cache_update_latency=statistics.mean(cache_update_times),
            state_update_latency=statistics.mean(state_update_times)
        )
    
    def _is_stop_token(self, token_list: List[int]) -> bool:
        """Check if token represents a stop/EOS token."""
        # Simple heuristic: assume last vocab token is EOS
        return any(token == self.config.vocab_size - 1 for token in token_list)

    def run_all_benchmarks(self) -> Dict[str, SamplingLatencyResults]:
        """Run benchmarks for all sampling methods."""
        results = {}
        
        pipeline_configs = [
            (SamplingConfig(temperature=0.0, repetition_penalty=1.1, repetition_window=self.config.repetition_window), 
             "greedy_rep_penalty"),
            (SamplingConfig(top_k=50, temperature=1.0, repetition_penalty=1.1, repetition_window=self.config.repetition_window),
             "top_k_rep_penalty"),
            (SamplingConfig(top_p=0.9, temperature=1.0, repetition_penalty=1.1, repetition_window=self.config.repetition_window),
             "top_p_rep_penalty"),
            (SamplingConfig(top_k=50, top_p=0.9, temperature=1.0, repetition_penalty=1.1, repetition_window=self.config.repetition_window),
             "top_k_p_rep_penalty"),
            (SamplingConfig(min_p=0.05, temperature=1.0, repetition_penalty=1.1, repetition_window=self.config.repetition_window),
             "min_p_rep_penalty"),
        ]
        
        for config, key in pipeline_configs:
            results[key] = self.run(config, key)
        
        return results

    def print_results(self, results: Dict[str, SamplingLatencyResults]):
        """Print formatted benchmark results."""

        # Calculate column widths
        if not results:
            return
            
        results_list = list(results.values())
        
        # Column headers and their minimum widths
        headers = ["Method", "Total", "P50", "P90", "P99", "Rep Penalty", "Sampling", "Cache Update", "State Update"]
        min_widths = [len(h) for h in headers]
        
        # Calculate maximum width needed for each column based on data
        max_widths = min_widths.copy()
        
        for result in results_list:
            # Method name
            max_widths[0] = max(max_widths[0], len(result.method_name))
            # Total latency (format: X.XXX)
            max_widths[1] = max(max_widths[1], len(f"{result.mean_latency:.3f}"))
            # P50 latency
            max_widths[2] = max(max_widths[2], len(f"{result.p50_latency:.3f}"))
            # P90 latency
            max_widths[3] = max(max_widths[3], len(f"{result.p90_latency:.3f}"))
            # P99 latency
            max_widths[4] = max(max_widths[4], len(f"{result.p99_latency:.3f}"))
            # Rep penalty latency
            max_widths[5] = max(max_widths[5], len(f"{result.repetition_penalty_latency:.3f}"))
            # Sampling latency
            max_widths[6] = max(max_widths[6], len(f"{result.sampling_latency:.3f}"))
            # Cache update latency
            max_widths[7] = max(max_widths[7], len(f"{result.cache_update_latency:.3f}"))
            # State update latency
            max_widths[8] = max(max_widths[8], len(f"{result.state_update_latency:.3f}"))
        
        # Print header
        header_row = "| " + " | ".join(f"{headers[i]:<{max_widths[i]}}" for i in range(len(headers))) + " |"
        print(header_row)
        
        # Print separator
        separator = "|" + "|".join(f"{'-' * (max_widths[i] + 2)}" for i in range(len(headers))) + "|"
        print(separator)
        
        # Print data rows
        for result in results_list:
            values = [
                f"{result.method_name:<{max_widths[0]}}",
                f"{result.mean_latency:.3f}".rjust(max_widths[1]),
                f"{result.p50_latency:.3f}".rjust(max_widths[2]),
                f"{result.p90_latency:.3f}".rjust(max_widths[3]),
                f"{result.p99_latency:.3f}".rjust(max_widths[4]),
                f"{result.repetition_penalty_latency:.3f}".rjust(max_widths[5]),
                f"{result.sampling_latency:.3f}".rjust(max_widths[6]),
                f"{result.cache_update_latency:.3f}".rjust(max_widths[7]),
                f"{result.state_update_latency:.3f}".rjust(max_widths[8])
            ]
            row = "| " + " | ".join(values) + " |"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark sampling method latencies")
    parser.add_argument("--vocab-size", type=int, default=32000, 
                       help="Vocabulary size (default: 32000)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations (default: 100)")
    parser.add_argument("--n-codebooks", type=int, default=4,
                       help="Number of codebooks for multi-codebook models (default: 4)")
    parser.add_argument("--hidden-size", type=int, default=2048,
                       help="Hidden size for input features (default: 2048)")
    parser.add_argument("--repetition-window", type=int, default=10,
                       help="Repetition penalty window size (default: 10)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Create benchmark configuration
    config = SamplingBenchmarkConfig(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        n_codebooks=args.n_codebooks,
        hidden_size=args.hidden_size,
        repetition_window=args.repetition_window,
        device=device
    )
    
    # Run benchmark
    benchmark = SamplingLatencyBenchmark(config)
    results = benchmark.run_all_benchmarks()
    benchmark.print_results(results)


if __name__ == "__main__":
    import sys
    sys.exit(main())