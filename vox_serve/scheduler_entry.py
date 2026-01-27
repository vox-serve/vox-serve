"""
Entry point for scheduler daemon subprocess.

This module is deliberately kept separate from launch.py to avoid importing
torch at the module level, which would break CUDA_VISIBLE_DEVICES setting.

IMPORTANT: This module must NOT import torch at the module level!
"""

import argparse
import os
import sys


def _run_scheduler_daemon(
    dp_rank: int,
    dp_size: int,
    model_name: str,
    scheduler_type: str,
    max_batch_size: int,
    max_num_pages,
    page_size: int,
    request_socket_path: str,
    result_socket_path: str,
    top_p,
    top_k: int,
    min_p,
    temperature,
    max_tokens: int,
    repetition_penalty,
    repetition_window: int,
    cfg_scale,
    greedy: bool,
    enable_cuda_graph: bool,
    enable_disaggregation: bool,
    enable_nvtx: bool,
    enable_torch_compile: bool,
    async_scheduling: bool,
    log_level: str,
) -> None:
    """Entry point for scheduler daemon that sets CUDA_VISIBLE_DEVICES before importing torch."""
    # DEBUG: Check if torch is already imported (should NOT be!)
    torch_already_imported = "torch" in sys.modules
    print(f"[DP ENTRY] Rank {dp_rank}: Starting, torch already imported: {torch_already_imported}", flush=True)
    print(
        f"[DP ENTRY] Rank {dp_rank}: CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}",
        flush=True,
    )

    # Now import torch (will see the CUDA_VISIBLE_DEVICES set by parent process)
    import torch

    print(f"[DP ENTRY] Rank {dp_rank}: torch imported, available devices: {torch.cuda.device_count()}", flush=True)
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
    print(f"[DP ENTRY] Rank {dp_rank}: Current device: {current_device}", flush=True)

    if dp_size > 1:
        torch.cuda.empty_cache()

    # Import scheduler
    from vox_serve.scheduler import load_scheduler
    from vox_serve.utils import get_logger, set_global_log_level

    # Set global log level in this subprocess
    set_global_log_level(log_level)
    logger = get_logger(__name__)

    if dp_size > 1:
        cuda_device = os.environ["CUDA_VISIBLE_DEVICES"]
        device_count = torch.cuda.device_count()
        logger.info(f"DP rank {dp_rank} using CUDA device {cuda_device}, torch sees {device_count} devices")

    # Adjust device for data parallel
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = load_scheduler(
        scheduler_type=scheduler_type,
        model_name_or_path=model_name,
        device=device,
        max_batch_size=max_batch_size,
        max_num_pages=max_num_pages,
        page_size=page_size,
        request_socket_path=request_socket_path,
        result_socket_path=result_socket_path,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        cfg_scale=cfg_scale,
        greedy=greedy,
        enable_cuda_graph=enable_cuda_graph,
        enable_disaggregation=enable_disaggregation,
        enable_nvtx=enable_nvtx,
        enable_torch_compile=enable_torch_compile,
        async_scheduling=async_scheduling,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )
    logger.info(f"Scheduler (DP rank {dp_rank}/{dp_size}) started successfully with model: {model_name}")
    scheduler.run_forever()


def main():
    """Main entry point when run as a module."""
    parser = argparse.ArgumentParser(description="Vox-Serve Scheduler Daemon")
    parser.add_argument("--dp-rank", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--scheduler-type", type=str, required=True)
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--max-num-pages", type=int, default=None)
    parser.add_argument("--page-size", type=int, required=True)
    parser.add_argument("--request-socket-path", type=str, required=True)
    parser.add_argument("--result-socket-path", type=str, required=True)
    parser.add_argument("--log-level", type=str, required=True)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--repetition-window", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--enable-disaggregation", action="store_true")
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")

    args = parser.parse_args()

    _run_scheduler_daemon(
        dp_rank=args.dp_rank,
        dp_size=args.dp_size,
        model_name=args.model_name,
        scheduler_type=args.scheduler_type,
        max_batch_size=args.max_batch_size,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        request_socket_path=args.request_socket_path,
        result_socket_path=args.result_socket_path,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        cfg_scale=args.cfg_scale,
        greedy=args.greedy,
        enable_cuda_graph=args.enable_cuda_graph,
        enable_disaggregation=args.enable_disaggregation,
        enable_nvtx=args.enable_nvtx,
        enable_torch_compile=args.enable_torch_compile,
        async_scheduling=args.async_scheduling,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
