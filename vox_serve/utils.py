import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safe_load


# Global log level configuration with thread safety
class _LogLevelManager:
    def __init__(self):
        self._level = "INFO"
        self._lock = threading.Lock()

    def set_level(self, level: str) -> None:
        with self._lock:
            self._level = level.upper()

    def get_level(self) -> str:
        with self._lock:
            return self._level


_log_level_manager = _LogLevelManager()


def set_global_log_level(level: str) -> None:
    """
    Set the global log level for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _log_level_manager.set_level(level)


def get_global_log_level() -> str:
    """Get the current global log level."""
    return _log_level_manager.get_level()


def load_hf_safetensor_state_dict(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    *,
    max_workers: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Downloads a safetensors HF repo and returns a merged state_dict (dict[str, Tensor]).

    Supports:
    - Sharded safetensors models with `*.safetensors.index.json`
    - Single `.safetensors` file without an index

    Parallelizes shard loading using a thread pool (good for I/O-heavy workloads).
    Set `max_workers` to tune parallelism; default picks a sensible number.
    """
    cache_dir = snapshot_download(repo_id=repo_id, revision=revision, token=token)
    repo = Path(cache_dir)

    # Try to find the index file
    index_candidates = sorted(repo.glob("*.safetensors.index.json"))

    if index_candidates:
        # ---- Sharded case ----
        index_path = index_candidates[0]
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        # files: list of shard filenames; weight_map: param_name -> shard_filename
        weight_map: Dict[str, str] = index["weight_map"]

        # Group params by shard
        shard_to_params: Dict[str, List[str]] = {}
        for name, shard in weight_map.items():
            shard_to_params.setdefault(shard, []).append(name)

        # Heuristic: keep concurrency moderate to avoid disk thrash
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(len(shard_to_params), max(4, min(8, cpu * 2)))

        def _load_one(shard_file: str, param_names: List[str]) -> Dict[str, torch.Tensor]:
            shard_path = repo / shard_file
            shard_tensors = safe_load(str(shard_path))  # dict[str, Tensor] for this shard
            if strict:
                missing = [k for k in param_names if k not in shard_tensors]
                if missing:
                    raise KeyError(f"Missing parameters in {shard_file}: {missing[:5]}...")
            return {k: shard_tensors[k] for k in param_names if k in shard_tensors}

        state_dict: Dict[str, torch.Tensor] = {}

        if len(shard_to_params) == 1:
            ((shard_file, param_names),) = shard_to_params.items()
            return _load_one(shard_file, param_names)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_load_one, shard_file, param_names): shard_file
                for shard_file, param_names in shard_to_params.items()
            }
            for fut in as_completed(futures):
                shard_file = futures[fut]
                part = fut.result()
                dup = set(part).intersection(state_dict)
                if dup:
                    raise RuntimeError(
                        f"Duplicate parameters across shards (e.g., {sorted(list(dup))[:5]}) while merging {shard_file}"
                    )
                state_dict.update(part)

        return state_dict

    else:
        # ---- Single-file case ----
        safetensor_files = sorted(repo.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError("No .safetensors or .safetensors.index.json files found in repo.")
        if len(safetensor_files) > 1:
            raise RuntimeError("Multiple .safetensors files found but no index.json — can't determine mapping.")
        # Load the single safetensors file
        return safe_load(str(safetensor_files[0]))


def download_github_file(owner: str, repo: str, path: str, branch: str = "main", cache_dir: str = None) -> Path:
    """
    Download a file from a GitHub repo, caching it locally.

    Returns the Path to the cached file.
    """
    # Decide on a cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "voxserve-files"
    else:
        cache_dir = Path(cache_dir)
    # Mirror the GitHub structure: owner/repo/branch/path/to/file
    dest = cache_dir / owner / repo / branch / path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If we’ve already downloaded it, just return
    if dest.exists():
        return dest

    # Otherwise fetch and write
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(url)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """
    Set up a centralized logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses the global log level.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Update existing logger's level if needed
        effective_level = level if level is not None else get_global_log_level()
        logger.setLevel(getattr(logging, effective_level.upper()))
        for handler in logger.handlers:
            handler.setLevel(getattr(logging, effective_level.upper()))
        return logger

    effective_level = level if level is not None else get_global_log_level()
    logger.setLevel(getattr(logging, effective_level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, effective_level.upper()))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str, level: str = None) -> logging.Logger:
    """Get or create a logger with the given name using the global log level by default."""
    return setup_logger(name, level)
