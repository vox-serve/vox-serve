import json
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


# ---------------------------------------------------------------------------
# Ming-UniAudio dynamic code resolver (HF-only, no Git clone required)
# ---------------------------------------------------------------------------
MING_CODE_REPO_DEFAULT = os.environ.get("MING_CODE_REPO", "inclusionAI/Ming-UniAudio")
MING_CODE_REV_DEFAULT = os.environ.get("MING_CODE_REV", "main")
MING_CODE_CACHE_ROOT = Path(
    os.environ.get(
        "MING_CODE_CACHE_DIR",
        Path.home() / ".cache" / "vox-serve" / "ming_code",
    )
)
MING_CODE_REQUIRED: tuple[str, ...] = (
    "configuration_bailingmm.py",
    "modeling_bailingmm.py",
    "modeling_utils.py",
    "audio_tokenizer/modeling_audio_vae.py",
    "fm/flowloss.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenization_bailing.py",
)
MING_CODE_ALLOW_PATTERNS: tuple[str, ...] = (
    "configuration_bailingmm.py",
    "configuration_bailing_moe.py",
    "configuration_glm.py",
    "modeling_bailingmm.py",
    "modeling_bailing_moe.py",
    "modeling_utils.py",
    "audio_processing_bailingmm.py",
    "bailingmm_utils.py",
    "chat_format.py",
    "audio_tokenizer/*",
    "fm/*",
    "processing_bailingmm.py",
    "tokenization_bailing.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)
MING_CODE_COPY_ITEMS: tuple[str, ...] = (
    "configuration_bailingmm.py",
    "configuration_bailing_moe.py",
    "modeling_bailingmm.py",
    "modeling_bailing_moe.py",
    "modeling_utils.py",
    "audio_processing_bailingmm.py",
    "bailingmm_utils.py",
    "chat_format.py",
    "tokenization_bailing.py",
    "processing_bailingmm.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "audio_tokenizer",
    "fm",
)


def _ming_contains_all(root: Path) -> bool:
    return all((root / relative).exists() for relative in MING_CODE_REQUIRED)


def _ming_candidate_dirs(model_path: str | Path | None) -> Iterable[Path]:
    vendor_dir = Path(__file__).resolve().parents[1] / "Ming-UniAudio"
    if vendor_dir.exists():
        yield vendor_dir

    explicit = os.environ.get("MING_CODE_DIR")
    if explicit:
        path = Path(explicit)
        if path.exists():
            yield path

    if model_path:
        mp = Path(model_path)
        if mp.exists():
            yield mp


def ensure_ming_code_available(model_path: str | Path | None) -> Path:
    """
    Guarantee the Ming-UniAudio dynamic Python modules are importable.

    The resolver prefers existing modules, then local directories (model snapshot,
    explicit overrides, vendored copy), and finally downloads a whitelist of files
    from Hugging Face Hub into a local cache.
    """
    try:
        import configuration_bailingmm  # type: ignore  # noqa: F401
        import modeling_bailingmm  # type: ignore  # noqa: F401
        return Path(configuration_bailingmm.__file__).resolve().parent  # type: ignore[attr-defined]
    except Exception:
        pass

    for candidate in _ming_candidate_dirs(model_path):
        if _ming_contains_all(candidate):
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            get_logger(__name__).info("Using Ming code from %s", candidate_str)
            return candidate.resolve()

    repo_id = os.environ.get("MING_CODE_REPO", MING_CODE_REPO_DEFAULT)
    revision = os.environ.get("MING_CODE_REV", MING_CODE_REV_DEFAULT)
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=list(MING_CODE_ALLOW_PATTERNS),
                cache_dir=str(MING_CODE_CACHE_ROOT),
            )
        )
    except Exception as exc:  # pragma: no cover - hub error
        get_logger(__name__).warning(
            "Failed to download Ming code from %s@%s: %s",
            repo_id,
            revision,
            exc,
        )
        snapshot_path = None

    if snapshot_path and _ming_contains_all(snapshot_path):
        cache_str = str(snapshot_path.resolve())
        if cache_str not in sys.path:
            sys.path.insert(0, cache_str)
        get_logger(__name__).info("Using Ming code from cached directory %s", cache_str)
        return snapshot_path.resolve()

    missing = "\n".join(f"  - {name}" for name in MING_CODE_WHITELIST)
    raise RuntimeError(
        "Unable to locate Ming-UniAudio dynamic Python modules. "
        "Ensure the following files are present in either the model directory, "
        "MING_CODE_DIR, or available via Hugging Face under the repo "
        f"{repo_id}@{revision} with allow_patterns {MING_CODE_ALLOW_PATTERNS}.\n"
        f"Missing entries:\n{missing}"
    )


def materialize_ming_code(model_dir: Path, code_source: Path) -> None:
    """
    Ensure essential Ming code files exist inside `model_dir` by copying them from `code_source`.
    """
    if not model_dir.exists():
        return

    # If the source of code is exactly the same directory as model_dir,
    # nothing to materialize.
    try:
        if model_dir.resolve() == code_source.resolve():
            return
    except Exception:
        # Best-effort: if resolve() fails (permissions, etc.), continue with file-level checks.
        pass

    for relative in MING_CODE_COPY_ITEMS:
        src = code_source / relative
        dst = model_dir / relative
        if not src.exists():
            continue
        if src.is_dir():
            # Avoid copying a directory onto itself
            try:
                if dst.exists() and dst.resolve() == src.resolve():
                    continue
            except Exception:
                pass
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            # Skip if destination already exists or is the same file
            if dst.exists():
                try:
                    if os.path.samefile(src, dst):
                        continue
                except Exception:
                    # If samefile fails, fall back to a conservative skip when file exists
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
