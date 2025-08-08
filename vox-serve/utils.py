import json
import requests
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safe_load

def load_hf_safetensor_state_dict(repo_id: str, revision: str | None = None, token: str | None = None):
    """
    Downloads a sharded safetensors HF repo and returns a merged state_dict (dict[str, Tensor]).
    Works with repos that include `model.safetensors.index.json`.
    """
    cache_dir = snapshot_download(repo_id=repo_id, revision=revision, token=token)
    repo = Path(cache_dir)

    # Find the index file (usually "model.safetensors.index.json")
    index_candidates = list(repo.glob("*.safetensors.index.json"))
    if not index_candidates:
        raise FileNotFoundError("No *.safetensors.index.json found in repo; is this a sharded safetensors model?")
    index_path = index_candidates[0]

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # files: list of shard filenames; weight_map: param_name -> shard_filename
    weight_map: dict[str, str] = index["weight_map"]

    # Load each shard once, then pull the tensors needed
    shard_to_params: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_to_params.setdefault(shard, []).append(name)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_file, param_names in shard_to_params.items():
        shard_path = repo / shard_file
        shard_tensors = safe_load(str(shard_path))  # returns dict[str, Tensor]
        # copy only needed keys (safe_load already returns only this shard’s tensors)
        for k in param_names:
            if k not in shard_tensors:
                raise KeyError(f"Parameter {k} not found in shard {shard_file}")
            state_dict[k] = shard_tensors[k]

    return state_dict


def download_github_file(
    owner: str,
    repo: str,
    path: str,
    branch: str = "main",
    cache_dir: str = None
) -> Path:
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
