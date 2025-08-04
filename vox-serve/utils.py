import os
import requests
from pathlib import Path

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
