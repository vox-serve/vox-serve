#!/bin/bash
# Builds what a release publishes, inside a manylinux_2_28 container (as root):
#   docker run --rm -v "$PWD:/io" -w /io quay.io/pypa/manylinux_2_28_x86_64:<tag> .github/scripts/build-dists.sh [outdir]
# - the sdist (the Python package plus the Rust sources, MANIFEST.in);
# - from that sdist, the wheel with the Rust extension vox_serve._rt, repaired by auditwheel into a
#   manylinux wheel. The extension uses the stable ABI (cp312-abi3), so this one wheel serves every Python
#   >= 3.12. libcudart.so.12 is not bundled (--exclude): the extension has to share the CUDA runtime torch
#   loads, and finds it through its rpath (the nvidia-cuda-runtime-cu12 package next to it, or torch/lib).
# Environment: CUDA_VERSION (default 12-8), RUST_TOOLCHAIN (default stable), VOX_CUDA_ARCHS (default: every
# architecture of rust/build.rs).
set -euo pipefail
OUT=${1:-dist}
CUDA_VERSION=${CUDA_VERSION:-12-8}
PY=/opt/python/cp312-cp312/bin/python

.github/scripts/install-cuda-toolkit.sh "$CUDA_VERSION"
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION//-/.}
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
  sh -s -- -y -q --profile minimal --default-toolchain "${RUST_TOOLCHAIN:-stable}"
export PATH=$HOME/.cargo/bin:$PATH
rustc --version

WORK=$(mktemp -d)
$PY -m pip install -q build
# With no --sdist / --wheel, build makes the sdist and then the wheel from the unpacked sdist, so the sdist
# is known to contain everything the Rust build needs.
$PY -m build --outdir "$WORK" .
# The extension is optional (pyproject.toml): a failed build still gives a wheel, without it. Not here.
if ! $PY -m zipfile -l "$WORK"/*.whl | grep -q "vox_serve/_rt\.abi3\.so"; then
  echo "error: the wheel does not contain the Rust extension" >&2
  exit 1
fi
mkdir -p "$OUT"
auditwheel repair --exclude libcudart.so.12 --plat manylinux_2_28_x86_64 -w "$OUT" "$WORK"/*.whl
cp "$WORK"/*.tar.gz "$OUT"/
# For the record (symbol versions). It calls the wheel "linux_x86_64" only because of the excluded libcudart.
auditwheel show "$OUT"/*.whl
ls -l "$OUT"
