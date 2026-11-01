#!/bin/bash
# Re-vendors the FlashInfer headers that csrc/fi_plan.cu includes, from the flashinfer-python installed in
# the environment of $VOX_PY (default: python3). Run after changing the flashinfer-python pin in pyproject.toml:
# the planner compiled from these headers must match the kernels that package runs.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUST=$(cd "$HERE/../.." && pwd)
PY=${VOX_PY:-python3}
NVCC=${CUDA_HOME:-/usr/local/cuda-12}/bin/nvcc
DATA=$("$PY" -c "import os, importlib.util as u; print(os.path.join(os.path.dirname(u.find_spec('flashinfer').origin), 'data'))")
DIST=$("$PY" -c "import importlib.metadata as m; print(m.distribution('flashinfer-python')._path)")

rm -rf "$HERE/include"
mkdir -p "$HERE/include"
cd "$DATA/include"
"$NVCC" -M -std=c++17 -arch=sm_90 -I . "$RUST/csrc/fi_plan.cu" | tr ' \\' '\n\n' | grep '^\./\|^flashinfer' \
  | sed 's|^\./||' | xargs realpath --relative-to=. | sort -u | xargs cp --parents -t "$HERE/include"
cp "$DATA/version.txt" "$HERE/version.txt"
cp "$DIST"/licenses/LICENSE "$DIST"/licenses/NOTICE "$HERE/"
echo "vendored FlashInfer $(cat "$HERE/version.txt"):"
find "$HERE/include" -type f | sed "s|$HERE/||" | sort
