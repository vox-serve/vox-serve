#!/bin/bash
# Installs what the Rust extension's build script needs from the CUDA toolkit (no driver, no GPU):
# nvcc, the runtime headers and libcudart (cuda-cudart-dev), CUB (cuda-cccl) and cuRAND's device
# header (libcurand-dev), from NVIDIA's apt repository. Usage: install-cuda-toolkit.sh [version, e.g. 12-8]
set -euo pipefail
CUDA_VERSION=${1:-12-8}
SUDO=$([ "$(id -u)" -eq 0 ] || echo sudo)
REPO="ubuntu$(. /etc/os-release && echo "${VERSION_ID//./}")"
curl -fsSLo /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
$SUDO dpkg -i /tmp/cuda-keyring.deb
$SUDO apt-get update -q
$SUDO apt-get install -y -q --no-install-recommends \
  "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-dev-${CUDA_VERSION}" "cuda-cccl-${CUDA_VERSION}" "libcurand-dev-${CUDA_VERSION}"
"/usr/local/cuda-${CUDA_VERSION//-/.}/bin/nvcc" --version
