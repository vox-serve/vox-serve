#!/bin/bash
# Installs what the Rust extension's build script needs from the CUDA toolkit (no driver, no GPU):
# nvcc, the runtime headers and libcudart (cuda-cudart-dev), CUB (cuda-cccl) and cuRAND's device
# header (libcurand-dev), from NVIDIA's apt (Ubuntu) or dnf (RHEL-compatible, e.g. the manylinux_2_28
# containers) repository. Usage: install-cuda-toolkit.sh [version, e.g. 12-8]
set -euo pipefail
CUDA_VERSION=${1:-12-8}
SUDO=$([ "$(id -u)" -eq 0 ] || echo sudo)
VERSION_ID=$(. /etc/os-release && echo "$VERSION_ID")
if command -v apt-get >/dev/null; then
  REPO="ubuntu${VERSION_ID//./}"
  curl -fsSLo /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i /tmp/cuda-keyring.deb
  $SUDO apt-get update -q
  $SUDO apt-get install -y -q --no-install-recommends \
    "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-dev-${CUDA_VERSION}" "cuda-cccl-${CUDA_VERSION}" "libcurand-dev-${CUDA_VERSION}"
else
  REPO="rhel${VERSION_ID%%.*}"
  $SUDO curl -fsSLo "/etc/yum.repos.d/cuda-${REPO}.repo" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-${REPO}.repo"
  $SUDO dnf install -y -q --setopt=install_weak_deps=False \
    "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-devel-${CUDA_VERSION}" "cuda-cccl-${CUDA_VERSION}" "libcurand-devel-${CUDA_VERSION}"
  $SUDO dnf clean all -q
fi
"/usr/local/cuda-${CUDA_VERSION//-/.}/bin/nvcc" --version
