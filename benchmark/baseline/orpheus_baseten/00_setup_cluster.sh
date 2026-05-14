#!/usr/bin/env bash
# =============================================================================
# Setup script for the Baseten Orpheus baseline on the PTC H100 80GB cluster.
#
# What this does:
#   1. Creates a conda env (in $HOME/miniconda — small) for the baseten serving
#      side, isolated from your existing vox-serve / multimodal_inference envs.
#   2. Installs PyTorch 2.7.1+cu128, tensorrt-llm (matching CUDA 12.8), snac,
#      batched, and FastAPI deps.
#   3. Downloads the two HF assets we need into /mnt/storage/atindra (NOT $HOME).
#   4. Writes a `.env` file alongside this script with all paths the later
#      scripts (01_build_engine.sh, server.py, etc.) will pick up.
#
# Run this on the cluster like:
#   bash 00_setup_cluster.sh
#
# Prerequisites on the cluster:
#   - miniconda/anaconda installed at $HOME/miniconda (already present per
#     your home-dir listing).
#   - git, git-lfs available system-wide.
#   - $HF_TOKEN exported in your shell (needed for HF downloads if rate-limited;
#     same token you used for SeedTTS before).
#   - /mnt/storage/atindra exists and is writable (your big storage dir).
# =============================================================================
set -euo pipefail

# ---- Paths (edit if your layout differs) ------------------------------------
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/storage/atindra}"
HF_HOME_DIR="${HF_HOME:-${STORAGE_ROOT}/.huggingface}"
ENGINE_DIR="${ENGINE_DIR:-${STORAGE_ROOT}/baseten_orpheus/engine}"
FP8_CKPT_DIR="${FP8_CKPT_DIR:-${STORAGE_ROOT}/baseten_orpheus/fp8_ckpt}"
DATA_DIR="${DATA_DIR:-${STORAGE_ROOT}/baseten_orpheus/data}"
SNAC_DIR="${SNAC_DIR:-${STORAGE_ROOT}/baseten_orpheus/snac_24khz}"
ORPHEUS_HF_DIR="${ORPHEUS_HF_DIR:-${STORAGE_ROOT}/baseten_orpheus/orpheus_hf}"
TRT_LLM_REPO="${TRT_LLM_REPO:-${STORAGE_ROOT}/TensorRT-LLM}"
TRUSS_EXAMPLES_REPO="${TRUSS_EXAMPLES_REPO:-${STORAGE_ROOT}/truss-examples}"

# Conda env
CONDA_ENV_NAME="${CONDA_ENV_NAME:-baseten_orpheus}"
PYTHON_VERSION="3.10"   # py3.9 per Baseten config, but 3.10 also supported by
                        # TRT-LLM wheels and avoids some deprecated-py warnings.

# Baseten's exact pin (per truss-examples/orpheus-best-performance/config.yaml)
ORPHEUS_REPO_ID="baseten/orpheus-3b-0.1-ft"
ORPHEUS_REVISION="b9eb57a06083cb9e5a083885fad991aa79c0bd24"
SNAC_REPO_ID="hubertsiuzdak/snac_24khz"

# Versions pinned to Baseten's config.yaml
TORCH_VERSION="2.7.1"
SNAC_VERSION="1.2.1"
BATCHED_VERSION="0.1.4"
CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu128"

# ---- Make storage dirs ------------------------------------------------------
mkdir -p "${STORAGE_ROOT}/baseten_orpheus" "${HF_HOME_DIR}" \
         "${ENGINE_DIR}" "${FP8_CKPT_DIR}" "${DATA_DIR}" "${SNAC_DIR}" "${ORPHEUS_HF_DIR}"

# ---- Conda env --------------------------------------------------------------
source "${HOME}/miniconda/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  echo "[setup] creating conda env ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})..."
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${CONDA_ENV_NAME}"

# ---- Python deps ------------------------------------------------------------
# Match Baseten's pinned versions where possible; tensorrt-llm pulled from
# pypi (NVIDIA's official wheel; targets CUDA 12.x).
echo "[setup] installing torch ${TORCH_VERSION} (cu128)..."
pip install --upgrade pip
pip install --extra-index-url "${CUDA_WHL_INDEX}" "torch==${TORCH_VERSION}"

echo "[setup] installing tensorrt-llm..."
# If you've already built TRT-LLM from source at ${TRT_LLM_REPO}, you can skip
# this and `pip install -e ${TRT_LLM_REPO}` instead. The pypi wheel is simpler.
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

echo "[setup] installing serving + benchmark deps..."
pip install \
  "snac==${SNAC_VERSION}" \
  "batched==${BATCHED_VERSION}" \
  fastapi \
  "uvicorn[standard]" \
  aiohttp \
  numpy \
  datasets \
  huggingface_hub \
  transformers

# ---- HF downloads -----------------------------------------------------------
export HF_HOME="${HF_HOME_DIR}"
echo "[setup] HF_HOME=${HF_HOME}"

echo "[setup] downloading ${ORPHEUS_REPO_ID} @ ${ORPHEUS_REVISION} ..."
huggingface-cli download "${ORPHEUS_REPO_ID}" \
  --revision "${ORPHEUS_REVISION}" \
  --local-dir "${ORPHEUS_HF_DIR}" \
  --max-workers 2

echo "[setup] downloading ${SNAC_REPO_ID} ..."
huggingface-cli download "${SNAC_REPO_ID}" \
  --local-dir "${SNAC_DIR}" \
  --max-workers 2

# Baseten's model.py expects the tokenizer under {data_dir}/tokenization/.
# We mirror that layout by symlinking the Orpheus repo's tokenizer files.
echo "[setup] populating ${DATA_DIR}/tokenization ..."
mkdir -p "${DATA_DIR}/tokenization"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt config.json generation_config.json; do
  if [ -f "${ORPHEUS_HF_DIR}/${f}" ]; then
    ln -sf "${ORPHEUS_HF_DIR}/${f}" "${DATA_DIR}/tokenization/${f}"
  fi
done

# ---- Write .env for downstream scripts --------------------------------------
ENV_FILE="$(dirname "$0")/.env"
cat > "${ENV_FILE}" <<EOF
# Auto-generated by 00_setup_cluster.sh. Source this from your shell:
#   set -a; source $(realpath "${ENV_FILE}"); set +a
export STORAGE_ROOT="${STORAGE_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export ENGINE_DIR="${ENGINE_DIR}"
export FP8_CKPT_DIR="${FP8_CKPT_DIR}"
export DATA_DIR="${DATA_DIR}"
export SNAC_DIR="${SNAC_DIR}"
export ORPHEUS_HF_DIR="${ORPHEUS_HF_DIR}"
export TRT_LLM_REPO="${TRT_LLM_REPO}"
export TRUSS_EXAMPLES_REPO="${TRUSS_EXAMPLES_REPO}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME}"
export ORPHEUS_REPO_ID="${ORPHEUS_REPO_ID}"
export ORPHEUS_REVISION="${ORPHEUS_REVISION}"
EOF
echo "[setup] wrote ${ENV_FILE}"

echo "[setup] DONE. Next: bash 01_build_engine.sh"
