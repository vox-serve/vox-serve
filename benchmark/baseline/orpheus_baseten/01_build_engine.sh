#!/usr/bin/env bash
# =============================================================================
# Build the FP8 TRT-LLM engine for Orpheus 3B, faithful to Baseten's recipe at
# truss-examples/orpheus-best-performance/config.yaml:
#
#   build:
#     base_model: decoder
#     checkpoint_repository:
#       repo: baseten/orpheus-3b-0.1-ft
#       revision: b9eb57a06083cb9e5a083885fad991aa79c0bd24
#     max_batch_size: 256
#     max_num_tokens: 16384
#     max_seq_len: 32768
#     num_builder_gpus: 1
#     quantization_config:
#       calib_dataset: cnn_dailymail
#     plugin_configuration:
#       use_fp8_context_fmha: true
#     quantization_type: fp8_kv
#     tensor_parallel_count: 1
#
# Two stages:
#   1. Quantize HF checkpoint to FP8 with KV-cache also in FP8.
#   2. trtllm-build with fp8_context_fmha + paged_context_fmha (paged is needed
#      for chunked-context runtime).
#
# Run on the cluster after 00_setup_cluster.sh:
#   set -a; source ./.env; set +a
#   bash 01_build_engine.sh
#
# Expect ~30-90 min on an H100 80GB (mostly FP8 calibration).
# =============================================================================
set -euo pipefail

# Load .env from this script's dir (if not already exported)
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${THIS_DIR}/.env" ]; then
  set -a; source "${THIS_DIR}/.env"; set +a
fi

: "${STORAGE_ROOT:?run 00_setup_cluster.sh first or source .env}"
: "${ORPHEUS_HF_DIR:?missing}"
: "${FP8_CKPT_DIR:?missing}"
: "${ENGINE_DIR:?missing}"
: "${TRT_LLM_REPO:?missing}"
: "${CONDA_ENV_NAME:?missing}"

source "${HOME}/miniconda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# TRT-LLM ships its quantize+convert scripts in examples/quantization/ and
# examples/llama/. Locate them relative to the cloned repo.
QUANTIZE_PY="${TRT_LLM_REPO}/examples/quantization/quantize.py"
if [ ! -f "${QUANTIZE_PY}" ]; then
  echo "ERROR: ${QUANTIZE_PY} not found. Did you clone TensorRT-LLM at ${TRT_LLM_REPO}?"
  exit 1
fi

# Force calibration data + scratch into /mnt/storage so we don't fill $HOME.
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

# ---- Stage 1: FP8 quantization ---------------------------------------------
# `--qformat fp8 --kv_cache_dtype fp8` matches `quantization_type: fp8_kv`.
# `--calib_dataset cnn_dailymail` mirrors Baseten's config.
# `--calib_size 512` is the TRT-LLM default for FP8 Llama calibration.
echo "[build] stage 1: FP8 calibration + checkpoint conversion -> ${FP8_CKPT_DIR}"
python3 "${QUANTIZE_PY}" \
  --model_dir "${ORPHEUS_HF_DIR}" \
  --output_dir "${FP8_CKPT_DIR}" \
  --dtype float16 \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --calib_size 512 \
  --calib_dataset "cnn_dailymail" \
  --tp_size 1

# ---- Stage 2: trtllm-build --------------------------------------------------
# `--use_fp8_context_fmha enable` mirrors `plugin_configuration.use_fp8_context_fmha: true`.
# `--use_paged_context_fmha enable` is required for the runtime's
# `enable_chunked_context: true` setting (chunked-context support).
# max_batch_size/max_num_tokens/max_seq_len match Baseten's config verbatim.
echo "[build] stage 2: trtllm-build -> ${ENGINE_DIR}"
trtllm-build \
  --checkpoint_dir "${FP8_CKPT_DIR}" \
  --output_dir "${ENGINE_DIR}" \
  --gemm_plugin auto \
  --use_fp8_context_fmha enable \
  --use_paged_context_fmha enable \
  --max_batch_size 256 \
  --max_num_tokens 16384 \
  --max_seq_len 32768 \
  --tp_size 1 \
  --workers 1

echo "[build] DONE. Engine at ${ENGINE_DIR}"
echo "[build] Next: python server.py --engine-dir \"${ENGINE_DIR}\" --tokenizer-dir \"${ORPHEUS_HF_DIR}\" --data-dir \"${DATA_DIR}\" --snac-dir \"${SNAC_DIR}\""
