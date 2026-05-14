#!/usr/bin/env bash
# =============================================================================
# Launch vox-serve's Orpheus serving, run vox-serve's own goodput.py sweep,
# tear server down. Numbers come from vox-serve/benchmark/goodput.py unchanged
# (so the metric definitions stay exactly the same as the paper's reported
# results — no chance of measurement drift between this and the Baseten run).
#
# Run AFTER setting up vox-serve on this cluster (separate conda env from the
# baseten one). Assumes vox-serve is importable as `vox_serve`.
#
# Run:
#   set -a; source ./.env; set +a
#   bash run_voxserve_sweep.sh
# =============================================================================
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${THIS_DIR}/.env" ]; then
  set -a; source "${THIS_DIR}/.env"; set +a
fi

: "${STORAGE_ROOT:?source .env first}"

# vox-serve uses a DIFFERENT conda env than baseten_orpheus; default is "vox-serve".
VOXSERVE_ENV="${VOXSERVE_ENV:-vox-serve}"
VOXSERVE_REPO="${VOXSERVE_REPO:-${STORAGE_ROOT}/vox-serve}"
PORT="${PORT:-8001}"   # different from baseten port so they can coexist if needed
RESULTS_DIR="${RESULTS_DIR:-${STORAGE_ROOT}/baseten_orpheus/results}"
RATES_DEFAULT="0.5 1.0 2.0 4.0 6.0 8.0 10.0"
RATES="${RATES:-$RATES_DEFAULT}"
DURATION="${DURATION:-60}"
DATA_SOURCE="${DATA_SOURCE:-libritts}"
SEED="${SEED:-42}"

# Paper-matched Orpheus sampling (paper §4.1, lines 693-695):
#   temp=0.6, top_p=0.8, repetition_penalty=1.3, max_batch=128
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.8}"
REP_PEN="${REP_PEN:-1.3}"
MAX_BATCH="${MAX_BATCH:-128}"

OUT_DIR="${RESULTS_DIR}/voxserve"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/server.log"

source "${HOME}/miniconda/etc/profile.d/conda.sh"
conda activate "${VOXSERVE_ENV}"

echo "[run] launching vox-serve Orpheus (logs -> ${LOG})..."
cd "${VOXSERVE_REPO}"
nohup python -m vox_serve.launch \
  --model canopylabs/orpheus-3b-0.1-ft \
  --host 0.0.0.0 --port "${PORT}" \
  --max-batch-size "${MAX_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --repetition-penalty "${REP_PEN}" \
  > "${LOG}" 2>&1 &
SERVER_PID=$!
trap '[ -n "${SERVER_PID:-}" ] && kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo "[run] waiting for vox-serve to become healthy on :${PORT}..."
for i in $(seq 1 300); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1 \
     || curl -fsS "http://localhost:${PORT}/" >/dev/null 2>&1; then
    echo "[run] vox-serve up after ${i}s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[run] vox-serve died during boot; tail of log:"
    tail -n 60 "${LOG}"
    exit 1
  fi
  sleep 1
done

echo "[run] running vox-serve goodput.py sweep..."
cd "${VOXSERVE_REPO}/benchmark"
python goodput.py \
  --host localhost --port "${PORT}" \
  --rate ${RATES} \
  --duration "${DURATION}" \
  --burstiness 1.0 \
  --data-source "${DATA_SOURCE}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_DIR}/sweep.log"

echo "[run] sweep complete; raw output at ${OUT_DIR}/sweep.log"
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "[run] done."
