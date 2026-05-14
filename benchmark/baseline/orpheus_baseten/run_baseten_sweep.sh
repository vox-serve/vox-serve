#!/usr/bin/env bash
# =============================================================================
# Launch the Baseten Orpheus local server, run goodput sweep, tear server down.
#
# Run on the cluster after 00_setup_cluster.sh + 01_build_engine.sh succeed:
#   set -a; source ./.env; set +a
#   bash run_baseten_sweep.sh
#
# Output goes to ${RESULTS_DIR}/baseten/.
# =============================================================================
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${THIS_DIR}/.env" ]; then
  set -a; source "${THIS_DIR}/.env"; set +a
fi

: "${STORAGE_ROOT:?source .env first}"
: "${ENGINE_DIR:?missing}"
: "${ORPHEUS_HF_DIR:?missing}"
: "${DATA_DIR:?missing}"
: "${SNAC_DIR:?missing}"
: "${TRUSS_EXAMPLES_REPO:?missing}"
: "${CONDA_ENV_NAME:?missing}"

PORT="${PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-${STORAGE_ROOT}/baseten_orpheus/results}"
RATES_DEFAULT="0.5 1.0 2.0 4.0 6.0 8.0 10.0"
RATES="${RATES:-$RATES_DEFAULT}"
DURATION="${DURATION:-60}"
WARMUP="${WARMUP:-3}"
DATA_SOURCE="${DATA_SOURCE:-libritts}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
VOICE="${VOICE:-tara}"

OUT_DIR="${RESULTS_DIR}/baseten"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/server.log"

source "${HOME}/miniconda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "[run] launching server (logs -> ${LOG})..."
nohup python "${THIS_DIR}/server.py" \
  --engine-dir "${ENGINE_DIR}" \
  --tokenizer  "${ORPHEUS_HF_DIR}" \
  --data-dir   "${DATA_DIR}" \
  --snac-dir   "${SNAC_DIR}" \
  --truss-repo "${TRUSS_EXAMPLES_REPO}" \
  --host 0.0.0.0 --port "${PORT}" \
  > "${LOG}" 2>&1 &
SERVER_PID=$!
trap '[ -n "${SERVER_PID:-}" ] && kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo "[run] waiting for server (pid=${SERVER_PID}) to become healthy on :${PORT}..."
for i in $(seq 1 180); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[run] server healthy after ${i}s"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[run] server died during boot; tail of log:"
    tail -n 60 "${LOG}"
    exit 1
  fi
  sleep 1
done

echo "[run] starting goodput sweep: rates=${RATES} duration=${DURATION}s"
python "${THIS_DIR}/goodput_baseten.py" \
  --host localhost --port "${PORT}" \
  --voice "${VOICE}" --max-tokens "${MAX_TOKENS}" \
  --rate ${RATES} \
  --duration "${DURATION}" \
  --burstiness 1.0 \
  --data-source "${DATA_SOURCE}" \
  --warmup "${WARMUP}" \
  --output-dir "${OUT_DIR}" \
  --seed 42 \
  2>&1 | tee "${OUT_DIR}/sweep.log"

echo "[run] sweep complete; output at ${OUT_DIR}"
echo "[run] stopping server (pid=${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "[run] done."
