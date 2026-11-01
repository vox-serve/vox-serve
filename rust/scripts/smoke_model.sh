#!/bin/bash
# Usage: smoke_model.sh <model> [extra vox-serve args...]: greedy + sampled bench runs, then Whisper ASR.
set -x
MODEL=$1; shift
TAG=$(basename "$MODEL")
OUT=rust/logs/wav_${TAG}
RUST_LOG=info vox-serve --runtime rust --model "$MODEL" --max-batch-size 4 --bench-requests 4 --bench-out "$OUT" "$@"
python rust/scripts/asr_check.py "$OUT"/bench_*.wav
