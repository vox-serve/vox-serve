#!/bin/bash
# Disaggregation: LM on GPU 0, detok on GPU 1, for a stateless (Orpheus) and a stateful (Qwen3)
# detokenizer; then the same Orpheus bench without disaggregation for comparison.
set -x
BIN="vox-serve --runtime rust"
for M in orpheus qwen3-tts; do
  OUT=rust/logs/wav_disagg_$M
  RUST_LOG=info $BIN --model $M --enable-disaggregation --max-batch-size 8 --bench-requests 8 --bench-out $OUT || exit 1
  python rust/scripts/asr_check.py $OUT/bench_0.wav $OUT/bench_7.wav
done
RUST_LOG=info $BIN --model orpheus --max-batch-size 8 --bench-requests 8
