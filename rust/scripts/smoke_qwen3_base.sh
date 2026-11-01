#!/bin/bash
# Qwen3-TTS Base voice cloning: ICL with the default reference, x-vector only with a given reference.
WAV=$HOME/.cache/voxserve-files/stepfun-ai/Step-Audio2/main/assets/default_female.wav
BIN="vox-serve --runtime rust"
set -x
RUST_LOG=info $BIN --model qwen3-tts-base --max-batch-size 4 --bench-requests 2 \
  --bench-out rust/logs/wav_qwen3_base_icl && python rust/scripts/asr_check.py rust/logs/wav_qwen3_base_icl/bench_*.wav
RUST_LOG=info $BIN --model qwen3-tts-base --max-batch-size 4 --bench-requests 2 --bench-audio "$WAV" \
  --bench-kwargs '{"x_vector_only_mode": true}' \
  --bench-out rust/logs/wav_qwen3_base_xvec && python rust/scripts/asr_check.py rust/logs/wav_qwen3_base_xvec/bench_*.wav
