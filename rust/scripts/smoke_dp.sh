#!/bin/bash
# Data parallel: two Orpheus engines (GPUs 0, 1); bench requests alternate between them.
set -x
OUT=rust/logs/wav_dp
RUST_LOG=info vox-serve --runtime rust --model orpheus --dp-size 2 --max-batch-size 4 \
  --bench-requests 8 --bench-out $OUT || exit 1
python rust/scripts/asr_check.py $OUT/bench_0.wav $OUT/bench_1.wav
nvidia-smi --query-gpu=index,memory.used --format=csv
