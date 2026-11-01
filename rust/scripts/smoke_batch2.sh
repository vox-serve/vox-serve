#!/bin/bash
# Zonos, Step-Audio-2 (text and audio input), GLM-4-Voice audio input.
WAV=$HOME/.cache/voxserve-files/stepfun-ai/Step-Audio2/main/assets/default_female.wav
bash rust/scripts/smoke_all.sh zonos step
echo "######## step audio-in"
RUST_LOG=info vox-serve --runtime rust --model step --max-batch-size 4 --bench-requests 2 \
  --bench-audio "$WAV" --bench-out rust/logs/wav_step_audioin && python rust/scripts/asr_check.py rust/logs/wav_step_audioin/bench_*.wav
