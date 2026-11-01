#!/bin/bash
# Qwen3-TTS input streaming: serve, stream text word by word from two clients, then Whisper ASR.
set -x
PORT=${PORT:-$((20000 + RANDOM % 10000))}
OUT=rust/logs/wav_stream
RUST_LOG=info vox-serve --runtime rust --model qwen3-tts --max-batch-size 4 --port $PORT &
SERVER=$!
until curl -sf "http://localhost:$PORT/health"; do
    kill -0 $SERVER 2>/dev/null || exit 1
    sleep 5
done
python rust/scripts/stream_client.py --port $PORT --out $OUT
STATUS=$?
kill $SERVER
wait $SERVER
python rust/scripts/asr_check.py $OUT/stream_*.wav
exit $STATUS
