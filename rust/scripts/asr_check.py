"""Transcribe generated WAVs with Whisper to sanity-check intelligibility.

python rust/scripts/asr_check.py [--json OUT] rust/logs/wav_sample/*.wav
"""

import json
import sys
import wave

import numpy as np
import torch
import torchaudio
from transformers import pipeline


def main():
    args = sys.argv[1:]
    out = None
    if args[:1] == ["--json"]:
        out, args = args[1], args[2:]
    rows = []
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0, torch_dtype=torch.float16)
    for path in args:
        with wave.open(path) as w:
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        x = torchaudio.functional.resample(torch.from_numpy(pcm), sr, 16000).numpy()
        text = asr({"raw": x, "sampling_rate": 16000})["text"]
        print(f"{path}: {len(pcm) / sr:.2f}s :: {text.strip()}")
        rows.append({"path": path, "audio_s": len(pcm) / sr, "text": text.strip()})
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
