"""Input-streaming client: sends text word by word while reading the audio stream.

python rust/scripts/stream_client.py --port 8000 --out rust/logs/wav_stream
"""

import argparse
import os
import struct
import threading
import time

import requests

TEXT = (
    "Streaming text input lets the server start speaking before the whole sentence has arrived, "
    "which keeps the latency low for conversational agents."
)


def run(base: str, out: str, delay: float, tag: str) -> None:
    rid = requests.post(f"{base}/generate/stream/start", data={"speaker": "ryan"}).json()["request_id"]
    audio = bytearray()
    first = []

    def read():
        with requests.get(f"{base}/generate/stream/{rid}/audio", stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None):
                if len(audio) > 44 and not first:
                    first.append(time.time())
                audio.extend(chunk)

    t = threading.Thread(target=read)
    t.start()
    t0 = time.time()
    for w in TEXT.split(" "):
        r = requests.post(f"{base}/generate/stream/{rid}/text", data={"text": w + " "})
        r.raise_for_status()
        time.sleep(delay)
    requests.post(f"{base}/generate/stream/{rid}/end").raise_for_status()
    t.join()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"stream_{tag}.wav")
    # The streamed header has unknown (zero) sizes; fill them in for the saved file.
    audio = bytearray(audio)
    struct.pack_into("<I", audio, 4, len(audio) - 8)
    struct.pack_into("<I", audio, 40, len(audio) - 44)
    with open(path, "wb") as f:
        f.write(audio)
    secs = (len(audio) - 44) / 48000
    ttfa = first[0] - t0 if first else float("nan")
    print(f"{tag}: {secs:.2f}s audio, first audio {ttfa:.2f}s after first text, total {time.time() - t0:.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", default="rust/logs/wav_stream")
    p.add_argument("--delay", type=float, default=0.15)
    p.add_argument("--concurrency", type=int, default=2)
    a = p.parse_args()
    base = f"http://localhost:{a.port}"
    ts = [threading.Thread(target=run, args=(base, a.out, a.delay, str(i))) for i in range(a.concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # Errors: unknown session and a non-streaming model check are covered by status codes.
    assert requests.post(f"{base}/generate/stream/nope/text", data={"text": "x"}).status_code == 404


if __name__ == "__main__":
    main()
