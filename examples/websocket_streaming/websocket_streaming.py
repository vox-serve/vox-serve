"""WebSocket speech streaming client for ``/v1/audio/speech/stream`` (Rust runtime).

Sends text word by word, as an LLM would, and saves the streamed audio. Each ``--text`` is one segment
(ended with ``input.commit``); the session ends with ``input.done``.

    pip install websockets
    python websocket_streaming.py --text "Sure, I can help with that. The station is two blocks north."
    python websocket_streaming.py --ref-audio voice.wav --ref-text "Transcript of voice.wav" \\
        --text "First answer." --text "Second answer, in the same session."
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import time
import wave

import websockets


async def main(args):
    config = {
        "response_format": "pcm",
        "stream_audio": True,
        "split_granularity": args.granularity,
    }
    if args.language:
        config["language"] = args.language
    if args.ref_audio:
        mime = mimetypes.guess_type(args.ref_audio)[0] or "audio/wav"
        with open(args.ref_audio, "rb") as f:
            config["ref_audio"] = f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
        if args.ref_text:
            config["ref_text"] = args.ref_text

    pcm = bytearray()
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.config", "session": config}))
        configured = json.loads(await ws.recv())
        if configured["type"] != "session.configured":
            raise SystemExit(f"session.config failed: {configured}")
        sample_rate = configured["session"]["sample_rate"]

        t0 = time.time()
        first_audio = {}

        async def send_text():
            for segment in args.text:
                for word in segment.split():
                    await ws.send(json.dumps({"type": "input.text", "text": word + " "}))
                    await asyncio.sleep(args.delay)
                await ws.send(json.dumps({"type": "input.commit"}))
            await ws.send(json.dumps({"type": "input.done"}))

        sender = asyncio.create_task(send_text())
        async for msg in ws:
            if isinstance(msg, bytes):
                first_audio.setdefault("t", time.time() - t0)
                pcm.extend(msg)
                continue
            event = json.loads(msg)
            if event["type"] == "input.committed":
                print(
                    f"segment {event['segment_index']} done: {event['segment_sentences']} sentence(s), "
                    f"{len(pcm) / 2 / sample_rate:.1f}s of audio so far"
                )
            elif event["type"] == "error":
                print("error:", event["message"])
            elif event["type"] == "session.done":
                break
        await sender

    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    print(
        f"first audio after {first_audio.get('t', float('nan')):.2f}s; "
        f"saved {len(pcm) / 2 / sample_rate:.1f}s to {args.output}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8000/v1/audio/speech/stream")
    p.add_argument("--text", action="append", help="one segment of text (repeat for several)")
    p.add_argument("--delay", type=float, default=0.05, help="seconds between words")
    p.add_argument("--granularity", choices=["clause", "sentence"], default="clause")
    p.add_argument("--language")
    p.add_argument("--ref-audio", help="reference audio for voice cloning")
    p.add_argument("--ref-text", help="transcript of the reference audio")
    p.add_argument("--output", default="websocket_output.wav")
    args = p.parse_args()
    args.text = args.text or [
        "Sure, I can help with that. The quickest way to the station is the number twelve "
        "bus from the corner, then a short walk north."
    ]
    asyncio.run(main(args))
