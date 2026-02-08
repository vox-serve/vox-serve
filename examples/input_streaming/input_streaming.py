#!/usr/bin/env python3
"""
Input Streaming Example for vox-serve

Demonstrates how to use the input streaming API to send text incrementally
while audio is being generated AND streamed back concurrently. This is useful for:
- Real-time transcription to speech
- Conversational AI with streaming LLM output
- Live captioning

The new flow uses the /audio endpoint to receive audio immediately:
1. POST /generate/stream/start -> get request_id
2. GET /generate/stream/{request_id}/audio (in background thread) -> receive audio chunks
3. POST /generate/stream/{request_id}/text (multiple times) -> send text chunks
4. POST /generate/stream/{request_id}/end -> signal text completion

Usage:
    # Basic usage with default text (sends word by word)
    python input_streaming.py

    # Custom text with simulated typing delay
    python input_streaming.py --text "Hello, this is a test of streaming input."

    # Faster word sending
    python input_streaming.py --delay 0.05

    # Connect to different server
    python input_streaming.py --url http://localhost:8080

    # LLM mode with an OpenAI-compatible API (e.g., Ollama)
    python input_streaming.py --mode llm --llm-url http://localhost:11434/v1 --llm-model llama3
"""

import argparse
import json
import re
import sys
import threading
import time

import requests


def split_into_words(text: str) -> list:
    """
    Split text into words while preserving punctuation attached to words.

    Args:
        text: Input text to split

    Returns:
        List of words with trailing spaces preserved where appropriate
    """
    # Split on whitespace but keep the structure
    # This regex splits on spaces while keeping punctuation with words
    words = re.findall(r'\S+\s*', text)
    return words


def stream_text_to_speech(
    text: str,
    output_path: str = "output.wav",
    base_url: str = "http://localhost:8000",
    word_delay: float = 0.1,
    speaker: str = None,
    language: str = None,
):
    """
    Stream text to the vox-serve API word by word and receive audio concurrently.

    Uses the new /audio endpoint to receive audio chunks immediately as they
    are generated, while continuing to send text via /text endpoint.

    Args:
        text: Full text to synthesize (will be sent word by word)
        output_path: Path to save the output audio file
        base_url: URL of the vox-serve API server
        word_delay: Delay between sending words (simulates typing)
        speaker: Optional speaker ID for multi-speaker models
        language: Optional language code
    """
    words = split_into_words(text)

    print("Starting input streaming request...")
    print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    print(f"Words: {len(words)}, delay: {word_delay}s per word")
    print()

    # Step 1: Start the streaming request
    start_data = {}
    if speaker:
        start_data["speaker"] = speaker
    if language:
        start_data["language"] = language

    try:
        resp = requests.post(f"{base_url}/generate/stream/start", data=start_data if start_data else None)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {base_url}")
        print("Make sure vox-serve is running with --scheduler-type input_streaming")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error starting stream: {e}")
        sys.exit(1)

    request_id = resp.json()["request_id"]
    print(f"Request ID: {request_id}")

    # Shared state for audio receiver thread
    audio_state = {
        "total_bytes": 0,
        "chunks_received": 0,
        "first_chunk_time": None,
        "error": None,
    }
    start_time = time.time()

    # Step 2: Start receiving audio in background thread
    def receive_audio():
        try:
            resp = requests.get(
                f"{base_url}/generate/stream/{request_id}/audio",
                stream=True
            )
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if audio_state["first_chunk_time"] is None:
                        audio_state["first_chunk_time"] = time.time() - start_time
                        print(f"  [Audio] First chunk received at {audio_state['first_chunk_time']:.2f}s")
                    f.write(chunk)
                    audio_state["total_bytes"] += len(chunk)
                    audio_state["chunks_received"] += 1
        except Exception as e:
            audio_state["error"] = str(e)

    audio_thread = threading.Thread(target=receive_audio, daemon=True)
    audio_thread.start()

    # Step 3: Send text word by word (while audio is being received)
    words_sent = 0

    for i, word in enumerate(words):
        try:
            resp = requests.post(
                f"{base_url}/generate/stream/{request_id}/text",
                data={"text": word}
            )
            resp.raise_for_status()
            words_sent += 1
            # Show word without trailing whitespace for cleaner output
            chunks = audio_state['chunks_received']
            chunk_info = f" (audio chunks: {chunks})" if chunks > 0 else ""
            print(f"  Sent word {words_sent}: '{word.rstrip()}'{chunk_info}")
        except requests.exceptions.HTTPError as e:
            print(f"Error sending word: {e}")
            sys.exit(1)

        # Simulate typing delay (skip delay after last word)
        if word_delay > 0 and i < len(words) - 1:
            time.sleep(word_delay)

    text_send_time = time.time() - start_time
    print()
    print(f"Sent {words_sent} words in {text_send_time:.2f}s")

    # Step 4: Signal end of text input
    print("Signaling end of text input...")
    try:
        resp = requests.post(f"{base_url}/generate/stream/{request_id}/end")
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error ending stream: {e}")
        sys.exit(1)

    # Step 5: Wait for audio thread to complete
    print("Waiting for remaining audio chunks...")
    audio_thread.join(timeout=60)  # 60 second timeout

    total_time = time.time() - start_time

    if audio_state["error"]:
        print(f"Error receiving audio: {audio_state['error']}")
        sys.exit(1)

    print()
    print(f"Audio saved to: {output_path}")
    print(f"Audio size: {audio_state['total_bytes'] / 1024:.1f} KB")
    print(f"Audio chunks received: {audio_state['chunks_received']}")
    if audio_state["first_chunk_time"]:
        print(f"Time to first audio chunk: {audio_state['first_chunk_time']:.2f}s")
    print(f"Text send time: {text_send_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")


def llm_mode(
    base_url: str = "http://localhost:8000",
    speaker: str = None,
    llm_url: str = None,
    llm_api_key: str = None,
    llm_model: str = None,
):
    """
    LLM mode: chat with an LLM and hear its responses spoken.

    User input is sent to the LLM and the streaming response is converted
    to speech in real-time. Audio is streamed back concurrently using the
    /audio endpoint.

    Args:
        base_url: URL of the vox-serve TTS API server
        speaker: Optional speaker ID for multi-speaker models
        llm_url: URL of the OpenAI-compatible LLM API server
        llm_api_key: Optional API key for the LLM server
        llm_model: Model name to use for chat completions
    """
    if llm_url is None:
        print("Error: --llm-url is required for LLM mode")
        sys.exit(1)

    print("LLM to TTS Streaming Mode")
    print("=" * 40)
    print(f"LLM: {llm_url}")
    if llm_model:
        print(f"Model: {llm_model}")
    print("Type your message and press Enter to chat with the LLM.")
    print("The LLM response will be streamed to TTS in real-time.")
    print("Audio is streamed back concurrently as it's generated.")
    print("Type 'QUIT' to exit.")
    print()

    # Conversation history
    messages = []

    while True:
        # Get user input first
        try:
            user_input = input("You: ")
        except EOFError:
            break

        if user_input.upper() == "QUIT":
            print("Exiting.")
            return

        if not user_input.strip():
            continue

        # Start streaming request for this turn
        start_data = {}
        if speaker:
            start_data["speaker"] = speaker

        try:
            resp = requests.post(
                f"{base_url}/generate/stream/start",
                data=start_data if start_data else None
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to TTS server at {base_url}")
            print("Make sure vox-serve is running with --scheduler-type input_streaming")
            sys.exit(1)

        request_id = resp.json()["request_id"]

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Audio state for receiver thread
        output_path = "llm_output.wav"
        audio_state = {"total_bytes": 0, "error": None}

        # Start receiving audio in background
        def receive_audio(rid, path, state):
            try:
                resp = requests.get(
                    f"{base_url}/generate/stream/{rid}/audio",
                    stream=True
                )
                resp.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        state["total_bytes"] += len(chunk)
            except Exception as e:
                state["error"] = str(e)

        audio_thread = threading.Thread(
            target=receive_audio,
            args=(request_id, output_path, audio_state),
            daemon=True
        )
        audio_thread.start()

        # Stream LLM response to TTS
        print("Assistant: ", end="", flush=True)

        try:
            tokens_sent = _stream_llm_to_tts(
                messages=messages,
                llm_url=llm_url,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                tts_url=base_url,
                request_id=request_id,
            )
        except Exception as e:
            print(f"\nError streaming from LLM: {e}")
            continue

        print()  # Newline after streaming response

        if tokens_sent == 0:
            print("(No response from LLM)")
            continue

        # Signal end of text input
        resp = requests.post(f"{base_url}/generate/stream/{request_id}/end")
        resp.raise_for_status()

        # Wait for audio to finish
        audio_thread.join(timeout=60)

        if audio_state["error"]:
            print(f"Error receiving audio: {audio_state['error']}")
        else:
            print(f"Audio saved to: {output_path} ({audio_state['total_bytes'] / 1024:.1f} KB)")
        print()


def _stream_llm_to_tts(
    messages: list,
    llm_url: str,
    llm_api_key: str,
    llm_model: str,
    tts_url: str,
    request_id: str,
) -> int:
    """
    Stream LLM response tokens to the TTS server.

    Args:
        messages: Conversation history for chat completion
        llm_url: URL of the OpenAI-compatible API server
        llm_api_key: Optional API key
        llm_model: Model name to use
        tts_url: URL of the vox-serve TTS server
        request_id: TTS streaming request ID

    Returns:
        Number of tokens sent to TTS
    """
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    # Prepare request body
    body = {
        "messages": messages,
        "stream": True,
    }
    if llm_model:
        body["model"] = llm_model

    # Make streaming request to LLM
    chat_url = llm_url.rstrip("/") + "/chat/completions"
    resp = requests.post(chat_url, headers=headers, json=body, stream=True)
    resp.raise_for_status()

    tokens_sent = 0
    full_response = ""

    # Process SSE stream
    for line in resp.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue

        data = line[6:]  # Remove "data: " prefix
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if content:
                # Print to console
                print(content, end="", flush=True)

                # Send to TTS
                tts_resp = requests.post(
                    f"{tts_url}/generate/stream/{request_id}/text",
                    data={"text": content}
                )
                tts_resp.raise_for_status()
                tokens_sent += 1
                full_response += content

        except json.JSONDecodeError:
            continue

    # Add assistant response to message history
    if full_response:
        messages.append({"role": "assistant", "content": full_response})

    return tokens_sent


def main():
    parser = argparse.ArgumentParser(
        description="Input streaming example for vox-serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic streaming with default text (word by word)
  python input_streaming.py

  # Custom text
  python input_streaming.py --text "Hello world, this is a test."

  # LLM mode with Ollama
  python input_streaming.py --mode llm --llm-url http://localhost:11434/v1 --llm-model llama3

  # LLM mode with OpenAI API
  python input_streaming.py --mode llm --llm-url https://api.openai.com/v1 \\
      --llm-api-key $OPENAI_API_KEY --llm-model gpt-4

  # Faster word sending
  python input_streaming.py --delay 0.01
        """
    )

    parser.add_argument(
        "--text",
        default="This is a demonstration of input streaming. "
                "Text is sent word by word while audio is being generated. "
                "This enables real-time text to speech applications.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between words in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Speaker ID for multi-speaker models"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'zh')"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "llm"],
        default="basic",
        help="Demo mode: basic (default), llm (chat with LLM)"
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        help="OpenAI-compatible LLM API URL (e.g., http://localhost:11434/v1). "
             "Required for LLM mode."
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the LLM server (optional)"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name for chat completions (e.g., 'gpt-4', 'llama3')"
    )

    args = parser.parse_args()

    if args.mode == "llm":
        llm_mode(
            base_url=args.url,
            speaker=args.speaker,
            llm_url=args.llm_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
        )
    else:
        stream_text_to_speech(
            text=args.text,
            output_path=args.output,
            base_url=args.url,
            word_delay=args.delay,
            speaker=args.speaker,
            language=args.language,
        )


if __name__ == "__main__":
    main()
