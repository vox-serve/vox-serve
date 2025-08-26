# Streaming TTS Web Client

(This client code is vibe-coded with Gemini)

A simple `index.html` client for a streaming voice synthesis server that uses the Web Audio API for true, low-latency playback.

### How to Run

1.  Make sure your synthesis server is running.
2.  From this directory, start a local web server using Python:
    ```sh
    python -m http.server <port>
    ```
3.  Open your browser and navigate to `http://localhost:<port>`.
4.  Confirm the server URL in the interface, enter text, and click "Generate and Play" to hear the audio stream.
