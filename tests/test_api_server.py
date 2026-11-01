"""The Python runtime's HTTP API (``vox_serve.launch``) over a scheduler that drives the CPU stand-in worker:
what clients get when a request fails (no GPU, no model)."""

import atexit
import shutil
import socket
import tempfile
import threading
import time

import pytest
import requests
import uvicorn
from scheduler_sim import FakeWorker

from vox_serve import launch
from vox_serve.scheduler import InputStreamingScheduler

WAV_HEADER = 44
AUDIO_BYTES = 10 * 24 * 2  # FakeWorker: 10 frames of 24 int16 samples


class _Server(launch.APIServer):
    """An API server whose scheduler runs in a thread with a ``FakeWorker`` instead of in a process."""

    def _start_schedulers(self):
        self.scheduler_processes = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()

    def _run_scheduler(self):
        scheduler = InputStreamingScheduler(
            model_name_or_path="fake",
            max_batch_size=4,
            request_socket_path=f"{self.request_socket_path}_0",
            result_socket_path=self.result_socket_path,
            model_worker=FakeWorker(),
        )
        while not self.stop.is_set():
            scheduler._step()
            time.sleep(0.001)
        for s in (scheduler.request_socket, scheduler.result_socket):
            s.close(linger=0)
        scheduler.context.term()

    def _stop_scheduler(self):
        self.stop.set()
        self.thread.join(timeout=5)


@pytest.fixture(scope="module")
def server():
    tmp = tempfile.mkdtemp(prefix="vs")  # IPC paths must stay short
    api = _Server(
        scheduler_type="input_streaming",
        request_socket_path=f"{tmp}/req.ipc",
        result_socket_path=f"{tmp}/res.ipc",
        output_dir=f"{tmp}/out",
        timeout_seconds=30,
    )
    atexit.unregister(api.cleanup)
    launch.api_server = api
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    http = uvicorn.Server(uvicorn.Config(launch.app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=http.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not http.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", api
    finally:
        http.should_exit = True
        thread.join(timeout=5)
        launch.api_server = None
        api.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)


def _generate(url, streaming, files=None, **fields):
    data = {"text": "Hello world", "streaming": str(streaming).lower(), **fields}
    return requests.post(f"{url}/generate", data=data, files=files, timeout=30)


def _detail(r):
    return r.json()["detail"]


@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
def test_generate(server, streaming):
    url, _ = server
    r = _generate(url, streaming, voice="tara")
    assert r.status_code == 200, r.text
    assert len(r.content) == WAV_HEADER + AUDIO_BYTES


@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
def test_rejected_request_returns_400_with_the_reason(server, streaming):
    url, _ = server
    r = _generate(url, streaming, voice="nobody")
    assert r.status_code == 400 and _detail(r) == "voice nobody is not available for this model"
    # Audio for a model without audio input is rejected, as in the Rust runtime (not ignored).
    r = _generate(url, streaming, files={"audio": ("ref.wav", b"RIFF....WAVE")})
    assert r.status_code == 400 and _detail(r) == "this model does not accept audio input"


def test_speech_endpoint(server):
    url, _ = server
    r = requests.post(f"{url}/v1/audio/speech", json={"input": "Hello world", "voice": "nobody"}, timeout=30)
    assert r.status_code == 400 and _detail(r) == "voice nobody is not available for this model"
    r = requests.post(f"{url}/v1/audio/speech", json={"input": "Hello world"}, timeout=30)
    assert r.status_code == 200 and len(r.content) == AUDIO_BYTES


def _start(url, **kwargs):
    r = requests.post(f"{url}/generate/stream/start", timeout=30, **kwargs)
    assert r.status_code == 200, r.text
    return r.json()["request_id"]


def _text(url, rid, text):
    return requests.post(f"{url}/generate/stream/{rid}/text", data={"text": text}, timeout=30)


def _end(url, rid):
    return requests.post(f"{url}/generate/stream/{rid}/end", timeout=30)


def _wait_finished(api, rid):
    deadline = time.monotonic() + 10
    while rid not in api.finished_input_streams and time.monotonic() < deadline:
        time.sleep(0.01)
    return api.finished_input_streams[rid][1]


def test_input_streaming(server):
    url, api = server
    rid = _start(url)
    with requests.get(f"{url}/generate/stream/{rid}/audio", stream=True, timeout=30) as audio:
        assert audio.status_code == 200
        assert _text(url, rid, "Hello there, this is a test.").status_code == 200
        assert _end(url, rid).status_code == 200
        assert len(audio.content) == WAV_HEADER + AUDIO_BYTES
    assert _wait_finished(api, rid) is None
    assert _end(url, rid).json()["status"] == "completed"
    r = _text(url, rid, "more")
    assert r.status_code == 400 and _detail(r) == f"Request {rid} already completed"


def test_input_streaming_failure(server):
    url, api = server
    rid = _start(url, files={"audio": ("ref.wav", b"RIFF....WAVE")})
    with requests.get(f"{url}/generate/stream/{rid}/audio", stream=True, timeout=30) as audio:
        assert audio.status_code == 200  # the prompt is built only once enough text has arrived
        assert _text(url, rid, "Hello there, this is a test.").status_code == 200
        # The request fails after its response has started: the connection is cut, not ended cleanly.
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            _ = audio.content
    reason = f"Request {rid} failed: this model does not accept audio input"
    assert _wait_finished(api, rid) == "this model does not accept audio input"
    for r in (_text(url, rid, "more"), _end(url, rid), requests.get(f"{url}/generate/stream/{rid}/audio")):
        assert r.status_code == 400 and _detail(r) == reason


def test_input_streaming_without_text(server):
    url, api = server
    rid = _start(url)
    assert _end(url, rid).status_code == 200
    assert _wait_finished(api, rid) == "no text provided"
    r = requests.get(f"{url}/generate/stream/{rid}/audio", timeout=30)
    assert r.status_code == 400 and _detail(r) == f"Request {rid} failed: no text provided"


def test_input_streaming_needs_its_scheduler(server, monkeypatch):
    url, api = server
    monkeypatch.setattr(api, "scheduler_type", "base")
    r = requests.post(f"{url}/generate/stream/start", timeout=30)
    assert r.status_code == 400 and "--scheduler-type input_streaming" in _detail(r)
    r = _generate(url, False)
    assert r.status_code == 200
