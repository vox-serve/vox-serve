import asyncio
import atexit
import collections
import io
import json
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Optional

import torch
import zmq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .utils import get_global_log_level, get_logger, set_global_log_level

# Module-level logger - will be updated with proper log level in main()
logger = get_logger(__name__)


class APIServer:
    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
        scheduler_type: str = "base",
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_result.ipc",
        output_dir: str = "/tmp/vox_serve_audio",
        timeout_seconds: float = 600.0,
        max_batch_size: int = 8,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_cuda_graph: bool = True,
        enable_disaggregation: bool = False,
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        max_num_pages: int = None,
        page_size: int = 2048,
        async_scheduling: bool = False,
        dp_size: int = 1,
    ):
        self.model_name = model_name
        self.request_socket_path = request_socket_path
        self.result_socket_path = result_socket_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.upload_dir = Path(output_dir) / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max_batch_size
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.cfg_scale = cfg_scale
        self.greedy = greedy
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_disaggregation = enable_disaggregation
        self.enable_nvtx = enable_nvtx
        self.enable_torch_compile = enable_torch_compile
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.scheduler_type = scheduler_type
        self.async_scheduling = async_scheduling
        self.dp_size = dp_size
        self.scheduler_processes = None  # Will be a list for DP mode
        self.logger = get_logger(__name__)

        # Concurrent request tracking
        self.pending_requests: Dict[str, Dict] = {}  # request_id -> {chunks: [], event: threading.Event()}
        # Track recently completed request_ids to absorb late messages without warnings
        self.recently_completed = collections.OrderedDict()  # request_id -> timestamp
        self.recently_completed_ttl_sec = 5.0
        self.request_lock = threading.Lock()
        self.running = True

        # Data parallel routing state
        self.dp_request_counter = 0  # Round-robin counter for request routing

        # Start scheduler process(es)
        self._start_schedulers()

        # Wait a moment for schedulers to initialize
        time.sleep(2)

        # Initialize ZMQ context and sockets
        # Always use rank suffix format for consistency
        self.context = zmq.Context()
        self.request_sockets = []
        self.result_socket = self.context.socket(zmq.PULL)

        # Connect to all scheduler sockets (even if just one)
        for rank in range(self.dp_size):
            req_socket = self.context.socket(zmq.PUSH)
            req_socket.connect(f"ipc://{self.request_socket_path}_{rank}")
            self.request_sockets.append(req_socket)

        # Bind result socket (schedulers connect to us)
        self.result_socket.bind(f"ipc://{result_socket_path}")

        # Set socket options: lower HWMs to surface backpressure earlier
        try:
            for req_socket in self.request_sockets:
                req_socket.setsockopt(zmq.SNDHWM, 256)
                req_socket.setsockopt(zmq.LINGER, 0)
            self.result_socket.setsockopt(zmq.RCVHWM, 1024)
            self.result_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        # Create bounded in-process queue and sender thread to avoid handler blocking on ZMQ
        # Scale queue size with dp_size to avoid bottleneck with multiple replicas
        self.to_scheduler: queue.Queue[bytes] = queue.Queue(maxsize=max(1, self.max_batch_size * 2 * self.dp_size))
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

        # Start background message processing thread
        self.message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_thread.start()

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _start_schedulers(self):
        """Start the scheduler process(es)"""
        try:
            import subprocess
            import sys

            if self.dp_size > 1:
                # Data parallel mode: use subprocess to set CUDA_VISIBLE_DEVICES before Python starts
                self.scheduler_processes = []

                # Parse existing CUDA_VISIBLE_DEVICES mask if present
                import os

                existing_cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if existing_cuda_mask is not None:
                    # User has pre-set a GPU mask, respect it
                    available_gpus = [int(x.strip()) for x in existing_cuda_mask.split(",") if x.strip().isdigit()]
                    if len(available_gpus) < self.dp_size:
                        self.logger.error(
                            f"CUDA_VISIBLE_DEVICES={existing_cuda_mask} provides {len(available_gpus)} GPUs, "
                            f"but --dp-size={self.dp_size} requires {self.dp_size} GPUs"
                        )
                        raise ValueError(f"Insufficient GPUs in CUDA_VISIBLE_DEVICES mask for dp_size={self.dp_size}")
                    self.logger.info(f"Using existing CUDA_VISIBLE_DEVICES mask: {existing_cuda_mask}")
                    gpu_mapping = available_gpus[: self.dp_size]
                else:
                    # No mask set, use 0, 1, 2, ... dp_size-1
                    gpu_mapping = list(range(self.dp_size))

                for rank in range(self.dp_size):
                    request_socket_path = f"{self.request_socket_path}_{rank}"
                    # All schedulers connect to the same result socket (no rank suffix)
                    result_socket_path = self.result_socket_path

                    # Create environment with CUDA_VISIBLE_DEVICES set to the mapped GPU
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_mapping[rank])

                    # Build command to run the scheduler entry point
                    cmd = [
                        sys.executable,
                        "-m",
                        "vox_serve.scheduler_entry",
                        "--dp-rank",
                        str(rank),
                        "--dp-size",
                        str(self.dp_size),
                        "--model-name",
                        self.model_name,
                        "--scheduler-type",
                        self.scheduler_type,
                        "--max-batch-size",
                        str(self.max_batch_size),
                        "--page-size",
                        str(self.page_size),
                        "--request-socket-path",
                        request_socket_path,
                        "--result-socket-path",
                        result_socket_path,
                        "--log-level",
                        get_global_log_level(),
                    ]

                    # Add optional parameters
                    if self.max_num_pages is not None:
                        cmd.extend(["--max-num-pages", str(self.max_num_pages)])
                    if self.top_p is not None:
                        cmd.extend(["--top-p", str(self.top_p)])
                    if self.top_k is not None:
                        cmd.extend(["--top-k", str(self.top_k)])
                    if self.min_p is not None:
                        cmd.extend(["--min-p", str(self.min_p)])
                    if self.temperature is not None:
                        cmd.extend(["--temperature", str(self.temperature)])
                    if self.max_tokens is not None:
                        cmd.extend(["--max-tokens", str(self.max_tokens)])
                    if self.repetition_penalty is not None:
                        cmd.extend(["--repetition-penalty", str(self.repetition_penalty)])
                    if self.repetition_window is not None:
                        cmd.extend(["--repetition-window", str(self.repetition_window)])
                    if self.cfg_scale is not None:
                        cmd.extend(["--cfg-scale", str(self.cfg_scale)])
                    if self.greedy:
                        cmd.append("--greedy")
                    if self.enable_cuda_graph:
                        cmd.append("--enable-cuda-graph")
                    if self.enable_disaggregation:
                        cmd.append("--enable-disaggregation")
                    if self.enable_nvtx:
                        cmd.append("--enable-nvtx")
                    if self.enable_torch_compile:
                        cmd.append("--enable-torch-compile")
                    if self.async_scheduling:
                        cmd.append("--async-scheduling")

                    self.logger.info(f"Starting DP rank {rank} with CUDA_VISIBLE_DEVICES={gpu_mapping[rank]}")
                    process = subprocess.Popen(cmd, env=env)
                    self.scheduler_processes.append(process)
                    self.logger.info(
                        f"Started scheduler process (DP rank {rank}/{self.dp_size}) with PID: {process.pid}"
                    )
            else:
                # Single scheduler mode - use subprocess for consistency
                self.scheduler_processes = None

                # Use rank 0 with suffix for request, but no suffix for result
                request_socket_path = f"{self.request_socket_path}_0"
                result_socket_path = self.result_socket_path

                # Build command to run the scheduler entry point
                cmd = [
                    sys.executable,
                    "-m",
                    "vox_serve.scheduler_entry",
                    "--dp-rank",
                    "0",
                    "--dp-size",
                    "1",
                    "--model-name",
                    self.model_name,
                    "--scheduler-type",
                    self.scheduler_type,
                    "--max-batch-size",
                    str(self.max_batch_size),
                    "--page-size",
                    str(self.page_size),
                    "--request-socket-path",
                    request_socket_path,
                    "--result-socket-path",
                    result_socket_path,
                    "--log-level",
                    get_global_log_level(),
                ]

                # Add optional parameters
                if self.max_num_pages is not None:
                    cmd.extend(["--max-num-pages", str(self.max_num_pages)])
                if self.top_p is not None:
                    cmd.extend(["--top-p", str(self.top_p)])
                if self.top_k is not None:
                    cmd.extend(["--top-k", str(self.top_k)])
                if self.min_p is not None:
                    cmd.extend(["--min-p", str(self.min_p)])
                if self.temperature is not None:
                    cmd.extend(["--temperature", str(self.temperature)])
                if self.max_tokens is not None:
                    cmd.extend(["--max-tokens", str(self.max_tokens)])
                if self.repetition_penalty is not None:
                    cmd.extend(["--repetition-penalty", str(self.repetition_penalty)])
                if self.repetition_window is not None:
                    cmd.extend(["--repetition-window", str(self.repetition_window)])
                if self.cfg_scale is not None:
                    cmd.extend(["--cfg-scale", str(self.cfg_scale)])
                if self.greedy:
                    cmd.append("--greedy")
                if self.enable_cuda_graph:
                    cmd.append("--enable-cuda-graph")
                if self.enable_disaggregation:
                    cmd.append("--enable-disaggregation")
                if self.enable_nvtx:
                    cmd.append("--enable-nvtx")
                if self.enable_torch_compile:
                    cmd.append("--enable-torch-compile")
                if self.async_scheduling:
                    cmd.append("--async-scheduling")

                process = subprocess.Popen(cmd)
                self.scheduler_process = process
                self.logger.info(f"Started scheduler process with PID: {process.pid}")

        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise RuntimeError(f"Could not start scheduler process: {e}") from e

    def _process_messages(self):
        """Background thread to process incoming messages from scheduler"""
        while self.running:
            try:
                while True:
                    # Use NOBLOCK to prevent message loss and add small sleep for efficiency
                    message = self.result_socket.recv(flags=zmq.NOBLOCK)

                    # Parse message format: request_id|TYPE|data
                    parts = message.split(b"|", 2)
                    if len(parts) >= 3:
                        request_id = parts[0].decode("utf-8")
                        message_type = parts[1].decode("utf-8")
                        data = parts[2]
                    else:
                        self.logger.warning(f"Malformed message received: {message[:100]}...")
                        continue

                    # Route message to the appropriate request
                    with self.request_lock:
                        # prune expired entries in recently_completed
                        if self.recently_completed:
                            now = time.time()
                            # pop from left while expired
                            to_pop = []
                            for rid, ts in self.recently_completed.items():
                                if now - ts > self.recently_completed_ttl_sec:
                                    to_pop.append(rid)
                                else:
                                    break
                            for rid in to_pop:
                                self.recently_completed.pop(rid, None)

                        if request_id in self.pending_requests:
                            if message_type == "AUDIO":
                                # Handle audio chunk
                                self.pending_requests[request_id]["chunks"].append(data)
                            elif message_type == "COMPLETION":
                                # Handle completion notification
                                completion_info = json.loads(data.decode("utf-8"))
                                self.logger.info(f"Request {request_id} completed: {completion_info}")
                                self.pending_requests[request_id]["event"].set()
                                # remember completion to suppress late messages
                                self.recently_completed[request_id] = time.time()
                        # If we've very recently completed this request, drop silently (debug only)
                        elif request_id in self.recently_completed:
                            self.logger.debug(
                                f"Dropping late {message_type} for recently completed request {request_id}"
                            )
                        else:
                            # Log when we receive messages for unknown requests
                            self.logger.warning(f"Received {message_type} message for unknown request {request_id}")

            except zmq.Again:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)
                continue
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.error(f"Error in message processing: {e}")
                continue

    def _stop_scheduler(self):
        if self.dp_size > 1:
            # Stop all scheduler processes in DP mode
            if self.scheduler_processes:
                self.logger.info(f"Stopping {self.dp_size} scheduler processes...")
                for i, process in enumerate(self.scheduler_processes):
                    if process.poll() is None:  # Process is still running
                        try:
                            process.terminate()
                            try:
                                process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                self.logger.warning(f"Scheduler {i} didn't terminate gracefully, forcing kill...")
                                process.kill()
                                process.wait(timeout=1)
                        except Exception as e:
                            self.logger.error(f"Error stopping scheduler {i}: {e}")
                self.logger.info("All scheduler processes stopped")
        elif hasattr(self, "scheduler_process") and self.scheduler_process and self.scheduler_process.poll() is None:
            self.logger.info("Stopping scheduler process...")
            try:
                self.scheduler_process.terminate()
                try:
                    self.scheduler_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Scheduler didn't terminate gracefully, forcing kill...")
                    self.scheduler_process.kill()
                    self.scheduler_process.wait(timeout=1)
            except Exception as e:
                self.logger.error(f"Error stopping scheduler: {e}")
            self.logger.info("Scheduler process stopped")

    def _enqueue_request(self, payload: bytes) -> None:
        """Enqueue a request payload to be forwarded to the scheduler.

        Refuses when saturated to keep latency bounded.
        """
        try:
            self.to_scheduler.put_nowait(payload)
        except queue.Full:
            raise HTTPException(status_code=429, detail="Server busy; please retry shortly") from None

    def _sender_loop(self) -> None:
        """Dedicated thread that drains the in-process queue and sends to ZMQ without blocking the handler."""
        backoff_initial = 0.001
        backoff_max = 0.02
        while self.running:
            try:
                try:
                    payload = self.to_scheduler.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Select target socket for data parallel routing (round-robin)
                # Pin this request to the selected rank even under backpressure
                target_socket = self.request_sockets[self.dp_request_counter % self.dp_size]
                self.dp_request_counter += 1

                backoff = backoff_initial
                while self.running:
                    try:
                        # Non-blocking send; back off briefly on ZMQ backpressure
                        target_socket.send(payload, flags=zmq.DONTWAIT)
                        break
                    except zmq.Again:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, backoff_max)
                    except Exception as e:
                        self.logger.error(f"Sender loop error during send: {e}")
                        break
            except Exception as e:
                if self.running:
                    self.logger.error(f"Sender loop error: {e}")

    def start_streaming_request(self, text: str = None, audio_path: str = None) -> str:
        """
        Create and enqueue a streaming request immediately and return its request_id.

        Scheduling is moved out of the streaming generator to avoid deferred start under load.
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Request {request_id} joined for streaming")

        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {
                "chunks": [],
                "event": completion_event,
                "streaming": True,
                "consumed_chunks": 0,
            }

        # Serialize and send to scheduler
        request_dict = {
            "request_id": request_id,
            "prompt": text,
            "audio_path": audio_path,
            "is_streaming": True,
        }
        request_json = json.dumps(request_dict)
        message = f"{request_json}|audio_data_placeholder".encode("utf-8")
        self._enqueue_request(message)

        return request_id

    async def async_stream_chunks(self, request_id: str):
        """
        Async generator yielding audio chunks for an already enqueued streaming request.

        Avoids threadpool saturation by using asyncio and decouples scheduling from streaming.
        """
        start_time = time.time()
        # Stream chunks until completion
        while True:
            # Timeout check
            if time.time() - start_time > self.timeout_seconds:
                # Cleanup on timeout
                with self.request_lock:
                    self.pending_requests.pop(request_id, None)
                raise HTTPException(status_code=500, detail="Generation timed out")

            new_chunks: list[bytes] = []
            done = False
            with self.request_lock:
                request_data = self.pending_requests.get(request_id)
                if request_data:
                    available = len(request_data["chunks"])
                    consumed = request_data.get("consumed_chunks", 0)
                    new_chunks = request_data["chunks"][consumed:available]
                    request_data["consumed_chunks"] = available
                    done = request_data["event"].is_set()
                else:
                    # No request found; treat as done
                    done = True

            for chunk in new_chunks:
                yield chunk

            if done:
                # Yield any remaining chunks and cleanup
                remaining: list[bytes] = []
                with self.request_lock:
                    request_data = self.pending_requests.get(request_id)
                    if request_data:
                        consumed = request_data.get("consumed_chunks", 0)
                        remaining = request_data["chunks"][consumed:]
                        self.pending_requests.pop(request_id, None)
                for chunk in remaining:
                    yield chunk
                break

            # Small async sleep to avoid busy-waiting
            await asyncio.sleep(0.001)

    def generate_audio(self, text: str = None, audio_path: str = None) -> str:
        """
        Generate audio from text and return path to the audio file.

        Args:
            text: Input text to synthesize (optional if audio_path provided)
            audio_path: Path to input audio file (optional)

        Returns:
            Path to the generated audio file

        Raises:
            HTTPException: If generation fails or times out
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Request {request_id} joined for generation")

        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {"chunks": [], "event": completion_event}

        try:
            # Serialize Request object to JSON
            request_dict = {
                "request_id": request_id,
                "prompt": text,
                "audio_path": audio_path,
                "is_streaming": False,
            }

            request_json = json.dumps(request_dict)
            message = f"{request_json}|audio_data_placeholder".encode("utf-8")

            # Enqueue request to scheduler (non-blocking)
            self._enqueue_request(message)

            # Wait for completion or timeout
            if not completion_event.wait(timeout=self.timeout_seconds):
                raise HTTPException(status_code=500, detail="Generation timed out")

            # Retrieve collected audio chunks
            with self.request_lock:
                audio_chunks = self.pending_requests[request_id]["chunks"][:]
                del self.pending_requests[request_id]

            if not audio_chunks:
                raise HTTPException(status_code=500, detail="No audio generated")

            # Save to WAV file
            output_file = self.output_dir / f"{request_id}.wav"
            with wave.open(str(output_file), "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz as per SNAC
                for chunk in audio_chunks:
                    wf.writeframes(chunk)

            return str(output_file)

        except Exception as e:
            # Clean up on error
            with self.request_lock:
                self.pending_requests.pop(request_id, None)
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}") from e

    def cleanup(self):
        """Clean up ZMQ resources and stop scheduler"""
        self.logger.info("Cleaning up API server...")

        # Stop background message processing
        self.running = False
        if hasattr(self, "message_thread") and self.message_thread.is_alive():
            self.message_thread.join(timeout=1)
        if hasattr(self, "sender_thread") and self.sender_thread.is_alive():
            self.sender_thread.join(timeout=1)

        try:
            # Close all request sockets
            if hasattr(self, "request_sockets"):
                for req_socket in self.request_sockets:
                    req_socket.close()
            if hasattr(self, "result_socket"):
                self.result_socket.close()
            if hasattr(self, "context"):
                self.context.term()  # Terminate ZMQ context
        except Exception as e:
            self.logger.error(f"Error cleaning up ZMQ: {e}")

        self._stop_scheduler()


# Initialize FastAPI app
app = FastAPI(title="Vox-Serve API", description="Text-to-Speech API using Orpheus model")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global API server instance (will be initialized in main)
api_server = None


@app.post("/generate")
async def generate(text: str = Form(...), audio: Optional[UploadFile] = File(None), streaming: bool = Form(True)):
    """
    Generate speech from text and return audio file or streaming response.

    Args:
        text: Input text to synthesize
        audio: Optional input audio file
        streaming: Whether to return streaming response (default: True)

    Returns:
        Audio file as direct response (if streaming=False) or streaming audio response (if streaming=True)
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    audio_path = None
    if audio:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = str(api_server.upload_dir / audio_filename)

        # Move file write off the event loop to avoid blocking under load
        content = await audio.read()
        await run_in_threadpool(Path(audio_path).write_bytes, content)

    try:
        if streaming:
            # Streaming response: enqueue request immediately, then stream asynchronously
            request_id = api_server.start_streaming_request(text, audio_path)

            async def audio_stream():
                # WAV header for 24kHz mono 16-bit audio
                wav_header = io.BytesIO()
                with wave.open(wav_header, "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(24000)  # 24kHz
                    wf.writeframes(b"")  # Empty data for header

                # Get header bytes and correct the chunk size for streaming
                wav_header.seek(0)
                header_bytes = wav_header.read()

                # Send WAV header first
                yield header_bytes

                # Stream audio chunks asynchronously
                async for chunk in api_server.async_stream_chunks(request_id):
                    yield chunk

            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=stream_{uuid.uuid4().hex[:8]}.wav",
                    "Cache-Control": "no-cache",
                },
            )
        else:
            # Non-streaming response
            audio_file = await run_in_threadpool(api_server.generate_audio, text, audio_path)
            request_id = Path(audio_file).stem

            return FileResponse(path=audio_file, media_type="audio/wav", filename=f"{request_id}.wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Schedule cleanup of uploaded file after a delay to ensure processing is complete
        if audio_path and Path(audio_path).exists():

            def delayed_cleanup():
                import time

                time.sleep(60)  # Wait 60 seconds before cleanup
                if Path(audio_path).exists():
                    Path(audio_path).unlink()

            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if api_server is not None:
        api_server.cleanup()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"\nReceived signal {signum}, shutting down...")
    if api_server is not None:
        api_server.cleanup()
    import os

    os._exit(0)  # Force immediate exit


def main():
    """Main entry point for the CLI command"""
    import argparse
    import multiprocessing as mp

    import uvicorn

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vox-Serve Text-to-Speech API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="canopylabs/orpheus-3b-0.1-ft",
        help="Model name or path to use for text-to-speech synthesis (default: canopylabs/orpheus-3b-0.1-ft)",
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="base",
        choices=["base", "online", "offline"],
        help="Type of scheduler to use (default: base)",
    )
    parser.add_argument("--async-scheduling", action="store_true", help="Enable async scheduling mode (default: False)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size for inference (default: 8)")
    parser.add_argument(
        "--max-num-pages", type=int, default=2048, help="Maximum number of KV cache pages (default: 1024)"
    )
    parser.add_argument("--page-size", type=int, default=128, help="Size of each KV cache page (default: 128)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter (default: None)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter (default: None)")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling parameter (default: None)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling (default: None)")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum number of tokens to generate (default: model-specific)"
    )
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (default: None)")
    parser.add_argument("--repetition-window", type=int, default=None, help="Repetition window size (default: None)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale for guidance (default: None)")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Enable greedy sampling (ignores top-k, top-p, min-p, and temperature parameters)",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        default=True,
        help="Enable CUDA graph optimization for decode phase (default: True)",
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph optimization for decode phase"
    )
    parser.add_argument(
        "--enable-disaggregation",
        action="store_true",
        help=(
            "Enable disaggregation mode (requires at least 2 GPUs): "
            "LLM on GPU 0, detokenizer on GPU 1 (default: False)"
        ),
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help=(
            "Enable data parallel mode with N replicas (default: 1, disables DP). "
            "Cannot be used with --enable-disaggregation. Requires N <= available GPUs."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--enable-nvtx", action="store_true", help="Enable NVTX profiling for performance analysis (default: False)"
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile optimization for model inference (default: False)",
    )
    parser.add_argument(
        "--socket-suffix",
        type=str,
        default="",
        help="Suffix to append to IPC socket paths to avoid conflicts (default: empty)",
    )
    args = parser.parse_args()

    # Set global log level for the entire application
    set_global_log_level(args.log_level)

    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set, ignore
        pass

    # Determine final CUDA graph setting
    enable_cuda_graph = args.enable_cuda_graph and not args.disable_cuda_graph

    # Validate data parallel mode
    if args.dp_size < 1:
        logger.error("--dp-size must be >= 1")
        sys.exit(1)

    # Check mutual exclusion between DP and disaggregation
    if args.dp_size > 1 and args.enable_disaggregation:
        logger.error(
            "Cannot enable both data parallel mode (--dp-size > 1) and disaggregation mode (--enable-disaggregation)"
        )
        logger.error("Please use one or the other")
        sys.exit(1)

    # Check GPU availability for data parallel
    if args.dp_size > 1:
        available_gpus = torch.cuda.device_count()
        if args.dp_size > available_gpus:
            logger.error(f"--dp-size {args.dp_size} exceeds available GPU count {available_gpus}")
            sys.exit(1)
        logger.info(f"Data parallel mode enabled with {args.dp_size} replicas (using GPUs 0-{args.dp_size - 1})")

    # Automatically select disaggregation scheduler if enable_disaggregation is set with CUDA graphs
    scheduler_type = args.scheduler_type
    if args.enable_disaggregation and enable_cuda_graph:
        logger.info(
            "Disaggregation mode enabled: using 'disaggregation' scheduler with parallel LM and detokenization loops"
        )
        scheduler_type = "disaggregation"

    # Construct socket paths with optional suffix
    request_socket_path = f"/tmp/vox_serve_request{args.socket_suffix}.ipc"
    result_socket_path = f"/tmp/vox_serve_result{args.socket_suffix}.ipc"

    # Initialize API server instance with specified model
    global api_server
    api_server = APIServer(
        model_name=args.model,
        scheduler_type=scheduler_type,
        request_socket_path=request_socket_path,
        result_socket_path=result_socket_path,
        max_batch_size=args.max_batch_size,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        cfg_scale=args.cfg_scale,
        greedy=args.greedy,
        enable_cuda_graph=enable_cuda_graph,
        enable_disaggregation=args.enable_disaggregation,
        enable_nvtx=args.enable_nvtx,
        enable_torch_compile=args.enable_torch_compile,
        async_scheduling=args.async_scheduling,
        dp_size=args.dp_size,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info(f"Starting vox-serve API server with model: {args.model}")
        logger.info("Scheduler and API server will be available shortly...")
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    finally:
        if api_server is not None:
            api_server.cleanup()


if __name__ == "__main__":
    main()
