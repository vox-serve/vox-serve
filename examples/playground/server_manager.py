"""VoxServe server subprocess lifecycle management."""

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


@dataclass
class ServerConfig:
    """Configuration for VoxServe server launch (mirrors CLI arguments)."""

    # Model and server
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    port: int = 12345
    cuda_devices: List[int] = field(default_factory=lambda: [0])
    runtime: str = "rust"  # rust, python, auto (see `vox-serve --help`)
    max_context: Optional[int] = None  # Rust runtime only

    # Scheduler
    scheduler_type: str = "base"  # base, online, offline
    async_scheduling: bool = False

    # Batch and memory
    max_batch_size: int = 8
    max_num_pages: Optional[int] = None  # None: the runtime's default
    page_size: Optional[int] = None

    # Sampling parameters
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    repetition_window: Optional[int] = None
    cfg_scale: Optional[float] = None
    greedy: bool = False

    # Performance
    enable_cuda_graph: bool = True
    enable_disaggregation: bool = False
    dp_size: int = 1
    enable_nvtx: bool = False
    enable_torch_compile: bool = False

    # Other
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    detokenize_interval: Optional[int] = None


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float


@dataclass
class ServerStatus:
    """Current status of the VoxServe server."""

    running: bool
    ready: bool = False
    runtime: Optional[str] = None
    scheduler_type: Optional[str] = None
    model: Optional[str] = None
    port: Optional[int] = None
    cuda_devices: Optional[List[int]] = None
    start_time: Optional[float] = None
    uptime_seconds: Optional[float] = None


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _mib_to_gb(v: str) -> float:
    """nvidia-smi memory in MiB; unified-memory devices (Jetson) report "[N/A]"."""
    try:
        return round(int(v) / 1024, 1)
    except ValueError:
        return 0.0


class VoxServeServerManager:
    """Manages VoxServe server subprocess lifecycle."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.config: Optional[ServerConfig] = None
        self.start_time: Optional[float] = None
        self._log_lines: List[str] = []
        self._max_log_lines: int = 500
        self._ready: bool = False

    def get_available_gpus(self) -> List[GPUInfo]:
        """Detect available NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(", ")
                if len(parts) >= 4:
                    gpus.append(
                        GPUInfo(
                            index=int(parts[0]),
                            name=parts[1].strip(),
                            memory_total_gb=_mib_to_gb(parts[2]),
                            memory_free_gb=_mib_to_gb(parts[3]),
                        )
                    )
            return gpus
        except Exception:
            return []

    def start(self, config: ServerConfig, timeout: float = 600.0) -> tuple[bool, str]:
        """
        Launch VoxServe server with specified configuration.

        Returns:
            Tuple of (success, message)
        """
        if self.process is not None and self.process.poll() is None:
            return False, "Server is already running"

        # Build environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.cuda_devices))
        env["NO_COLOR"] = "1"  # plain Rust-runtime logs (ANSI codes are also stripped below)
        # The Rust runtime rejects --async-scheduling, --repetition-window, --disable-cuda-graph and --enable-nvtx
        # (and `--runtime auto` would fall back to Python for them), so they are left out for it.
        rust = config.runtime == "rust"
        log_level = config.log_level
        if rust:  # the Rust runtime takes debug/info/warning/error
            log_level = "error" if log_level.upper() == "CRITICAL" else log_level.lower()

        # Build command with all CLI arguments
        cmd = [
            sys.executable,
            "-m",
            "vox_serve",
            "--runtime",
            config.runtime,
            "--model",
            config.model,
            "--port",
            str(config.port),
            "--scheduler-type",
            config.scheduler_type,
            "--max-batch-size",
            str(config.max_batch_size),
            "--log-level",
            log_level,
        ]
        if config.max_num_pages:
            cmd.extend(["--max-num-pages", str(config.max_num_pages)])
        if config.page_size:
            cmd.extend(["--page-size", str(config.page_size)])
        if rust and config.max_context:
            cmd.extend(["--max-context", str(config.max_context)])

        # Async scheduling
        if config.async_scheduling and not rust:
            cmd.append("--async-scheduling")

        # Sampling parameters
        if config.top_p is not None:
            cmd.extend(["--top-p", str(config.top_p)])
        if config.top_k is not None:
            cmd.extend(["--top-k", str(config.top_k)])
        if config.min_p is not None:
            cmd.extend(["--min-p", str(config.min_p)])
        if config.temperature is not None:
            cmd.extend(["--temperature", str(config.temperature)])
        if config.max_tokens is not None:
            cmd.extend(["--max-tokens", str(config.max_tokens)])
        if config.repetition_penalty is not None:
            cmd.extend(["--repetition-penalty", str(config.repetition_penalty)])
        if config.repetition_window is not None and not rust:
            cmd.extend(["--repetition-window", str(config.repetition_window)])
        if config.cfg_scale is not None:
            cmd.extend(["--cfg-scale", str(config.cfg_scale)])
        if config.greedy:
            cmd.append("--greedy")

        # Performance options
        if not config.enable_cuda_graph and not rust:
            cmd.append("--disable-cuda-graph")
        if config.enable_disaggregation:
            cmd.append("--enable-disaggregation")
        if config.dp_size > 1:
            cmd.extend(["--dp-size", str(config.dp_size)])
        if config.enable_nvtx and not rust:
            cmd.append("--enable-nvtx")
        if config.enable_torch_compile:
            cmd.append("--enable-torch-compile")

        # Other
        if config.detokenize_interval is not None:
            cmd.extend(["--detokenize-interval", str(config.detokenize_interval)])

        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.config = config
            self.start_time = time.time()
            self._log_lines = [" ".join(cmd[2:])]
            self._ready = False

            # Start log reader thread
            def read_logs():
                if self.process and self.process.stdout:
                    for line in self.process.stdout:
                        self._log_lines.append(_ANSI.sub("", line.rstrip()))
                        if len(self._log_lines) > self._max_log_lines:
                            self._log_lines.pop(0)

            log_thread = threading.Thread(target=read_logs, daemon=True)
            log_thread.start()

            # Wait for server to be ready
            health_url = f"http://localhost:{config.port}/health"
            start_wait = time.time()

            while time.time() - start_wait < timeout:
                # Check if process died
                if self.process.poll() is not None:
                    return (
                        False,
                        f"Server process exited with code {self.process.returncode}",
                    )

                # Check health endpoint
                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        self._ready = True
                        return (
                            True,
                            f"Server started successfully on port {config.port}",
                        )
                except requests.exceptions.RequestException:
                    pass

                time.sleep(1)

            # Timeout - kill process
            self.stop()
            return False, f"Server failed to start within {timeout} seconds"

        except Exception as e:
            self.process = None
            self.config = None
            self.start_time = None
            return False, f"Failed to start server: {str(e)}"

    def stop(self) -> tuple[bool, str]:
        """
        Gracefully terminate the server process.

        Returns:
            Tuple of (success, message)
        """
        if self.process is None:
            return True, "No server running"

        if self.process.poll() is not None:
            self.process = None
            self.config = None
            self.start_time = None
            return True, "Server was already stopped"

        try:
            # Send SIGTERM for graceful shutdown
            self.process.terminate()

            # Wait for process to exit
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.process.kill()
                self.process.wait(timeout=5)

            self.process = None
            self.config = None
            self.start_time = None
            return True, "Server stopped successfully"

        except Exception as e:
            return False, f"Failed to stop server: {str(e)}"

    def get_status(self) -> ServerStatus:
        """Return current server status."""
        if self.process is None or self.process.poll() is not None:
            return ServerStatus(running=False)

        uptime = None
        if self.start_time:
            uptime = time.time() - self.start_time

        return ServerStatus(
            running=True,
            ready=self._ready,
            runtime=self.config.runtime if self.config else None,
            scheduler_type=self.config.scheduler_type if self.config else None,
            model=self.config.model if self.config else None,
            port=self.config.port if self.config else None,
            cuda_devices=self.config.cuda_devices if self.config else None,
            start_time=self.start_time,
            uptime_seconds=uptime,
        )

    def get_logs(self, lines: int = 100) -> List[str]:
        """Get recent log lines from the server process."""
        return self._log_lines[-lines:]
