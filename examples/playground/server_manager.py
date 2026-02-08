"""VoxServe server subprocess lifecycle management."""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


@dataclass
class ServerConfig:
    """Configuration for VoxServe server launch."""

    model: str = "canopylabs/orpheus-3b-0.1-ft"
    port: int = 8000
    cuda_devices: List[int] = field(default_factory=lambda: [0])
    max_batch_size: int = 8
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_cuda_graph: bool = True


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
    model: Optional[str] = None
    port: Optional[int] = None
    cuda_devices: Optional[List[int]] = None
    start_time: Optional[float] = None
    uptime_seconds: Optional[float] = None


class VoxServeServerManager:
    """Manages VoxServe server subprocess lifecycle."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.config: Optional[ServerConfig] = None
        self.start_time: Optional[float] = None
        self._log_lines: List[str] = []
        self._max_log_lines: int = 100

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
                            memory_total_gb=round(int(parts[2]) / 1024, 1),
                            memory_free_gb=round(int(parts[3]) / 1024, 1),
                        )
                    )
            return gpus
        except Exception:
            return []

    def start(self, config: ServerConfig, timeout: float = 60.0) -> tuple[bool, str]:
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

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "vox_serve.launch",
            "--model",
            config.model,
            "--port",
            str(config.port),
            "--max-batch-size",
            str(config.max_batch_size),
        ]

        # Add optional sampling parameters
        if config.top_p is not None:
            cmd.extend(["--top-p", str(config.top_p)])
        if config.top_k is not None:
            cmd.extend(["--top-k", str(config.top_k)])
        if config.temperature is not None:
            cmd.extend(["--temperature", str(config.temperature)])
        if config.max_tokens is not None:
            cmd.extend(["--max-tokens", str(config.max_tokens)])

        # CUDA graph setting
        if not config.enable_cuda_graph:
            cmd.append("--disable-cuda-graph")

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
            self._log_lines = []

            # Start log reader thread
            import threading

            def read_logs():
                if self.process and self.process.stdout:
                    for line in self.process.stdout:
                        self._log_lines.append(line.rstrip())
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
                    return False, f"Server process exited with code {self.process.returncode}"

                # Check health endpoint
                try:
                    resp = requests.get(health_url, timeout=2)
                    if resp.status_code == 200:
                        return True, f"Server started successfully on port {config.port}"
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
            model=self.config.model if self.config else None,
            port=self.config.port if self.config else None,
            cuda_devices=self.config.cuda_devices if self.config else None,
            start_time=self.start_time,
            uptime_seconds=uptime,
        )

    def get_logs(self, lines: int = 50) -> List[str]:
        """Get recent log lines from the server process."""
        return self._log_lines[-lines:]
