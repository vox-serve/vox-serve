"""FastAPI backend for VoxServe Playground."""

import argparse
import struct
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from server_manager import ServerConfig, VoxServeServerManager
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI(title="VoxServe Playground")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Global server manager instance
server_manager = VoxServeServerManager()


# Pydantic models for API
class ServerStartRequest(BaseModel):
    """Request body for starting the server (mirrors all CLI arguments)."""

    # Model and server
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    port: int = 12345
    cuda_devices: List[int] = [0]

    # Scheduler
    scheduler_type: str = "base"
    async_scheduling: bool = False

    # Batch and memory
    max_batch_size: int = 8
    max_num_pages: int = 2048
    page_size: int = 128

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
    log_level: str = "INFO"
    detokenize_interval: Optional[int] = None


class ServerStartResponse(BaseModel):
    """Response from server start."""

    success: bool
    message: str


class ServerStatusResponse(BaseModel):
    """Response for server status."""

    running: bool
    model: Optional[str] = None
    port: Optional[int] = None
    cuda_devices: Optional[List[int]] = None
    uptime_seconds: Optional[float] = None


class GPUInfoResponse(BaseModel):
    """GPU information response."""

    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    name: str
    supports_audio_input: bool = False
    requires_audio: bool = False
    supports_language: bool = False
    supports_speaker: bool = False
    supports_ref_text: bool = False
    supports_instruct: bool = False


class ModelsResponse(BaseModel):
    """Response for models list."""

    models: List[ModelInfo]


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main playground page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/gpus", response_model=List[GPUInfoResponse])
async def list_gpus():
    """List available NVIDIA GPUs."""
    gpus = server_manager.get_available_gpus()
    return [asdict(gpu) for gpu in gpus]


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """List supported models with their capabilities (full HuggingFace IDs).

    Qwen3-TTS model capabilities:
    - All models support language selection
    - 1.7B models support instruct (voice style instructions)
    - CustomVoice: uses predefined speaker IDs
    - Base: supports voice cloning with optional reference audio + ref_text
    - VoiceDesign: voice controlled primarily by instruct text
    """
    models = [
        # Qwen3-TTS models (prioritized, default is CustomVoice 1.7B)
        # CustomVoice 1.7B: speaker + language + instruct
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            supports_language=True,
            supports_speaker=True,
            supports_instruct=True,
        ),
        # Base 1.7B: voice cloning with optional audio + ref_text + instruct
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            supports_audio_input=True,
            supports_language=True,
            supports_ref_text=True,
            supports_instruct=True,
        ),
        # VoiceDesign 1.7B: language + instruct (voice from instruct)
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            supports_language=True,
            supports_instruct=True,
        ),
        # CustomVoice 0.6B: speaker + language (no instruct for 0.6B)
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            supports_language=True,
            supports_speaker=True,
        ),
        # Base 0.6B: voice cloning with optional audio + ref_text (no instruct)
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            supports_audio_input=True,
            supports_language=True,
            supports_ref_text=True,
        ),
        # VoiceDesign 0.6B: language only (no instruct for 0.6B)
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            name="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            supports_language=True,
        ),
        # Standard models
        ModelInfo(
            id="canopylabs/orpheus-3b-0.1-ft",
            name="canopylabs/orpheus-3b-0.1-ft",
            supports_audio_input=True,
        ),
        ModelInfo(
            id="sesame/csm-1b",
            name="sesame/csm-1b",
        ),
        ModelInfo(
            id="Zyphra/Zonos-v0.1-transformer",
            name="Zyphra/Zonos-v0.1-transformer",
        ),
        ModelInfo(
            id="zai-org/glm-4-voice-9b",
            name="zai-org/glm-4-voice-9b",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="stepfun-ai/Step-Audio-2-mini",
            name="stepfun-ai/Step-Audio-2-mini",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="ResembleAI/chatterbox",
            name="ResembleAI/chatterbox",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="FunAudioLLM/CosyVoice2-0.5B",
            name="FunAudioLLM/CosyVoice2-0.5B",
            supports_audio_input=True,
        ),
    ]
    return ModelsResponse(models=models)


@app.post("/api/server/start", response_model=ServerStartResponse)
async def start_server(request: ServerStartRequest):
    """Start the VoxServe server with full CLI configuration."""
    config = ServerConfig(
        model=request.model,
        port=request.port,
        cuda_devices=request.cuda_devices,
        scheduler_type=request.scheduler_type,
        async_scheduling=request.async_scheduling,
        max_batch_size=request.max_batch_size,
        max_num_pages=request.max_num_pages,
        page_size=request.page_size,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        repetition_window=request.repetition_window,
        cfg_scale=request.cfg_scale,
        greedy=request.greedy,
        enable_cuda_graph=request.enable_cuda_graph,
        enable_disaggregation=request.enable_disaggregation,
        dp_size=request.dp_size,
        enable_nvtx=request.enable_nvtx,
        enable_torch_compile=request.enable_torch_compile,
        log_level=request.log_level,
        detokenize_interval=request.detokenize_interval,
    )
    success, message = server_manager.start(config)
    return ServerStartResponse(success=success, message=message)


@app.post("/api/server/stop", response_model=ServerStartResponse)
async def stop_server():
    """Stop the VoxServe server."""
    success, message = server_manager.stop()
    return ServerStartResponse(success=success, message=message)


@app.get("/api/server/status", response_model=ServerStatusResponse)
async def server_status():
    """Get current server status."""
    status = server_manager.get_status()
    return ServerStatusResponse(
        running=status.running,
        model=status.model,
        port=status.port,
        cuda_devices=status.cuda_devices,
        uptime_seconds=status.uptime_seconds,
    )


@app.get("/api/server/logs")
async def server_logs(lines: int = 200):
    """Get recent server logs."""
    return {"logs": server_manager.get_logs(lines)}


@app.post("/api/generate")
async def generate_audio(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    streaming: bool = Form(True),
    language: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
):
    """
    Proxy audio generation request to the VoxServe server.

    This endpoint forwards the request to the running VoxServe server
    and streams the response back to the client.
    """
    status = server_manager.get_status()
    if not status.running:
        return {"error": "Server is not running"}

    voxserve_url = f"http://localhost:{status.port}/generate"

    # Build form data for the upstream request
    form_data = aiohttp.FormData()
    form_data.add_field("text", text)
    form_data.add_field("streaming", "true" if streaming else "false")

    if audio:
        audio_content = await audio.read()
        form_data.add_field(
            "audio",
            audio_content,
            filename=audio.filename,
            content_type=audio.content_type or "audio/wav",
        )

    if language:
        form_data.add_field("language", language)
    if speaker:
        form_data.add_field("speaker", speaker)
    if ref_text:
        form_data.add_field("ref_text", ref_text)
    if instruct:
        form_data.add_field("instruct", instruct)

    async def stream_response():
        """Stream the audio response from VoxServe with TTFA measurement.

        Protocol: First 4 bytes are TTFA in milliseconds (uint32 little-endian),
        followed by the WAV data.

        We buffer the first chunk (WAV header), wait for the second chunk (first audio),
        measure TTFA, then send: TTFA prefix + buffered header + audio chunks.
        """
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_start = time.perf_counter()
            async with session.post(voxserve_url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VoxServe error: {response.status} - {error_text}")

                chunk_count = 0
                header_chunk = None
                ttfa_sent = False

                async for chunk in response.content.iter_any():
                    chunk_count += 1

                    if chunk_count == 1:
                        # Buffer the first chunk (WAV header)
                        header_chunk = chunk
                        continue

                    if chunk_count == 2 and not ttfa_sent:
                        # Second chunk = first audio data, measure TTFA now
                        ttfa_ms = int((time.perf_counter() - request_start) * 1000)
                        # Send TTFA prefix first
                        yield struct.pack("<I", ttfa_ms)
                        # Then send buffered header
                        if header_chunk:
                            yield header_chunk
                        ttfa_sent = True

                    yield chunk

                # If we only got header chunk, still send TTFA and header
                if not ttfa_sent:
                    ttfa_ms = int((time.perf_counter() - request_start) * 1000)
                    yield struct.pack("<I", ttfa_ms)
                    if header_chunk:
                        yield header_chunk

    return StreamingResponse(
        stream_response(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
        },
    )


# ============================================================================
# Input Streaming Endpoints (for LLM Chat mode)
# ============================================================================


class InputStreamStartRequest(BaseModel):
    """Request body for starting an input stream."""

    speaker: Optional[str] = None
    language: Optional[str] = None


class InputStreamStartResponse(BaseModel):
    """Response from starting an input stream."""

    request_id: str


@app.post("/api/input-stream/start", response_model=InputStreamStartResponse)
async def input_stream_start(
    speaker: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """Start an input streaming request to the VoxServe server."""
    status = server_manager.get_status()
    if not status.running:
        raise HTTPException(status_code=400, detail="Server is not running")

    voxserve_url = f"http://localhost:{status.port}/generate/stream/start"

    # Build form data only if we have parameters (matching input_streaming.py)
    form_data = None
    if speaker or language:
        form_data = aiohttp.FormData()
        if speaker:
            form_data.add_field("speaker", speaker)
        if language:
            form_data.add_field("language", language)

    async with aiohttp.ClientSession() as session:
        async with session.post(voxserve_url, data=form_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"VoxServe error: {error_text}",
                )
            result = await response.json()
            return InputStreamStartResponse(request_id=result["request_id"])


@app.post("/api/input-stream/{request_id}/text")
async def input_stream_text(
    request_id: str,
    text: str = Form(...),
):
    """Send a text chunk to the input stream."""
    status = server_manager.get_status()
    if not status.running:
        raise HTTPException(status_code=400, detail="Server is not running")

    voxserve_url = f"http://localhost:{status.port}/generate/stream/{request_id}/text"

    form_data = aiohttp.FormData()
    form_data.add_field("text", text)

    async with aiohttp.ClientSession() as session:
        async with session.post(voxserve_url, data=form_data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"VoxServe error: {error_text}",
                )
            return {"success": True}


@app.get("/api/input-stream/{request_id}/audio")
async def input_stream_audio(request_id: str):
    """Stream audio chunks as they are generated (concurrent with text input)."""
    status = server_manager.get_status()
    if not status.running:
        raise HTTPException(status_code=400, detail="Server is not running")

    voxserve_url = f"http://localhost:{status.port}/generate/stream/{request_id}/audio"

    async def stream_response():
        """Stream audio chunks from VoxServe with TTFA prefix."""
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_start = time.perf_counter()
            async with session.get(voxserve_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VoxServe error: {response.status} - {error_text}")

                first_chunk = True
                async for chunk in response.content.iter_any():
                    if first_chunk:
                        # Send TTFA prefix before the first chunk
                        ttfa_ms = int((time.perf_counter() - request_start) * 1000)
                        yield struct.pack("<I", ttfa_ms)
                        first_chunk = False
                    yield chunk

    return StreamingResponse(
        stream_response(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
        },
    )


@app.post("/api/input-stream/{request_id}/end")
async def input_stream_end(request_id: str):
    """Signal end of text input."""
    status = server_manager.get_status()
    if not status.running:
        raise HTTPException(status_code=400, detail="Server is not running")

    voxserve_url = f"http://localhost:{status.port}/generate/stream/{request_id}/end"

    async with aiohttp.ClientSession() as session:
        async with session.post(voxserve_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"VoxServe error: {error_text}",
                )
            return {"success": True}


def main():
    """Run the playground server."""
    parser = argparse.ArgumentParser(description="VoxServe Playground")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    args = parser.parse_args()

    print(f"Starting VoxServe Playground on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
