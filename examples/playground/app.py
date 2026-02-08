"""FastAPI backend for VoxServe Playground."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
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
    """Request body for starting the server."""

    model: str = "canopylabs/orpheus-3b-0.1-ft"
    port: int = 8000
    cuda_devices: List[int] = [0]
    max_batch_size: int = 8
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_cuda_graph: bool = True


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
    """List supported models with their capabilities."""
    models = [
        ModelInfo(
            id="orpheus",
            name="Orpheus-3B",
            supports_audio_input=True,
        ),
        ModelInfo(
            id="csm",
            name="CSM-1B",
        ),
        ModelInfo(
            id="zonos",
            name="Zonos-v0.1",
        ),
        ModelInfo(
            id="glm",
            name="GLM-4-Voice-9B",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="step",
            name="Step-Audio-2-Mini",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="chatterbox",
            name="Chatterbox",
            supports_audio_input=True,
            requires_audio=True,
        ),
        ModelInfo(
            id="cosyvoice2",
            name="CosyVoice2-0.5B",
            supports_audio_input=True,
        ),
        # Qwen3-TTS models from https://huggingface.co/collections/Qwen/qwen3-tts
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            name="Qwen3-TTS-1.7B-Base",
            supports_audio_input=True,
            supports_language=True,
            supports_speaker=True,
            supports_ref_text=True,
        ),
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            name="Qwen3-TTS-0.6B-Base",
            supports_audio_input=True,
            supports_language=True,
            supports_speaker=True,
            supports_ref_text=True,
        ),
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            name="Qwen3-TTS-1.7B-CustomVoice",
            supports_audio_input=True,
            supports_language=True,
            supports_speaker=True,
            supports_ref_text=True,
        ),
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            name="Qwen3-TTS-0.6B-CustomVoice",
            supports_audio_input=True,
            supports_language=True,
            supports_speaker=True,
            supports_ref_text=True,
        ),
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            name="Qwen3-TTS-1.7B-VoiceDesign",
            supports_audio_input=True,
            supports_language=True,
            supports_instruct=True,
        ),
        ModelInfo(
            id="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            name="Qwen3-TTS-0.6B-VoiceDesign",
            supports_audio_input=True,
            supports_language=True,
            supports_instruct=True,
        ),
    ]
    return ModelsResponse(models=models)


@app.post("/api/server/start", response_model=ServerStartResponse)
async def start_server(request: ServerStartRequest):
    """Start the VoxServe server."""
    config = ServerConfig(
        model=request.model,
        port=request.port,
        cuda_devices=request.cuda_devices,
        max_batch_size=request.max_batch_size,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        enable_cuda_graph=request.enable_cuda_graph,
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
async def server_logs(lines: int = 100):
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
        """Stream the audio response from VoxServe."""
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(voxserve_url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"VoxServe error: {response.status} - {error_text}")

                async for chunk in response.content.iter_any():
                    yield chunk

    return StreamingResponse(
        stream_response(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
        },
    )


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
