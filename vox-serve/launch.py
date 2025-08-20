import io
import json
import uuid
import wave
import time
import subprocess
import signal
import atexit
import threading
from typing import Optional, Dict, List, Iterator
from pathlib import Path
from collections import defaultdict, deque

import zmq
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .scheduler import Scheduler
from .requests import Request
from .sampling import SamplingConfig
from .utils import get_logger

# Module-level logger
logger = get_logger(__name__)

def run_scheduler_daemon(model_name, request_socket_path, result_socket_path):
    """Function to run scheduler in daemon subprocess"""
    logger = get_logger(__name__)
    scheduler = Scheduler(
        model_name_or_path=model_name,
        request_socket_path=request_socket_path,
        result_socket_path=result_socket_path
    )
    logger.info(f"Scheduler started successfully with model: {model_name}")
    scheduler.run_forever()
    

class APIServer:
    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_reqult.ipc",
        output_dir: str = "/tmp/vox_serve_audio",
        timeout_seconds: float = 30.0
    ):
        self.model_name = model_name
        self.request_socket_path = request_socket_path
        self.result_socket_path = result_socket_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.upload_dir = Path(output_dir) / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.scheduler_process = None
        self.logger = get_logger(__name__)
        
        # Concurrent request tracking
        self.pending_requests: Dict[str, Dict] = {}  # request_id -> {chunks: [], event: threading.Event()}
        self.request_lock = threading.Lock()
        self.running = True
        
        # Start scheduler process
        self._start_scheduler()
        
        # Wait a moment for scheduler to initialize
        time.sleep(2)
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.PUSH)
        self.result_socket = self.context.socket(zmq.PULL)
        
        # Connect to scheduler sockets
        self.request_socket.connect(f"ipc://{request_socket_path}")
        self.result_socket.connect(f"ipc://{result_socket_path}")
        
        # Set socket timeouts for faster shutdown
        self.result_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for message processing
        self.request_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
        # Start background message processing thread
        self.message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _start_scheduler(self):
        """Start the scheduler process"""
        try:
            import multiprocessing as mp
            
            # Create and start scheduler process
            self.scheduler_process = mp.Process(
                target=run_scheduler_daemon,
                args=(self.model_name, self.request_socket_path, self.result_socket_path),
                daemon=True
            )
            self.scheduler_process.start()
            
            self.logger.info(f"Started scheduler process with PID: {self.scheduler_process.pid}")
            
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise RuntimeError(f"Could not start scheduler process: {e}")
    
    def _process_messages(self):
        """Background thread to process incoming messages from scheduler"""
        while self.running:
            try:
                # Receive message from scheduler (request_id|TYPE|data)
                message = self.result_socket.recv()
                
                # Parse message format: request_id|TYPE|data
                parts = message.split(b'|', 2)
                if len(parts) >= 3:
                    request_id = parts[0].decode('utf-8')
                    message_type = parts[1].decode('utf-8')
                    data = parts[2]
                    
                    # Route message to the appropriate request
                    with self.request_lock:
                        if request_id in self.pending_requests:
                            if message_type == 'AUDIO':
                                # Handle audio chunk
                                self.pending_requests[request_id]['chunks'].append(data)
                            elif message_type == 'COMPLETION':
                                # Handle completion notification
                                completion_info = json.loads(data.decode('utf-8'))
                                self.logger.info(f"Request {request_id} completed: {completion_info}")
                                self.pending_requests[request_id]['event'].set()
                
            except zmq.Again:
                # Timeout, continue loop
                continue
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.error(f"Error in message processing: {e}")
                continue
    
    def _stop_scheduler(self):
        if self.scheduler_process and self.scheduler_process.is_alive():
            self.logger.info("Stopping scheduler process...")
            try:
                self.scheduler_process.terminate()
                self.scheduler_process.join(timeout=1)  # Reduced timeout
                if self.scheduler_process.is_alive():
                    self.logger.warning("Scheduler didn't terminate gracefully, forcing kill...")
                    self.scheduler_process.kill()
                    self.scheduler_process.join(timeout=1)  # Quick final join
            except Exception as e:
                self.logger.error(f"Error stopping scheduler: {e}")
            self.logger.info("Scheduler process stopped")
    
    def stream_audio_chunks(self, text: str = None, voice: str = "tara", audio_path: str = None) -> Iterator[bytes]:
        """
        Generate audio from text and yield audio chunks as they arrive.
        
        Args:
            text: Input text to synthesize (optional if audio_path provided)
            voice: Voice to use for synthesis
            audio_path: Path to input audio file (optional)
            
        Yields:
            Audio chunks as bytes
            
        Raises:
            HTTPException: If generation fails or times out
        """
        request_id = str(uuid.uuid4())
        
        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {
                'chunks': [],
                'event': completion_event,
                'streaming': True,
                'consumed_chunks': 0
            }
        
        try:
            # Serialize Request object to JSON
            request_dict = {
                'request_id': request_id,
                'prompt': text,
                'voice': voice,
                'audio_path': audio_path,
            }
            
            request_json = json.dumps(request_dict)
            message = f"{request_json}|audio_data_placeholder".encode('utf-8')
            
            # Send request to scheduler
            self.request_socket.send(message)
            
            # Stream chunks as they arrive
            start_time = time.time()
            while not completion_event.is_set():
                if time.time() - start_time > self.timeout_seconds:
                    raise HTTPException(status_code=500, detail="Generation timed out")
                
                # Check for new chunks
                with self.request_lock:
                    request_data = self.pending_requests.get(request_id)
                    if request_data:
                        available_chunks = len(request_data['chunks'])
                        consumed = request_data['consumed_chunks']
                        
                        # Yield any new chunks
                        for i in range(consumed, available_chunks):
                            yield request_data['chunks'][i]
                        
                        request_data['consumed_chunks'] = available_chunks
                
                # Small sleep to avoid busy waiting
                time.sleep(0.01)
            
            # Yield any remaining chunks after completion
            with self.request_lock:
                request_data = self.pending_requests.get(request_id)
                if request_data:
                    consumed = request_data['consumed_chunks']
                    for i in range(consumed, len(request_data['chunks'])):
                        yield request_data['chunks'][i]
                    del self.pending_requests[request_id]
            
        except Exception as e:
            # Clean up on error
            with self.request_lock:
                self.pending_requests.pop(request_id, None)
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    def generate_audio(self, text: str = None, voice: str = "tara", audio_path: str = None) -> str:
        """
        Generate audio from text and return path to the audio file.
        
        Args:
            text: Input text to synthesize (optional if audio_path provided)
            voice: Voice to use for synthesis
            audio_path: Path to input audio file (optional)
            
        Returns:
            Path to the generated audio file
            
        Raises:
            HTTPException: If generation fails or times out
        """
        request_id = str(uuid.uuid4())
        
        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {
                'chunks': [],
                'event': completion_event
            }
        
        try:
            # Serialize Request object to JSON
            request_dict = {
                'request_id': request_id,
                'prompt': text,
                'voice': voice,
                'audio_path': audio_path,
            }
            
            request_json = json.dumps(request_dict)
            message = f"{request_json}|audio_data_placeholder".encode('utf-8')
            
            # Send request to scheduler
            self.request_socket.send(message)
            
            # Wait for completion or timeout
            if not completion_event.wait(timeout=self.timeout_seconds):
                raise HTTPException(status_code=500, detail="Generation timed out")
            
            # Retrieve collected audio chunks
            with self.request_lock:
                audio_chunks = self.pending_requests[request_id]['chunks'][:]
                del self.pending_requests[request_id]
            
            if not audio_chunks:
                raise HTTPException(status_code=500, detail="No audio generated")
            
            # Save to WAV file
            output_file = self.output_dir / f"{request_id}.wav"
            with wave.open(str(output_file), 'wb') as wf:
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
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def cleanup(self):
        """Clean up ZMQ resources and stop scheduler"""
        self.logger.info("Cleaning up API server...")
        
        # Stop background message processing
        self.running = False
        if hasattr(self, 'message_thread') and self.message_thread.is_alive():
            self.message_thread.join(timeout=1)
        
        try:
            if hasattr(self, 'request_socket'):
                self.request_socket.close()
            if hasattr(self, 'result_socket'):
                self.result_socket.close()
            if hasattr(self, 'context'):
                self.context.term(linger=0)  # Don't wait for pending messages
        except Exception as e:
            self.logger.error(f"Error cleaning up ZMQ: {e}")
        
        self._stop_scheduler()


# Initialize FastAPI app
app = FastAPI(title="Vox-Serve API", description="Text-to-Speech API using Orpheus model")

# Global API server instance (will be initialized in main)
api_server = None


@app.post("/generate")
async def generate(
    text: str = Form(...),
    voice: Optional[str] = Form("tara"),
    audio: Optional[UploadFile] = File(None)
):
    """
    Generate speech from text and return audio file directly.
    
    Args:
        text: Input text to synthesize
        voice: Voice to use for synthesis
        audio: Optional input audio file
        
    Returns:
        Audio file as direct response
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    audio_path = None
    if audio:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = str(api_server.upload_dir / audio_filename)
        
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
    
    try:
        audio_file = api_server.generate_audio(text, voice, audio_path)
        request_id = Path(audio_file).stem
        
        return FileResponse(
            path=audio_file,
            media_type="audio/wav",
            filename=f"{request_id}.wav"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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


@app.post("/generate-stream")
async def generate_stream(
    text: str = Form(...),
    voice: Optional[str] = Form("tara"),
    audio: Optional[UploadFile] = File(None)
):
    """
    Generate speech from text and stream audio chunks as they are generated.
    
    Args:
        text: Input text to synthesize
        voice: Voice to use for synthesis
        audio: Optional input audio file
        
    Returns:
        Streaming audio response with WAV format chunks
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    audio_path = None
    if audio:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = str(api_server.upload_dir / audio_filename)
        
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
    
    try:
        def audio_stream():
            # WAV header for 24kHz mono 16-bit audio
            wav_header = io.BytesIO()
            with wave.open(wav_header, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # 24kHz
                wf.writeframes(b'')  # Empty data for header
            
            # Get header bytes and correct the chunk size for streaming
            wav_header.seek(0)
            header_bytes = wav_header.read()
            
            # Send WAV header first
            yield header_bytes
            
            # Stream audio chunks
            for chunk in api_server.stream_audio_chunks(text, voice, audio_path):
                yield chunk
        
        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=stream_{uuid.uuid4().hex[:8]}.wav",
                "Cache-Control": "no-cache"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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


if __name__ == "__main__":
    import argparse
    import uvicorn
    import multiprocessing as mp
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vox-Serve Text-to-Speech API Server")
    parser.add_argument(
        "--model", 
        type=str, 
        default="canopylabs/orpheus-3b-0.1-ft",
        help="Model name or path to use for text-to-speech synthesis (default: canopylabs/orpheus-3b-0.1-ft)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    args = parser.parse_args()
    
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Initialize API server instance with specified model
    api_server = APIServer(model_name=args.model)
    
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