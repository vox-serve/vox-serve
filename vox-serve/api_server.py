import io
import json
import uuid
import wave
import time
import subprocess
import signal
import atexit
from typing import Optional
from pathlib import Path

import zmq
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .scheduler import Scheduler
from .requests import Request
from .sampling import SamplingConfig

def run_scheduler_daemon(model_name, request_socket_path, result_socket_path):
    """Function to run scheduler in daemon subprocess"""
    scheduler = Scheduler(
        model_name_or_path=model_name,
        request_socket_path=request_socket_path,
        result_socket_path=result_socket_path
    )
    print(f"Scheduler started successfully with model: {model_name}")
    scheduler.run_forever()


class GenerateRequest(BaseModel):
    text: str
    voice: Optional[str] = "tara"
    

    

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
        self.timeout_seconds = timeout_seconds
        self.scheduler_process = None
        
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
        self.result_socket.setsockopt(zmq.RCVTIMEO, int(timeout_seconds * 1000))
        self.request_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        
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
            
            print(f"Started scheduler process with PID: {self.scheduler_process.pid}")
            
        except Exception as e:
            print(f"Failed to start scheduler: {e}")
            raise RuntimeError(f"Could not start scheduler process: {e}")
    
    def _stop_scheduler(self):
        """Stop the scheduler process"""
        if self.scheduler_process and self.scheduler_process.is_alive():
            print("Stopping scheduler process...")
            try:
                self.scheduler_process.terminate()
                self.scheduler_process.join(timeout=1)  # Reduced timeout
                if self.scheduler_process.is_alive():
                    print("Scheduler didn't terminate gracefully, forcing kill...")
                    self.scheduler_process.kill()
                    self.scheduler_process.join(timeout=1)  # Quick final join
            except Exception as e:
                print(f"Error stopping scheduler: {e}")
            print("Scheduler process stopped")
    
    def generate_audio(self, text: str, voice: str = "tara") -> str:
        """
        Generate audio from text and return path to the audio file.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            
        Returns:
            Path to the generated audio file
            
        Raises:
            HTTPException: If generation fails or times out
        """
        request_id = str(uuid.uuid4())
        
        # Serialize Request object to JSON
        request_dict = {
            'request_id': request_id,
            'prompt': text,
            'voice': voice,
        }
        
        request_json = json.dumps(request_dict)
        message = f"{request_json}|audio_data_placeholder".encode('utf-8')
        
        try:
            # Send request to scheduler
            self.request_socket.send(message)
            
            # Collect audio chunks
            audio_chunks = []
            start_time = time.time()
            
            while True:
                try:
                    # Receive audio chunk from scheduler
                    audio_bytes = self.result_socket.recv()
                    
                    # Convert bytes back to numpy array (int16, 2048 samples)
                    # audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_chunks.append(audio_bytes)
                    
                    # Check for timeout
                    if time.time() - start_time > self.timeout_seconds:
                        break
                        
                except zmq.Again:
                    # Timeout occurred, assume generation is complete
                    break
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error receiving audio: {str(e)}")
            
            if not audio_chunks:
                raise HTTPException(status_code=500, detail="No audio generated")
            
            # Concatenate all audio chunks
            # full_audio = np.concatenate(audio_chunks)
            
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
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def cleanup(self):
        """Clean up ZMQ resources and stop scheduler"""
        print("Cleaning up API server...")
        try:
            if hasattr(self, 'request_socket'):
                self.request_socket.close()
            if hasattr(self, 'result_socket'):
                self.result_socket.close()
            if hasattr(self, 'context'):
                self.context.term(linger=0)  # Don't wait for pending messages
        except Exception as e:
            print(f"Error cleaning up ZMQ: {e}")
        
        self._stop_scheduler()


# Initialize FastAPI app
app = FastAPI(title="Vox-Serve API", description="Text-to-Speech API using Orpheus model")

# Global API server instance (will be initialized in main)
api_server = None


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate speech from text and return audio file directly.
    
    Args:
        request: Generation request containing text and optional voice
        
    Returns:
        Audio file as direct response
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        audio_file = api_server.generate_audio(request.text, request.voice)
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
    print(f"\nReceived signal {signum}, shutting down...")
    if api_server is not None:
        api_server.cleanup()
    import os
    os._exit(0)  # Force immediate exit


if __name__ == "__main__":
    import uvicorn
    import multiprocessing as mp
    
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Initialize API server instance
    api_server = APIServer()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting vox-serve API server...")
        print("Scheduler and API server will be available shortly...")
        uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        if api_server is not None:
            api_server.cleanup()