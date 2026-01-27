import io
import json
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import zmq

from .model import load_model
from .tokenizer.base import DecoderCache
from .utils import get_logger


def _serialize_cache(cache: DecoderCache) -> bytes:
    buffer = io.BytesIO()
    torch.save(cache, buffer)
    return buffer.getvalue()


def _deserialize_cache(payload: bytes) -> DecoderCache:
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu")


def _serialize_tokens(token_tensor: torch.Tensor) -> Tuple[bytes, str, Tuple[int, ...]]:
    tokens_cpu = token_tensor.detach().cpu().contiguous()
    tokens_np = tokens_cpu.numpy()
    return tokens_np.tobytes(), str(tokens_np.dtype), tokens_np.shape


def _deserialize_tokens(payload: bytes, dtype: str, shape: Tuple[int, ...]) -> torch.Tensor:
    tokens_np = np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(shape)
    return torch.from_numpy(tokens_np)


class RemoteDetokenizerClient:
    def __init__(
        self,
        token_endpoint: str,
        audio_endpoint: str,
        on_audio_chunk: Callable[[Dict, bytes], None],
        logger=None,
    ):
        self.logger = logger or get_logger(__name__)
        self.context = zmq.Context()
        self.token_socket = self.context.socket(zmq.PUSH)
        self.audio_socket = self.context.socket(zmq.PULL)
        self.token_socket.connect(token_endpoint)
        self.audio_socket.connect(audio_endpoint)

        # Reduce shutdown latency
        try:
            self.token_socket.setsockopt(zmq.LINGER, 0)
            self.audio_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        self._on_audio_chunk = on_audio_chunk
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        try:
            self.token_socket.close()
            self.audio_socket.close()
            self.context.term()
        except Exception:
            pass

    def send_token_chunk(
        self,
        request_id: str,
        chunk_id: int,
        token_tensor: torch.Tensor,
        token_count: int,
        is_final: bool,
        decoder_cache: Optional[DecoderCache] = None,
    ) -> None:
        token_payload, dtype, shape = _serialize_tokens(token_tensor)
        header = {
            "type": "TOKEN_CHUNK",
            "request_id": request_id,
            "chunk_id": chunk_id,
            "token_count": token_count,
            "is_final": is_final,
            "dtype": dtype,
            "shape": shape,
            "has_cache": decoder_cache is not None,
        }

        parts = [json.dumps(header).encode("utf-8"), token_payload]
        if decoder_cache is not None:
            cache_payload = _serialize_cache(decoder_cache.to("cpu"))
            parts.append(cache_payload)

        self.token_socket.send_multipart(parts)

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                parts = self.audio_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            except Exception as exc:
                self.logger.error("Remote detokenizer receive error: %s", exc)
                continue

            if len(parts) < 2:
                self.logger.warning("Malformed detokenizer response: %s parts", len(parts))
                continue

            try:
                header = json.loads(parts[0].decode("utf-8"))
            except Exception as exc:
                self.logger.error("Failed to parse detokenizer header: %s", exc)
                continue

            self._on_audio_chunk(header, parts[1])


class RemoteDetokenizerServer:
    def __init__(
        self,
        model_name: str,
        token_bind: str,
        audio_bind: str,
        device: str = "cuda:0",
        enable_torch_compile: bool = False,
        logger=None,
    ):
        self.logger = logger or get_logger(__name__)
        self.context = zmq.Context()
        self.token_socket = self.context.socket(zmq.PULL)
        self.audio_socket = self.context.socket(zmq.PUSH)
        self.token_socket.bind(token_bind)
        self.audio_socket.bind(audio_bind)

        try:
            self.token_socket.setsockopt(zmq.RCVHWM, 256)
            self.audio_socket.setsockopt(zmq.SNDHWM, 1024)
            self.token_socket.setsockopt(zmq.LINGER, 0)
            self.audio_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        # Load full model for postprocess (audio decoder)
        self.model = load_model(
            model_name,
            device=device,
            enable_torch_compile=enable_torch_compile,
            audio_decoder_device=device,
        )
        self.device = device
        self.detokenize_interval = self.model.detokenize_interval
        self.needs_watermarking = self.model.needs_watermarking
        self.watermarker_type = self.model.watermarker_type if self.needs_watermarking else None
        self._init_watermark_models()

        self._decoder_caches: Dict[str, DecoderCache] = {}

    def _init_watermark_models(self):
        if not self.needs_watermarking:
            self.watermark_model = None
            return

        if self.watermarker_type == "silentcipher":
            from .watermarker import silentcipher

            self.watermark_model = silentcipher.get_model(
                model_type="44.1k",
                device=self.device,
            )
            self.watermark_key = [11, 91, 60, 147, 209]
        elif self.watermarker_type == "parth":
            from .watermarker.perth import PerthImplicitWatermarker

            self.watermark_model = PerthImplicitWatermarker(device=self.device)
        else:
            raise ValueError(f"Unknown watermarker type: {self.watermarker_type}")

    def _run_watermark(self, audio_tensor: torch.Tensor, orig_sr: int = 24000) -> torch.Tensor:
        import torchaudio

        if self.watermarker_type == "silentcipher":
            audio_array_44khz = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=orig_sr,
                new_freq=self.watermark_model.sr,
            )
            encoded = self.watermark_model.encode_wav(
                audio_array_44khz,
                self.watermark_model.sr,
                self.watermark_key,
            )
            encoded = torchaudio.functional.resample(
                encoded,
                orig_freq=self.watermark_model.sr,
                new_freq=orig_sr,
            )
            return encoded

        if self.watermarker_type == "parth":
            audio_array_32khz = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=orig_sr,
                new_freq=self.watermark_model.sr,
            )
            encoded = self.watermark_model.encode_wav(
                audio_array_32khz,
                self.watermark_model.sr,
            )
            encoded = torch.from_numpy(encoded).to(audio_tensor.device)
            encoded = torchaudio.functional.resample(
                encoded,
                orig_freq=self.watermark_model.sr,
                new_freq=orig_sr,
            )
            return encoded

        return audio_tensor

    def _handle_token_chunk(self, header: Dict, token_payload: bytes, cache_payload: Optional[bytes]):
        request_id = header["request_id"]
        token_count = int(header["token_count"])
        is_final = bool(header["is_final"])
        dtype = header["dtype"]
        shape = tuple(header["shape"])

        token_tensor = _deserialize_tokens(token_payload, dtype, shape).to(self.device)
        if token_tensor.dim() == 2:
            token_tensor = token_tensor.unsqueeze(0)

        if cache_payload is not None:
            cache = _deserialize_cache(cache_payload)
            self._decoder_caches[request_id] = cache.to(self.device)

        decoder_cache = self._decoder_caches.get(request_id)

        if decoder_cache is not None:
            audio_tensors = self.model.postprocess(token_tensor, decoder_cache=decoder_cache)
        else:
            audio_tensors = self.model.postprocess(token_tensor)

        if self.needs_watermarking:
            for i in range(audio_tensors.shape[0]):
                audio_tensors[i, 0] = self._run_watermark(audio_tensors[i, 0], orig_sr=24000)

        audio = audio_tensors[0].detach().cpu().numpy()
        audio_int16 = (audio * 32767).astype(np.int16)

        if token_count < self.detokenize_interval:
            trim_len = int(audio_int16.shape[1] * token_count / self.detokenize_interval)
            audio_int16 = audio_int16[:trim_len]

        audio_bytes = audio_int16.tobytes()
        response_header = {
            "type": "AUDIO_CHUNK",
            "request_id": request_id,
            "chunk_id": header["chunk_id"],
            "is_final": is_final,
        }
        self.audio_socket.send_multipart([json.dumps(response_header).encode("utf-8"), audio_bytes])

        if is_final and request_id in self._decoder_caches:
            del self._decoder_caches[request_id]

    def run_forever(self):
        self.logger.info(
            "Remote detokenizer listening (tokens: %s, audio: %s)",
            self.token_socket.getsockopt_string(zmq.LAST_ENDPOINT),
            self.audio_socket.getsockopt_string(zmq.LAST_ENDPOINT),
        )
        while True:
            parts = self.token_socket.recv_multipart()
            if len(parts) < 2:
                self.logger.warning("Malformed token chunk: %s parts", len(parts))
                continue

            header = json.loads(parts[0].decode("utf-8"))
            if header.get("type") != "TOKEN_CHUNK":
                self.logger.warning("Unknown message type: %s", header.get("type"))
                continue

            token_payload = parts[1]
            cache_payload = parts[2] if len(parts) > 2 else None
            self._handle_token_chunk(header, token_payload, cache_payload)
