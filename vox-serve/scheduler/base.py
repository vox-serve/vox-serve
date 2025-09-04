import json
import time
from typing import List

import torch
import zmq

from ..requests import Request
from ..utils import get_logger
from ..worker import CudaGraphWorker, ModelWorker


class Scheduler:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cuda"),
        max_batch_size: int = 8,
        max_num_pages: int = 1024,
        page_size: int = 128,
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_result.ipc",
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_cuda_graph: bool = True,
        enable_nvtx: bool = False,
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.logger = get_logger(__name__)

        # Switch between CudaGraphWorker and ModelWorker based on user input
        if enable_cuda_graph:
            self.logger.info("Using CudaGraphWorker with CUDA graph optimization")
            self.model_worker = CudaGraphWorker(
                model_name_or_path,
                max_batch_size=max_batch_size,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                cfg_scale=cfg_scale,
                greedy=greedy,
                max_num_pages=max_num_pages,
                page_size=page_size,
                enable_nvtx=enable_nvtx,
            )
        else:
            self.logger.info("Using ModelWorker without CUDA graph optimization")
            self.model_worker = ModelWorker(
                model_name_or_path,
                max_batch_size=max_batch_size,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                cfg_scale=cfg_scale,
                greedy=greedy,
                max_num_pages=max_num_pages,
                page_size=page_size,
                enable_nvtx=enable_nvtx,
            )

        self.active_requests: List[Request] = []

        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(f"ipc://{request_socket_path}")
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(f"ipc://{result_socket_path}")

        # self.available_batch_sizes = self.model_worker.available_batch_sizes
        self.available_batch_sizes = [16, 32, 64, 128]

    def _step(self):
        """
        Process the next batch of requests.
        """

        # insert/remove requests to self.active_requests
        self._prepare_requests()

        # Naive scheduling: take up to max_batch_size requests
        lm_requests = []

        for req in self.active_requests:
            if len(lm_requests) > self.max_batch_size:
                break
            if not req.done_lm_generation:
                lm_requests.append(req)

        is_prefill = False
        for req in lm_requests:
            is_prefill = is_prefill or (not req.done_lm_prefill)

        if len(lm_requests) == 0:
            time.sleep(0.1)
            return

        # Prepare LM inputs outside the worker and run either prefill or decode
        lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests)
        if is_prefill:
            self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
        else:
            self.model_worker.run_lm_decode(lm_requests, lm_inputs)

        detokenize_requests = []

        for req in self.active_requests:
            if len(detokenize_requests) > self.max_batch_size:
                break
            if req.done_lm_generation or self.model_worker.do_detokenize(req):
                detokenize_requests.append(req)

        # run detokenization if needed
        self.model_worker.run_detokenize(detokenize_requests)

        # return results to clients
        for req in detokenize_requests:
            while not req.output_audio.empty():
                # Send audio chunk message: request_id|AUDIO|audio_data
                message = req.request_id.encode("utf-8") + b"|AUDIO|" + req.output_audio.get()
                self.result_socket.send(message)

            # send completion notification for finished requests
            if req.done_all:
                self.model_worker.free_kv_cache(req)
                completion_message = {"status": "completed", "reason": req.finish_reason or "unknown"}
                # Send completion message: request_id|COMPLETION|json_data
                completion_payload = (
                    req.request_id.encode("utf-8") + b"|COMPLETION|" + json.dumps(completion_message).encode("utf-8")
                )
                self.logger.debug(f"Sending completion for request {req.request_id}")
                self.result_socket.send(completion_payload)

        return

    def _step_async(self):
        """
        Process the next batch of requests asynchronously.
        For now, just calls _step method.
        """
        return self._step()

    def run_forever(self):
        """
        Run the scheduler indefinitely.
        """
        while True:
            self._step()
            # Optionally, sleep or yield to avoid busy-waiting
            torch.cuda.synchronize()

    def _prepare_requests(self):
        """
        Prepare requests for processing.
        This method should be implemented to gather new requests from clients.
        TODO: make it async with _step() function
        """

        # get new requests from ZMQ
        while True:
            try:
                message_payload = self.request_socket.recv(flags=zmq.NOBLOCK)
                delimiter_pos = message_payload.find(b"|")
                if delimiter_pos != -1:
                    # Parse JSON request data
                    json_data = message_payload[:delimiter_pos].decode("utf-8")
                    request_dict = json.loads(json_data)

                    # Create Request object from deserialized data
                    new_request = Request(
                        request_id=request_dict["request_id"],
                        prompt=request_dict["prompt"],
                        audio_path=request_dict.get("audio_path") if self.model_worker.supports_audio_input else None,
                        is_streaming=request_dict.get("is_streaming", False),
                        is_pressing=request_dict.get("is_streaming", False), # at first, streaming requests are pressing
                    )

                    self.logger.debug(f"{new_request=}")

                    # Store voice information as attribute (not part of Request dataclass)
                    # new_request.voice = request_dict.get('voice', 'tara')

                    self.active_requests.append(new_request)
                else:
                    self.logger.warning(f"Received malformed audio message: {message_payload[:50]}...")
            except zmq.Again:
                break
            except Exception as e:
                self.logger.error(f"Error receiving requests: {str(e)}")
                import traceback

                self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Filter out completed requests
        self.active_requests = [req for req in self.active_requests if not req.done_all]

