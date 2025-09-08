import asyncio
import json
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
        async_scheduling: bool = False,
    ):
        self.device = device
        self.max_batch_size = max_batch_size
        self.async_scheduling = async_scheduling
        self.logger = get_logger(__name__)
        self.logger.info(f"Using {'async' if async_scheduling else 'sync'} scheduling mode")

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

        self.available_batch_sizes = self.model_worker.available_batch_sizes

    def _step(self):
        """
        Process the next batch of requests.
        """

        # insert/remove requests to self.active_requests
        self._prepare_requests()

        # Select requests for detokenization
        detokenize_requests = self._select_detokenize_requests()

        # run detokenization if needed
        self.model_worker.run_detokenize(detokenize_requests)

        # return results to clients
        self._send_responses(detokenize_requests)

        # Select requests for LM processing
        lm_requests = self._select_lm_requests()

        # Prepare LM inputs outside the worker and run either prefill or decode
        lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests)

        if lm_inputs is not None and lm_inputs["is_prefill"]:
            task = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
        else:
            task = self.model_worker.run_lm_decode(lm_requests, lm_inputs)

        # Execute the sampling task right away for synchronous scheduling
        if task is not None:
            asyncio.run(task)


    async def _step_async(self, task, lm_requests, detokenize_requests):
        """
        Process the next batch of requests asynchronously.
        """

        # insert/remove requests to self.active_requests
        self._prepare_requests()

        # Prepare LM inputs outside the worker and run either prefill or decode
        lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests)

        async def run_model():
            # run detokenization if needed
            self.model_worker.run_detokenize(detokenize_requests)

            # return results to clients
            self._send_responses(detokenize_requests)

            if lm_inputs is not None and lm_inputs["is_prefill"]:
                next_task = self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            else:
                next_task = self.model_worker.run_lm_decode(lm_requests, lm_inputs)

            return next_task

        async def run_scheduling():
            if task is not None:
                await task

            # Select requests for detokenization
            next_detokenize_requests = self._select_detokenize_requests()

            # Select requests for LM processing
            next_lm_requests = self._select_lm_requests()

            return next_lm_requests, next_detokenize_requests

        model_result, scheduling_result = await asyncio.gather(
            run_model(),
            run_scheduling()
        )

        # Unpack the results from the completed tasks
        next_task = model_result
        next_lm_requests, next_detokenize_requests = scheduling_result

        return next_task, next_lm_requests, next_detokenize_requests


    def _select_lm_requests(self):
        """
        Select requests that need LM processing.
        For prefill requests, ensure batch size and sequence length don't exceed worker limitations.
        Allocate prefill requests first, then decode requests in remaining slots.
        """
        lm_requests = []

        # Get worker limitations
        if isinstance(self.model_worker, CudaGraphWorker):
            max_prefill_batch_size = self.model_worker.prefill_graph_batch_size
            max_seq_len = max(self.model_worker.cuda_graph_seq_len_buckets)
        else:
            # Fallback for non-CUDA graph worker
            max_prefill_batch_size = self.max_batch_size
            max_seq_len = 1024  # Default assumption

        # Separate prefill and decode requests
        prefill_requests = []
        decode_requests = []

        for req in self.active_requests:
            if req.done_lm_generation:
                continue

            if not req.done_lm_prefill:
                prefill_requests.append(req)
            else:
                decode_requests.append(req)

        # First, allocate prefill requests with constraints
        if prefill_requests:
            current_batch_size = 0
            current_seq_len = 0

            for req in prefill_requests:
                req_seq_len = req.input_length if req.input_length else 0

                # Check if adding this request would exceed constraints
                if (current_batch_size + 1 <= max_prefill_batch_size and
                    current_seq_len + req_seq_len <= max_seq_len):

                    lm_requests.append(req)
                    current_batch_size += 1
                    current_seq_len += req_seq_len

                    if current_batch_size >= max_prefill_batch_size:
                        break

            remaining_slots = max_prefill_batch_size - len(lm_requests)

        else:
            remaining_slots = self.max_batch_size

        for i in range(remaining_slots):
            if len(lm_requests) >= self.max_batch_size:
                break

            if i >= len(decode_requests):
                break

            lm_requests.append(decode_requests[i])

        return lm_requests

    def _select_detokenize_requests(self):
        """
        Select requests that need detokenization.
        """
        detokenize_requests = []

        for req in self.active_requests:
            if len(detokenize_requests) > self.max_batch_size:
                break
            if req.done_lm_generation or self.model_worker.do_detokenize(req):
                detokenize_requests.append(req)

        return detokenize_requests

    def _send_responses(self, detokenize_requests):
        """
        Send responses back to clients for detokenized requests.
        """
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

    def run_forever(self):
        """
        Run the scheduler indefinitely.
        """
        task, lm_requests, detokenize_requests = None, [], []
        while True:
            if self.async_scheduling:
                task, lm_requests, detokenize_requests = asyncio.run(
                    self._step_async(task, lm_requests, detokenize_requests)
                )
            else:
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

