import json
from typing import List

import numpy as np

from ..requests import Request
from .base import Scheduler


class OnlineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute CUDA graph batch sizes for optimization
        self.cuda_graph_batch_sizes = [2**i for i in range(int(np.log2(self.max_batch_size)) + 1)]

    def _step(self):
        """
        Priority-aware scheduling step that considers request criticality (is_pressing).
        """
        # insert/remove requests to self.active_requests
        self._prepare_requests()

        if len(self.active_requests) == 0:
            return

        # Select requests for LM forward with priority-aware batching
        lm_requests = self._select_lm_requests()

        if len(lm_requests) == 0:
            return

        # Check if any request needs prefill
        is_prefill = any(not req.done_lm_prefill for req in lm_requests)

        # Run LM inference (prefill or decode)
        if is_prefill:
            self.model_worker.run_lm_prefill(lm_requests)
        else:
            self.model_worker.run_lm_decode(lm_requests)

        # Check for request completion and mark as done
        for req in lm_requests:
            if self.model_worker.is_finished(req):
                req.done_all = True

        # Select requests for detokenization with priority-aware batching
        detokenize_requests = self._select_detokenize_requests()

        # Run detokenization if needed
        if detokenize_requests:
            self.model_worker.run_detokenize(detokenize_requests)

        # Return results to clients
        for req in lm_requests:
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

    def _select_lm_requests(self) -> List[Request]:
        """
        Select requests for LM forward processing with priority-aware batching.
        First allocate critical requests, then piggyback non-critical ones to reach optimal batch size.
        """
        # Separate critical (pressing) and non-critical requests
        critical_requests = [req for req in self.active_requests if req.is_pressing]
        non_critical_requests = [req for req in self.active_requests if not req.is_pressing]

        # First, take critical requests up to max_batch_size
        selected_requests = critical_requests[: self.max_batch_size]

        # If we don't have a full batch and the current size is not a CUDA graph size,
        # try to piggyback non-critical requests to reach the next CUDA graph batch size
        current_batch_size = len(selected_requests)

        if current_batch_size < self.max_batch_size and current_batch_size not in self.cuda_graph_batch_sizes:
            # Find the next larger CUDA graph batch size
            target_batch_size = None
            for size in sorted(self.cuda_graph_batch_sizes):
                if size > current_batch_size:
                    target_batch_size = min(size, self.max_batch_size)
                    break

            if target_batch_size:
                # Piggyback non-critical requests to reach target batch size
                available_slots = target_batch_size - current_batch_size
                selected_requests.extend(non_critical_requests[:available_slots])

        return selected_requests

    def _select_detokenize_requests(self) -> List[Request]:
        """
        Select requests for detokenization with priority-aware batching.
        Similar logic to LM request selection but for detokenization pipeline.
        """
        # Get all requests that are ready for detokenization
        detokenize_candidates = []
        for req in self.active_requests:
            if req.done_all or self.model_worker.do_detokenize(req):
                detokenize_candidates.append(req)

        if not detokenize_candidates:
            return []

        # Separate critical and non-critical detokenization requests
        critical_requests = [req for req in detokenize_candidates if req.is_pressing]
        non_critical_requests = [req for req in detokenize_candidates if not req.is_pressing]

        # First, take critical requests up to max_batch_size
        selected_requests = critical_requests[: self.max_batch_size]

        # If we don't have a full batch and the current size is not a CUDA graph size,
        # try to piggyback non-critical requests to reach the next CUDA graph batch size
        current_batch_size = len(selected_requests)

        if current_batch_size < self.max_batch_size and current_batch_size not in self.cuda_graph_batch_sizes:
            # Find the next larger CUDA graph batch size
            target_batch_size = None
            for size in sorted(self.cuda_graph_batch_sizes):
                if size > current_batch_size:
                    target_batch_size = min(size, self.max_batch_size)
                    break

            if target_batch_size:
                # Piggyback non-critical requests to reach target batch size
                available_slots = target_batch_size - current_batch_size
                selected_requests.extend(non_critical_requests[:available_slots])

        return selected_requests
