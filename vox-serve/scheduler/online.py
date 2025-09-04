import json
import time
from typing import List

from ..requests import Request
from .base import Scheduler


class OnlineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Audio parameters for duration calculation
        # Assuming 24kHz mono 16-bit audio
        self.sample_rate = 24000
        self.bytes_per_sample = 2  # 16-bit = 2 bytes
        self.channels = 1  # mono

    def _step(self):
        """
        Priority-aware scheduling step that considers request criticality (is_pressing).
        Updates pressing status based on audio chunk playback timing.
        """
        # insert/remove requests to self.active_requests
        self._prepare_requests()

        if len(self.active_requests) == 0:
            return

        # Select requests for LM forward with priority-aware batching
        lm_requests = self._select_lm_requests()

        if lm_requests:
            # Check if any request needs prefill
            is_prefill = any(not req.done_lm_prefill for req in lm_requests)

            # Prepare inputs and run LM inference (prefill or decode)
            lm_inputs = self.model_worker.prepare_lm_inputs(lm_requests)
            if is_prefill:
                self.model_worker.run_lm_prefill(lm_requests, lm_inputs)
            else:
                self.model_worker.run_lm_decode(lm_requests, lm_inputs)

        # Select requests for detokenization with priority-aware batching
        detokenize_requests = self._select_detokenize_requests()

        # Run detokenization if needed
        if detokenize_requests:
            self.model_worker.run_detokenize(detokenize_requests)

        # Return results to clients with timestamp tracking
        for req in self.active_requests:
            while not req.output_audio.empty():
                audio_chunk = req.output_audio.get()

                # Record timestamp and duration for streaming requests
                if req.is_streaming:
                    send_time = time.time()
                    duration = self._calculate_chunk_duration(audio_chunk)
                    req.chunk_send_timestamps.append(send_time)
                    req.chunk_durations.append(duration)

                # Send audio chunk message: request_id|AUDIO|audio_data
                message = req.request_id.encode("utf-8") + b"|AUDIO|" + audio_chunk
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

    def _select_lm_requests(self) -> List[Request]:
        """
        Select requests for LM forward processing with priority-aware batching.
        First allocate critical requests, then piggyback non-critical ones to reach optimal batch size.

        TODO: consider prefill vs. decode separately for better batching.
        """
        # Update pressing status for all streaming requests
        self._update_pressing_status()

        lm_candidates = [req for req in self.active_requests if not req.done_lm_generation]

        # Separate critical (pressing) and non-critical requests
        critical_requests = [req for req in lm_candidates if req.is_pressing]
        non_critical_requests = [req for req in lm_candidates if not req.is_pressing]

        # First, take critical requests up to max_batch_size
        selected_requests = critical_requests[: self.max_batch_size]

        # If we don't have a full batch and the current size is not a CUDA graph size,
        # try to piggyback non-critical requests to reach the next CUDA graph batch size
        current_batch_size = len(selected_requests)

        if current_batch_size < self.max_batch_size and current_batch_size not in self.available_batch_sizes:
            # Find the next larger CUDA graph batch size
            target_batch_size = None
            for size in sorted(self.available_batch_sizes):
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
        Only processes detokenization if there's at least one pressing request ready.
        """
        # Get all requests that are ready for detokenization
        detokenize_candidates = [
            req for req in self.active_requests if req.done_lm_generation or self.model_worker.do_detokenize(req)
        ]

        if not detokenize_candidates:
            return []

        # Separate critical and non-critical detokenization requests
        critical_requests = [req for req in detokenize_candidates if req.is_pressing]
        non_critical_requests = [req for req in detokenize_candidates if not req.is_pressing]

        # If there are no pressing requests ready for detokenization, do nothing
        if not critical_requests:
            return []

        # First, take critical requests up to max_batch_size
        selected_requests = critical_requests[: self.max_batch_size]

        # If we don't have a full batch and the current size is not a CUDA graph size,
        # try to piggyback non-critical requests to reach the next CUDA graph batch size
        current_batch_size = len(selected_requests)

        if current_batch_size < self.max_batch_size and current_batch_size not in self.available_batch_sizes:
            # Find the next larger CUDA graph batch size
            target_batch_size = None
            for size in sorted(self.available_batch_sizes):
                if size > current_batch_size:
                    target_batch_size = min(size, self.max_batch_size)
                    break

            if target_batch_size:
                # Piggyback non-critical requests to reach target batch size
                available_slots = target_batch_size - current_batch_size
                selected_requests.extend(non_critical_requests[:available_slots])

        return selected_requests

    def _calculate_chunk_duration(self, audio_chunk: bytes) -> float:
        """
        Calculate the duration of an audio chunk in seconds.
        Assumes 24kHz mono 16-bit PCM audio.
        """
        num_samples = len(audio_chunk) // (self.channels * self.bytes_per_sample)
        duration_seconds = num_samples / self.sample_rate
        return duration_seconds

    def _update_pressing_status(self):
        """
        Update the pressing status for all streaming requests based on playback timing.
        A request is pressing if:
        1. No chunk has been generated yet, or
        2. The latest chunk is already started playing at the client side
        """
        current_time = time.time()

        for req in self.active_requests:
            if not req.is_streaming:
                # Non-streaming requests are never pressing
                req.is_pressing = False
                continue

            # If no chunks have been sent yet, the request is pressing
            if not req.chunk_send_timestamps:
                req.is_pressing = True
                continue

            # Calculate when the latest chunk started playing at client side
            # Client starts playing immediately when it receives the first chunk
            first_chunk_send_time = req.chunk_send_timestamps[0]

            # Calculate accumulated playback time from when first chunk was sent
            total_playback_time = sum(req.chunk_durations)
            latest_chunk_start_time = first_chunk_send_time + total_playback_time - req.chunk_durations[-1]

            # If the latest chunk has already started playing, request is pressing
            req.is_pressing = current_time >= latest_chunk_start_time - 2.0 # 1.0s buffer
