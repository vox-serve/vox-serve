import time
from typing import List

from ..requests import Request
from ..worker import CudaGraphWorker
from .base import Scheduler


class OnlineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.detokenize_max_batch_size = min(16, self.max_batch_size)

    def _select_lm_requests(self) -> List[Request]:
        """
        Select requests for LM forward processing with priority-aware batching.
        First allocate critical requests, then piggyback non-critical ones to reach optimal batch size.
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

        # Separate prefill and decode requests based on criticality
        # (prefill requests are always critical)
        prefill_requests = []
        critical_decode_requests = []
        non_critical_decode_requests = []

        for req in self.active_requests:
            if req.done_lm_generation:
                continue

            if not req.done_lm_prefill:
                prefill_requests.append(req)
            elif req.is_pressing:
                critical_decode_requests.append(req)
            else:
                non_critical_decode_requests.append(req)

        # if len(prefill_requests) + len(critical_decode_requests) + len(non_critical_decode_requests) > 0:
        #     print(f"{len(prefill_requests)} prefill, "
        #           f"{len(critical_decode_requests)} critical decode, "
        #           f"{len(non_critical_decode_requests)} non-critical decode requests")

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

            max_batch_size_this_cycle = max_prefill_batch_size

        else:
            max_batch_size_this_cycle = self.max_batch_size

        # Then, allocate decode requests that is critical
        for i in range(len(critical_decode_requests)):
            if len(lm_requests) >= max_batch_size_this_cycle:
                break

            lm_requests.append(critical_decode_requests[i])

        # Finally, piggyback non-critical decode requests, if any slots remain
        for i in range(len(non_critical_decode_requests)):
            if len(lm_requests) >= max_batch_size_this_cycle:
                break

            lm_requests.append(non_critical_decode_requests[i])

        return lm_requests

    def _select_detokenize_requests(self) -> List[Request]:
        """
        Select requests for detokenization with priority-aware batching.
        Only processes detokenization if there's at least one pressing request ready.
        """
        detokenize_interval = self.model_worker.detokenize_interval
        detokenize_overlap = self.model_worker.detokenize_overlap
        step = detokenize_interval - detokenize_overlap

        # Get all requests that are ready for detokenization
        detokenize_candidates = [
            req for req in self.active_requests
            if req.done_lm_generation or (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            ) + detokenize_interval <= len(req.lm_output_audio_tokens)
        ]

        if not detokenize_candidates:
            return []

        # Separate critical and non-critical detokenization requests
        critical_requests = [req for req in detokenize_candidates if req.is_pressing]
        non_critical_requests = [req for req in detokenize_candidates if not req.is_pressing]

        # if len(critical_requests) > 0:
        #     print(f"Critical detokenize requests: {len(critical_requests)}, "
        #           f"Non-critical: {len(non_critical_requests)}")

        # If there are no pressing requests ready for detokenization, do nothing
        if not critical_requests:
            return []

        selected_requests = []

        # Build up selected_requests while keeping the cumulative number of indices <= detokenize_max_batch_size
        # Calculate remaining chunks per request
        remaining_chunks = []
        for req in critical_requests:
            next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            remaining_tokens = len(req.lm_output_audio_tokens) - next_decode_idx
            count = max(0, remaining_tokens // step)
            if req.done_lm_generation and remaining_tokens > 0:
                count += 1  # Include final partial chunk
            remaining_chunks.append(count)

        total_chunks = sum(remaining_chunks)
        if total_chunks <= self.detokenize_max_batch_size:
            assigned_chunks = remaining_chunks
        else:
            # Proportional allocation
            assigned_chunks = [
                max(1, (count * self.detokenize_max_batch_size) // total_chunks) for count in remaining_chunks
            ]

            # Adjust in case of rounding issues
            while sum(assigned_chunks) > self.detokenize_max_batch_size:
                for i in range(len(assigned_chunks)):
                    if assigned_chunks[i] > 1:
                        assigned_chunks[i] -= 1
                        if sum(assigned_chunks) <= self.detokenize_max_batch_size:
                            break

        batch_used = 0
        for i, req in enumerate(critical_requests):
            # remaining = self.detokenize_max_batch_size - batch_used
            remaining = assigned_chunks[i]
            if remaining <= 0:
                continue

            # Compute candidate indices for this req
            next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            audio_idx_list = []

            # Add as many indices as we can for this request without exceeding the global budget
            while (
                remaining > 0
                and next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens)
            ):
                audio_idx_list.append(next_decode_idx)
                next_decode_idx += step
                remaining -= 1

            # If generation is done, try to include the final index if there's still budget
            if req.done_lm_generation and remaining > 0:
                audio_idx_list.append(next_decode_idx)
                remaining -= 1

            # If we couldn't allocate any indices for this req, skip it
            if not audio_idx_list:
                continue

            # Commit the allocation
            req.next_audio_decode_idx = audio_idx_list
            batch_used += len(audio_idx_list)
            selected_requests.append(req)
            # print(f"{req.request_id} {req.next_audio_decode_idx=} {len(req.lm_output_audio_tokens)=}")

        # If we don't have a full batch, try to piggyback non-critical requests
        if batch_used < self.detokenize_max_batch_size:
            remaining_slots = self.detokenize_max_batch_size - batch_used

            for req in non_critical_requests:
                if remaining_slots <= 0:
                    break

                next_decode_idx = req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
                audio_idx_list = []

                while (
                    remaining_slots > 0
                    and next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens)
                ):
                    audio_idx_list.append(next_decode_idx)
                    next_decode_idx += step
                    remaining_slots -= 1

                if req.done_lm_generation and remaining_slots > 0:
                    audio_idx_list.append(next_decode_idx)
                    remaining_slots -= 1

                if not audio_idx_list:
                    continue

                req.next_audio_decode_idx = audio_idx_list
                selected_requests.append(req)
                # print(f"{req.request_id} {req.next_audio_decode_idx=} {len(req.lm_output_audio_tokens)=}")

        return selected_requests

    def _prepare_requests(self):
        super()._prepare_requests()

        # Update pressing status for all streaming requests
        self._update_pressing_status()

    async def _prepare_requests_async(self):
        await super()._prepare_requests_async()

        # Update pressing status for all streaming requests
        self._update_pressing_status()

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
            req.is_pressing = current_time >= latest_chunk_start_time - 1.0 # 1.0s buffer
