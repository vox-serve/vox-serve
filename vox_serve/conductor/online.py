"""
Online Conductor with priority-aware batching for streaming.

Optimized for low-latency streaming scenarios where playback timing
matters. Prioritizes "pressing" requests that need chunks urgently.
"""

import time
from typing import List

from ..pool import CudaGraphPool
from ..requests import Request
from .base import Conductor


class OnlineConductor(Conductor):
    """
    Conductor with priority-aware batching for streaming scenarios.

    Features:
    - Tracks playback timing to identify "pressing" requests
    - Prioritizes critical requests over non-critical ones
    - Piggybacks non-critical work to fill remaining batch slots
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detokenize_max_batch_size = self.max_batch_size

    def _select_lm_requests(self) -> List[Request]:
        """
        Select requests for LM processing with priority-aware batching.
        """
        lm_requests = []

        # Get constraints
        if isinstance(self.pool, CudaGraphPool):
            max_prefill_batch_size = self.pool.prefill_graph_batch_size
            max_seq_len = max(self.pool.cuda_graph_seq_len_buckets)
        else:
            max_prefill_batch_size = self.max_batch_size
            max_seq_len = 1024

        # Separate by criticality
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

        # Allocate prefill requests
        if prefill_requests:
            current_batch_size = 0
            current_seq_len = 0

            for req in prefill_requests:
                req_seq_len = req.input_length if req.input_length else 0

                if (
                    current_batch_size + 1 <= max_prefill_batch_size
                    and current_seq_len + req_seq_len <= max_seq_len
                ):
                    lm_requests.append(req)
                    current_batch_size += 1
                    current_seq_len += req_seq_len

                    if current_batch_size >= max_prefill_batch_size:
                        break

                # Allow only one prefill request for now
                break

            max_batch_size_this_cycle = max_prefill_batch_size
        else:
            max_batch_size_this_cycle = self.max_batch_size

        # Add critical decode requests
        for req in critical_decode_requests:
            if len(lm_requests) >= max_batch_size_this_cycle:
                break
            lm_requests.append(req)

        # Piggyback non-critical requests
        for req in non_critical_decode_requests:
            if len(lm_requests) >= max_batch_size_this_cycle:
                break
            lm_requests.append(req)

        return lm_requests

    def _select_detokenize_requests(self) -> List[Request]:
        """
        Select requests for detokenization with priority-aware batching.
        """
        detokenize_interval = self.pool.detokenize_interval
        detokenize_overlap = self.pool.detokenize_overlap
        step = detokenize_interval - detokenize_overlap

        # Get candidates
        detokenize_candidates = []
        for req in self.active_requests:
            next_decode_idx = (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            )
            if req.done_lm_generation:
                if next_decode_idx < len(req.lm_output_audio_tokens):
                    detokenize_candidates.append(req)
            elif next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens):
                detokenize_candidates.append(req)

        if not detokenize_candidates:
            return []

        # Separate by criticality
        critical_requests = [req for req in detokenize_candidates if req.is_pressing]
        non_critical_requests = [req for req in detokenize_candidates if not req.is_pressing]

        # Only process if there are pressing requests
        if not critical_requests:
            return []

        selected_requests = []

        # Calculate remaining chunks per critical request
        remaining_chunks = []
        for req in critical_requests:
            next_decode_idx = (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            )
            remaining_tokens = len(req.lm_output_audio_tokens) - next_decode_idx
            count = max(0, remaining_tokens // step)
            if req.done_lm_generation and remaining_tokens > 0:
                count += 1
            remaining_chunks.append(count)

        total_chunks = sum(remaining_chunks)
        if total_chunks <= self.detokenize_max_batch_size:
            assigned_chunks = remaining_chunks
        else:
            # Proportional allocation
            assigned_chunks = [
                max(1, (count * self.detokenize_max_batch_size) // total_chunks)
                for count in remaining_chunks
            ]

            while sum(assigned_chunks) > self.detokenize_max_batch_size:
                for i in range(len(assigned_chunks)):
                    if assigned_chunks[i] > 1:
                        assigned_chunks[i] -= 1
                        if sum(assigned_chunks) <= self.detokenize_max_batch_size:
                            break

        batch_used = 0
        for i, req in enumerate(critical_requests):
            remaining = assigned_chunks[i]
            if remaining <= 0:
                continue

            next_decode_idx = (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            )
            audio_idx_list = []

            while (
                remaining > 0
                and next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens)
            ):
                audio_idx_list.append(next_decode_idx)
                next_decode_idx += step
                remaining -= 1

            if (
                req.done_lm_generation
                and remaining > 0
                and next_decode_idx < len(req.lm_output_audio_tokens)
            ):
                audio_idx_list.append(next_decode_idx)
                remaining -= 1

            if not audio_idx_list:
                continue

            req.next_audio_decode_idx = audio_idx_list
            batch_used += len(audio_idx_list)
            selected_requests.append(req)

        # Piggyback non-critical requests
        if batch_used < self.detokenize_max_batch_size:
            remaining_slots = self.detokenize_max_batch_size - batch_used

            for req in non_critical_requests:
                if remaining_slots <= 0:
                    break

                next_decode_idx = (
                    req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
                )
                audio_idx_list = []

                while (
                    remaining_slots > 0
                    and next_decode_idx + detokenize_interval
                    <= len(req.lm_output_audio_tokens)
                ):
                    audio_idx_list.append(next_decode_idx)
                    next_decode_idx += step
                    remaining_slots -= 1

                if (
                    req.done_lm_generation
                    and remaining_slots > 0
                    and next_decode_idx < len(req.lm_output_audio_tokens)
                ):
                    audio_idx_list.append(next_decode_idx)
                    remaining_slots -= 1

                if not audio_idx_list:
                    continue

                req.next_audio_decode_idx = audio_idx_list
                selected_requests.append(req)

        return selected_requests

    def _prepare_requests(self):
        super()._prepare_requests()
        self._update_pressing_status()

    async def _prepare_requests_async(self):
        await super()._prepare_requests_async()
        self._update_pressing_status()

    def _update_pressing_status(self):
        """
        Update pressing status based on playback timing.

        A request is pressing if:
        1. No chunk has been generated yet, or
        2. The latest chunk is already started playing
        """
        current_time = time.time()

        for req in self.active_requests:
            if not req.is_streaming:
                req.is_pressing = False
                continue

            if not req.chunk_send_timestamps:
                req.is_pressing = True
                continue

            first_chunk_send_time = req.chunk_send_timestamps[0]
            total_playback_time = sum(req.chunk_durations)
            latest_chunk_start_time = (
                first_chunk_send_time + total_playback_time - req.chunk_durations[-1]
            )

            req.is_pressing = current_time >= latest_chunk_start_time - 1.0
