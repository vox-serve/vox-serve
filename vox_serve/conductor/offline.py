"""
Offline Conductor for batch processing scenarios.

Optimized for throughput over latency. Prioritizes LM generation completion
before detokenization to maximize GPU utilization for token generation.
"""

from typing import List

from ..pool import CudaGraphPool
from ..requests import Request
from .base import Conductor


class OfflineConductor(Conductor):
    """
    Conductor optimized for offline/batch processing.

    Features:
    - Prioritizes LM requests when any are ongoing
    - Only detokenizes when all LM generation is complete
    - Maximizes batch utilization for throughput
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _select_lm_requests(self) -> List[Request]:
        """
        Offline scheduling: prioritize LM requests when any are ongoing.
        """
        lm_requests = []

        # Get constraints
        if isinstance(self.pool, CudaGraphPool):
            max_prefill_batch_size = self.pool.prefill_graph_batch_size
            max_seq_len = max(self.pool.cuda_graph_seq_len_buckets)
        else:
            max_prefill_batch_size = self.max_batch_size
            max_seq_len = 1024

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

        # Check if there are any ongoing LM requests
        has_ongoing_lm = len(prefill_requests) > 0 or len(decode_requests) > 0

        if not has_ongoing_lm:
            return []

        # Process prefill requests first with constraints
        if prefill_requests:
            current_batch_size = 0
            current_seq_len = 0

            for req in prefill_requests:
                req_seq_len = req.input_length if req.input_length else 200

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

            remaining_slots = max_prefill_batch_size - len(lm_requests)
        else:
            remaining_slots = self.max_batch_size

        # Fill remaining slots with decode requests
        for i in range(remaining_slots):
            if len(lm_requests) >= self.max_batch_size:
                break

            if i >= len(decode_requests):
                break

            lm_requests.append(decode_requests[i])

        return lm_requests

    def _select_detokenize_requests(self) -> List[Request]:
        """
        Offline scheduling: only detokenize when no LM requests are ongoing.

        Naive approach: go through requests one by one, add every chunk
        until max batch size is reached.
        """
        # Check if there are any ongoing LM requests
        has_ongoing_lm = False
        for req in self.active_requests:
            if not req.done_lm_generation:
                has_ongoing_lm = True
                break

        # If there are ongoing LM requests, don't do detokenization
        if has_ongoing_lm:
            return []

        # No ongoing LM requests, do detokenization with biggest batch size
        detokenize_interval = self.pool.detokenize_interval
        detokenize_overlap = self.pool.detokenize_overlap
        step = detokenize_interval - detokenize_overlap

        selected_requests = []
        total_chunks = 0

        # Naive approach: go through requests one by one, add all available chunks
        for req in self.active_requests:
            if total_chunks >= self.max_batch_size:
                break

            next_decode_idx = (
                req.next_audio_decode_idx[-1] + step if req.next_audio_decode_idx else 0
            )
            audio_idx_list = []

            # Add all available chunks for this request until we hit the batch limit
            while (
                total_chunks < self.max_batch_size
                and next_decode_idx + detokenize_interval <= len(req.lm_output_audio_tokens)
            ):
                audio_idx_list.append(next_decode_idx)
                next_decode_idx += step
                total_chunks += 1

            # If generation is done, try to include the final chunk
            if (
                req.done_lm_generation
                and total_chunks < self.max_batch_size
                and next_decode_idx < len(req.lm_output_audio_tokens)
            ):
                audio_idx_list.append(next_decode_idx)
                total_chunks += 1

            # If we have chunks to process for this request, add it
            if audio_idx_list:
                req.next_audio_decode_idx = audio_idx_list
                selected_requests.append(req)

        return selected_requests
