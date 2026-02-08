"""
Input Streaming Scheduler for incremental text input during generation.

This scheduler extends the base Scheduler to support receiving text chunks
while audio generation is in progress.

Key Design (aligned with Qwen3-TTS reference implementation):
- Text is buffered until minimum initial text is received
- Prefill starts with only 1 text token (like reference streaming mode)
- Remaining buffered text tokens are queued for decode injection
- Additional text arriving during decode is also queued token-by-token
- If text doesn't arrive fast enough during decode, generation pauses
"""

import json
from typing import List, Optional

from ..requests import Request
from .base import Scheduler

# Minimum characters before starting prefill.
# We need at least a few characters to produce 1+ tokens.
MIN_INITIAL_TEXT_CHARS = 20


class InputStreamingScheduler(Scheduler):
    """
    Scheduler with support for incremental text input.

    Handles three new message types:
    - TEXT_STREAM_START: Initialize a streaming request (buffering mode)
    - TEXT_UPDATE: Add text - buffered before prefill, queued after prefill
    - TEXT_COMPLETE: Signal end of text input
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("InputStreamingScheduler initialized with text streaming support")

    def _prepare_prefill_with_minimal_text(self, req: Request) -> None:
        """
        Prepare request for prefill with minimal text (1 token), queue the rest.

        Aligned with Qwen3-TTS reference implementation streaming mode:
        - Only 1 text token is used at prefill time
        - Remaining tokens become "trailing text" consumed during decode

        Args:
            req: The request to prepare
        """
        # Tokenize all buffered text (without special tokens)
        all_tokens = self.model_worker.model.text_tokenizer.encode(
            req.input_text_buffer, add_special_tokens=False
        )

        if not all_tokens:
            # Edge case: text didn't produce any tokens
            req.prompt = req.input_text_buffer
            req.prefill_ready = True
            return

        # Decode only the first token back to text for minimal prompt
        first_token_text = self.model_worker.model.text_tokenizer.decode(
            [all_tokens[0]], skip_special_tokens=True
        )
        req.prompt = first_token_text

        # Queue remaining tokens for decode injection
        for tok in all_tokens[1:]:
            req.pending_text_tokens.put(tok)
        req.total_text_tokens = len(all_tokens) - 1

        req.prefill_ready = True
        self.logger.debug(
            f"Prefill prepared: prompt='{first_token_text}' (1 token), "
            f"queued {len(all_tokens) - 1} tokens"
        )

    def _handle_request_payload(self, message_payload: bytes) -> Optional[Request]:
        """
        Handle request payload parsing, including text streaming messages.

        Message formats:
        - Regular: {json_data}|audio_data
        - TEXT_STREAM_START: request_id|TEXT_STREAM_START|{json_config}
        - TEXT_UPDATE: request_id|TEXT_UPDATE|text_chunk
        - TEXT_COMPLETE: request_id|TEXT_COMPLETE|
        """
        # Try to parse as text streaming message first
        parts = message_payload.split(b"|", 2)

        if len(parts) >= 2:
            try:
                msg_type = parts[1].decode("utf-8")
            except UnicodeDecodeError:
                msg_type = None

            if msg_type == "TEXT_STREAM_START":
                return self._handle_text_stream_start(parts[0], parts[2] if len(parts) > 2 else b"{}")
            elif msg_type == "TEXT_UPDATE":
                self._handle_text_update(parts[0], parts[2] if len(parts) > 2 else b"")
                return None  # No new request created
            elif msg_type == "TEXT_COMPLETE":
                self._handle_text_complete(parts[0])
                return None  # No new request created

        # Fall back to parent implementation for regular requests
        return super()._handle_request_payload(message_payload)

    def _handle_text_stream_start(self, request_id_bytes: bytes, config_json: bytes) -> Request:
        """
        Initialize a text-streaming request in buffering mode.

        Args:
            request_id_bytes: The request ID as bytes
            config_json: JSON configuration for the request

        Returns:
            New Request object with is_input_streaming=True
        """
        request_id = request_id_bytes.decode("utf-8")
        try:
            config = json.loads(config_json.decode("utf-8"))
        except json.JSONDecodeError:
            config = {}

        new_request = Request(
            request_id=request_id,
            prompt="",  # Will be set when enough text buffered
            audio_path=config.get("audio_path") if self.model_worker.supports_audio_input else None,
            is_streaming=config.get("is_streaming", True),
            is_pressing=config.get("is_streaming", True),
            is_input_streaming=True,
            input_text_buffer="",  # Buffer for initial text before prefill
            text_complete=False,
            prefill_ready=False,  # Not ready until enough text buffered
            waiting_for_text=False,
            model_kwargs=config.get("model_kwargs", {}),
        )

        self.logger.info(f"Created input streaming request: {request_id}")
        return new_request

    def _handle_text_update(self, request_id_bytes: bytes, text_chunk: bytes) -> None:
        """
        Handle text update - buffer before prefill, queue tokens after prefill.

        Args:
            request_id_bytes: The request ID as bytes
            text_chunk: Text chunk to process
        """
        request_id = request_id_bytes.decode("utf-8")
        text = text_chunk.decode("utf-8")

        if not text:
            return

        # Find the request
        for req in self.active_requests:
            if req.request_id == request_id:
                if req.text_complete:
                    self.logger.warning(f"TEXT_UPDATE after TEXT_COMPLETE for {request_id}, ignoring")
                    return

                if not req.done_lm_prefill:
                    # Before prefill: buffer text
                    req.input_text_buffer += text
                    self.logger.debug(
                        f"Buffered text for {request_id}: +{len(text)} chars "
                        f"(total: {len(req.input_text_buffer)} chars)"
                    )

                    # Check if we have enough to start prefill
                    if not req.prefill_ready and len(req.input_text_buffer) >= MIN_INITIAL_TEXT_CHARS:
                        # Aligned with reference: only 1 text token at prefill, queue the rest
                        self._prepare_prefill_with_minimal_text(req)
                        self.logger.info(
                            f"Request {request_id} ready for prefill with 1 token, "
                            f"{req.pending_text_tokens.qsize()} tokens queued"
                        )
                else:
                    # After prefill: tokenize and queue for decode injection
                    tokens = self.model_worker.model.text_tokenizer.encode(
                        text, add_special_tokens=False
                    )
                    for tok in tokens:
                        req.pending_text_tokens.put(tok)
                    req.total_text_tokens += len(tokens)

                    # Wake up if waiting for text
                    if req.waiting_for_text:
                        req.waiting_for_text = False

                    self.logger.debug(
                        f"Queued {len(tokens)} tokens for {request_id} "
                        f"(total queued: {req.total_text_tokens})"
                    )

                return

        self.logger.warning(f"TEXT_UPDATE for unknown request: {request_id}")

    def _handle_text_complete(self, request_id_bytes: bytes) -> None:
        """
        Mark text input as complete.

        If prefill hasn't started yet, trigger it with whatever text is buffered.

        Args:
            request_id_bytes: The request ID as bytes
        """
        request_id = request_id_bytes.decode("utf-8")

        for req in self.active_requests:
            if req.request_id == request_id:
                req.text_complete = True

                # If prefill hasn't started, start it with whatever we have
                if not req.prefill_ready:
                    if req.input_text_buffer:
                        # Use same minimal prefill approach
                        self._prepare_prefill_with_minimal_text(req)
                        self.logger.info(
                            f"TEXT_COMPLETE for {request_id}: starting prefill with 1 token, "
                            f"{req.pending_text_tokens.qsize()} tokens queued"
                        )
                    else:
                        self.logger.warning(f"TEXT_COMPLETE for {request_id} but no text buffered")
                        req.done_lm_generation = True
                        req.done_all = True
                        req.finish_reason = "no_text_provided"
                else:
                    # Prefill already started/done, just mark complete
                    # Wake up if waiting - will use pad tokens from now on
                    if req.waiting_for_text:
                        req.waiting_for_text = False
                    self.logger.info(
                        f"TEXT_COMPLETE for {request_id} "
                        f"(queued tokens: {req.pending_text_tokens.qsize()})"
                    )
                return

        self.logger.warning(f"TEXT_COMPLETE for unknown request: {request_id}")

    def _select_lm_requests(self) -> List[Request]:
        """
        Select requests that need LM processing.

        For input streaming requests:
        - Skip prefill until prefill_ready=True (enough initial text)
        - During decode, skip if waiting for text and not text_complete
        """
        lm_requests = []

        # Get worker limitations (same as parent)
        from ..worker import CudaGraphWorker
        if isinstance(self.model_worker, CudaGraphWorker):
            max_prefill_batch_size = self.model_worker.prefill_graph_batch_size
            max_seq_len = max(self.model_worker.cuda_graph_seq_len_buckets)
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
                # For input streaming, only allow prefill when ready
                if req.is_input_streaming and not req.prefill_ready:
                    continue  # Skip - still buffering initial text
                prefill_requests.append(req)
            else:
                # Decode request
                if req.is_input_streaming:
                    # Check if we have text available or text is complete
                    if req.pending_text_tokens.empty() and not req.text_complete:
                        req.waiting_for_text = True
                        continue  # Skip - waiting for more text
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

                # allow only one prefill request for now
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
