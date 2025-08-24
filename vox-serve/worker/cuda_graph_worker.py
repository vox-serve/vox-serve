import time
from typing import Dict, List

import numpy as np
import torch

from ..flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from ..requests import Request
from .base import ModelWorker


class CudaGraphWorker(ModelWorker):
    """
    ModelWorker subclass that adds CUDA graph optimization for improved inference performance.

    CUDA graphs capture and replay computation graphs to eliminate Python overhead
    during the decode phase, providing significant speedup for inference.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CUDA graph related attributes
        self.cuda_graphs_lm: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_detokenization: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_depth_prefill: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graphs_depth_decode: Dict[int, torch.cuda.CUDAGraph] = {}
        self.cuda_graph_buffers: Dict[str, torch.Tensor] = {}

        # Initialize CUDA graphs after parent initialization
        self._initialize_cuda_graphs()

        self.warmup()
        self.kv_cache.zero_()

        if self.has_depth_transformer:
            self.depth_kv_cache.zero_()

    def _prepare_attention_wrappers(self):
        self.cuda_graph_batch_sizes = [2**i for i in range(int(np.log2(self.max_batch_size)) + 1)]

        self.flashinfer_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.paged_kv_indptr_buffer = torch.zeros(self.max_batch_size + 1).to(self.device).to(torch.int32)
        self.paged_kv_indices_buffer = torch.zeros(self.max_num_pages).to(self.device).to(torch.int32)
        self.paged_kv_last_page_len_buffer = torch.zeros(self.max_batch_size).to(self.device).to(torch.int32)

        self.prefill_wrapper = FlashInferPrefillWrapper(
            attn_buffer=self.flashinfer_buffer,
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
        )

        self.decode_wrappers = {}
        for batch_size in self.cuda_graph_batch_sizes:
            self.decode_wrappers[batch_size] = FlashInferDecodeWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=self.model.num_attention_heads,
                n_kv_head=self.model.num_key_value_heads,
                n_state=self.model.hidden_size,
                page_size=self.page_size,
                batch_size=batch_size,
                paged_kv_indptr_buffer=self.paged_kv_indptr_buffer[: batch_size + 1],
                paged_kv_indices_buffer=self.paged_kv_indices_buffer,
                paged_kv_last_page_len_buffer=self.paged_kv_last_page_len_buffer[:batch_size],
                use_cuda_graph=True,
            )

        self.kv_cache = torch.zeros(
            self.model.num_hidden_layers,
            self.max_num_pages,
            2,  # K/V
            self.page_size,
            self.model.num_key_value_heads,  # kv heads
            self.model.hidden_size // self.model.num_attention_heads,  # head dim
            dtype=torch.bfloat16,
            device="cuda",
        )

        kv_cache_size = self.kv_cache.numel() * self.kv_cache.element_size()
        self.logger.info(f"KV cache size: {kv_cache_size / 1024 / 1024:.2f} MB")

        self.has_depth_transformer = self.model.has_depth_transformer
        if self.has_depth_transformer:
            # NOTE: for depth, there is always one page per request
            self.depth_qo_indptr_buffer = torch.zeros(self.max_batch_size + 1).to(self.device).to(torch.int32)
            self.depth_paged_kv_indptr_buffer = torch.zeros(self.max_batch_size + 1).to(self.device).to(torch.int32)
            self.depth_paged_kv_indices_buffer = torch.zeros(self.max_batch_size).to(self.device).to(torch.int32)
            self.depth_paged_kv_last_page_len_buffer = torch.zeros(self.max_batch_size).to(self.device).to(torch.int32)

            self.depth_prefill_wrappers = {}
            self.depth_decode_wrappers = {}

            for batch_size in self.cuda_graph_batch_sizes:
                # We enable CUDA graph for prefill phase as well since the sequence length (2) is fixed.
                self.depth_prefill_wrappers[batch_size] = FlashInferPrefillWrapper(
                    attn_buffer=self.flashinfer_buffer,
                    n_qo_head=self.model.depth_num_attention_heads,
                    n_kv_head=self.model.depth_num_key_value_heads,
                    n_state=self.model.depth_hidden_size,
                    page_size=self.model.depth_n_codebooks,
                    batch_size=batch_size,
                    qo_indptr_buffer=self.depth_qo_indptr_buffer[: batch_size + 1],
                    paged_kv_indptr_buffer=self.depth_paged_kv_indptr_buffer[: batch_size + 1],
                    paged_kv_indices_buffer=self.depth_paged_kv_indices_buffer[:batch_size],
                    paged_kv_last_page_len_buffer=self.depth_paged_kv_last_page_len_buffer[:batch_size],
                    use_cuda_graph=True,
                )

                self.depth_decode_wrappers[batch_size] = FlashInferDecodeWrapper(
                    attn_buffer=self.flashinfer_buffer,
                    n_qo_head=self.model.depth_num_attention_heads,
                    n_kv_head=self.model.depth_num_key_value_heads,
                    n_state=self.model.depth_hidden_size,
                    page_size=self.model.depth_n_codebooks,
                    batch_size=batch_size,
                    paged_kv_indptr_buffer=self.depth_paged_kv_indptr_buffer[: batch_size + 1],
                    paged_kv_indices_buffer=self.depth_paged_kv_indices_buffer[:batch_size],
                    paged_kv_last_page_len_buffer=self.depth_paged_kv_last_page_len_buffer[:batch_size],
                    use_cuda_graph=True,
                )

            self.depth_kv_cache = torch.zeros(
                self.model.depth_num_hidden_layers,
                self.max_batch_size,
                2,  # K/V
                self.model.depth_n_codebooks,
                self.model.depth_num_key_value_heads,  # kv heads
                self.model.depth_hidden_size // self.model.depth_num_attention_heads,  # head dim
                dtype=torch.bfloat16,
                device="cuda",
            )
        else:
            self.depth_prefill_wrappers = None
            self.depth_decode_wrappers = None
            self.depth_kv_cache = None

    def _initialize_cuda_graphs(self):
        """Initialize CUDA graphs for different batch sizes."""
        self.nvtx_range_push("cuda_graph_initialization")

        self.logger.info("Initializing CUDA graphs for LM decode phase...")

        # Create input buffers
        input_ids_buffer = torch.zeros(
            self.max_batch_size, self.model.n_codebooks, dtype=torch.int32, device=self.device
        )
        position_ids_buffer = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)
        input_features_buffer = torch.zeros(
            self.max_batch_size,
            self.model.n_codebooks,
            self.model.hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        input_masks_buffer = torch.zeros(
            self.max_batch_size, self.model.n_codebooks, dtype=torch.bool, device=self.device
        )

        # Create output buffer (assuming vocab size, will be adjusted based on model)
        logits_buffer = torch.zeros(
            self.max_batch_size,
            1 if self.has_depth_transformer else self.model.n_codebooks,  # TODO: revisit here
            self.model.vocab_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        backbone_hidden_states_buffer = torch.zeros(
            self.max_batch_size, self.model.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        # Store buffers
        self.cuda_graph_buffers = {
            "input_ids": input_ids_buffer,
            "position_ids": position_ids_buffer,
            "logits": logits_buffer,
            "input_features": input_features_buffer,
            "input_masks": input_masks_buffer,
            "backbone_hidden_states": backbone_hidden_states_buffer,
        }

        self.logger.info("Initializing CUDA graphs for decode phase...")

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing CUDA graph for batch size {batch_size}")

            # Create buffers for flashinfer inputs
            paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            paged_kv_indices = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            paged_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

            # Plan decode wrapper outside the graph capture
            self.decode_wrappers[batch_size].plan(
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # do warmup run to initialize the graph
            for _ in range(5):
                self.model.forward(
                    input_ids=self.cuda_graph_buffers["input_ids"][:batch_size],
                    position_ids=self.cuda_graph_buffers["position_ids"][:batch_size],
                    attn_wrapper=self.decode_wrappers[batch_size],
                    kv_cache=self.kv_cache,
                    input_features=self.cuda_graph_buffers["input_features"][:batch_size],
                    input_masks=self.cuda_graph_buffers["input_masks"][:batch_size],
                )
            torch.cuda.synchronize()

            # Create and capture CUDA graph
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                # Only capture model forward pass, NOT the attention wrapper planning
                if self.has_depth_transformer:
                    logits_output, backbone_hidden_states = self.model.forward(
                        input_ids=self.cuda_graph_buffers["input_ids"][:batch_size],
                        position_ids=self.cuda_graph_buffers["position_ids"][:batch_size],
                        attn_wrapper=self.decode_wrappers[batch_size],
                        kv_cache=self.kv_cache,
                        input_features=self.cuda_graph_buffers["input_features"][:batch_size],
                        input_masks=self.cuda_graph_buffers["input_masks"][:batch_size],
                    )

                    self.cuda_graph_buffers["logits"][:batch_size].copy_(logits_output)
                    self.cuda_graph_buffers["backbone_hidden_states"][:batch_size].copy_(backbone_hidden_states)

                else:
                    logits_output = self.model.forward(
                        input_ids=self.cuda_graph_buffers["input_ids"][:batch_size],
                        position_ids=self.cuda_graph_buffers["position_ids"][:batch_size],
                        attn_wrapper=self.decode_wrappers[batch_size],
                        kv_cache=self.kv_cache,
                        input_features=self.cuda_graph_buffers["input_features"][:batch_size],
                        input_masks=self.cuda_graph_buffers["input_masks"][:batch_size],
                    )

                    self.cuda_graph_buffers["logits"][:batch_size].copy_(logits_output)

            # Store the captured graph
            self.cuda_graphs_lm[batch_size] = graph

        self.logger.info("CUDA graphs for decode phase initialized.")

        self.logger.info("Initializing CUDA graphs for detokenization phase...")

        detokenize_input_buffer = torch.zeros(
            self.max_batch_size,
            self.model.detokenize_interval,
            self.model.n_codebooks,
            dtype=torch.int32,
            device=self.device,
        )

        detokenize_output_buffer = torch.zeros(
            self.max_batch_size,
            self.model.n_channels,
            self.model.output_audio_length,
            dtype=torch.float32,
            device=self.device,
        )

        # Add detokenization buffers to unified buffer dictionary
        self.cuda_graph_buffers.update(
            {
                "detokenize_input": detokenize_input_buffer,
                "detokenize_output": detokenize_output_buffer,
            }
        )

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing detokenization CUDA graph for batch size {batch_size}")

            # Warmup runs for detokenization
            for _ in range(5):
                audio_output = self.model.postprocess(self.cuda_graph_buffers["detokenize_input"][:batch_size])
            torch.cuda.synchronize()

            detokenize_graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(detokenize_graph):
                audio_output = self.model.postprocess(self.cuda_graph_buffers["detokenize_input"][:batch_size])

                self.cuda_graph_buffers["detokenize_output"][:batch_size].copy_(audio_output)

            self.cuda_graphs_detokenization[batch_size] = detokenize_graph

        self.logger.info("CUDA graphs for detokenization phase initialized.")

        if not self.has_depth_transformer:
            self.logger.info(f"CUDA graphs initialized for batch sizes: {list(self.cuda_graphs_lm.keys())}")
            return

        self.logger.info("Initializing CUDA graphs for depth transformer...")

        # We reserve input tensors with batch size of `2 * self.max_batch_size` since the first step of
        # depth transformer has sequence length of 2 per request.
        depth_hidden_states_buffer = torch.zeros(
            2 * self.max_batch_size, self.model.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        depth_position_ids_buffer = torch.zeros(2 * self.max_batch_size, dtype=torch.int32, device=self.device)

        depth_logits_buffer = torch.zeros(
            2 * self.max_batch_size, self.model.vocab_size, dtype=torch.bfloat16, device=self.device
        )

        # Add depth transformer buffers to the unified buffer dictionary
        self.cuda_graph_buffers.update(
            {
                "depth_hidden_states": depth_hidden_states_buffer,
                "depth_position_ids": depth_position_ids_buffer,
                "depth_logits": depth_logits_buffer,
            }
        )

        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            self.logger.info(f"Capturing depth CUDA graph for batch size {batch_size}")

            # Create buffers for flashinfer inputs for depth transformer
            depth_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device) * 2
            depth_paged_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
            depth_paged_kv_indices = torch.arange(batch_size, dtype=torch.int32, device=self.device)
            depth_paged_kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=self.device) * 2

            # Prefill graph capturing
            self.depth_prefill_wrappers[batch_size].plan(
                depth_qo_indptr,
                depth_paged_kv_indptr,
                depth_paged_kv_indices,
                depth_paged_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # Warmup runs for depth transformer
            for _ in range(5):
                self.model.depth_forward(
                    hidden_states=self.cuda_graph_buffers["depth_hidden_states"][: 2 * batch_size],
                    position_ids=self.cuda_graph_buffers["depth_position_ids"][: 2 * batch_size],
                    attn_wrapper=self.depth_prefill_wrappers[batch_size],
                    kv_cache=self.depth_kv_cache,
                )
            torch.cuda.synchronize()

            # Create and capture CUDA graph for depth transformer
            depth_graph_prefill = torch.cuda.CUDAGraph()

            with torch.cuda.graph(depth_graph_prefill):
                depth_logits_output = self.model.depth_forward(
                    hidden_states=self.cuda_graph_buffers["depth_hidden_states"][: 2 * batch_size],
                    position_ids=self.cuda_graph_buffers["depth_position_ids"][: 2 * batch_size],
                    attn_wrapper=self.depth_prefill_wrappers[batch_size],
                    kv_cache=self.depth_kv_cache,
                )

                self.cuda_graph_buffers["depth_logits"][: 2 * batch_size].copy_(depth_logits_output)

            # Store the captured depth graph
            self.cuda_graphs_depth_prefill[batch_size] = depth_graph_prefill

            # Decode graph capturing
            self.depth_decode_wrappers[batch_size].plan(
                depth_paged_kv_indptr,
                depth_paged_kv_indices,
                depth_paged_kv_last_page_len,
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # Warmup runs for depth transformer
            for _ in range(5):
                self.model.depth_forward(
                    hidden_states=self.cuda_graph_buffers["depth_hidden_states"][:batch_size],
                    position_ids=self.cuda_graph_buffers["depth_position_ids"][:batch_size],
                    attn_wrapper=self.depth_decode_wrappers[batch_size],
                    kv_cache=self.depth_kv_cache,
                )
            torch.cuda.synchronize()

            # Create and capture CUDA graph for depth transformer
            depth_graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(depth_graph):
                depth_logits_output = self.model.depth_forward(
                    hidden_states=self.cuda_graph_buffers["depth_hidden_states"][:batch_size],
                    position_ids=self.cuda_graph_buffers["depth_position_ids"][:batch_size],
                    attn_wrapper=self.depth_decode_wrappers[batch_size],
                    kv_cache=self.depth_kv_cache,
                )

                self.cuda_graph_buffers["depth_logits"][:batch_size].copy_(depth_logits_output)

            # Store the captured depth graph
            self.cuda_graphs_depth_decode[batch_size] = depth_graph

        self.logger.info("CUDA graphs for depth transformer decode phase initialized.")

        self.logger.info(f"CUDA graphs initialized for batch sizes: {list(self.cuda_graphs_lm.keys())}")
        self.nvtx_range_pop() # cuda_graph_initialization

    def _get_cuda_graph_batch_size(self, actual_batch_size: int) -> int:
        """
        Find the next valid CUDA graph batch size for padding.
        Always returns a valid batch size from the captured CUDA graphs.
        """
        for batch_size in self.cuda_graph_batch_sizes:
            if batch_size >= actual_batch_size:
                return batch_size
        # If actual batch size exceeds all captured sizes, use the largest one
        return max(self.cuda_graph_batch_sizes)

    def run_lm_prefill(self, requests: List[Request]):
        """
        Override parent's run_lm_prefill to add CUDA graph optimization.
        We use CUDA graph only for depth transformer.
        """
        self.nvtx_range_push(f"lm_prefill_bs{len(requests)}")
        lm_inputs = self._prepare_lm_inputs(requests)

        qo_indptr = lm_inputs["qo_indptr"]
        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]

        input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.int32)
        position_ids = torch.tensor(position_ids, device=self.device, dtype=torch.int32)

        batch_size = len(requests)

        # TODO: maybe model should have property about this?
        if input_masks[0] is not None:
            input_masks = torch.cat(input_masks, dim=0)
        else:
            input_masks = None

        # TODO: for zonos's purpose, the input_features has to be list of tensors for prefill.
        # This is not a good design and we should fix it.
        # if input_features[0] is not None:
        #     input_features = torch.cat(input_features, dim=0)
        # else:
        #     input_features = None

        qo_indptr_tensor = torch.tensor(qo_indptr, device=self.device, dtype=torch.int32)
        paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr, device=self.device, dtype=torch.int32)
        paged_kv_indices_tensor = torch.tensor(paged_kv_indices, device=self.device, dtype=torch.int32)
        paged_kv_last_page_len_tensor = torch.tensor(paged_kv_last_page_len, device=self.device, dtype=torch.int32)

        self.prefill_wrapper.plan(
            qo_indptr_tensor,
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        torch.cuda.synchronize()

        # prefill run
        self.nvtx_range_push("backbone_forward")
        if self.has_depth_transformer:
            logits, backbone_hidden_states = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.prefill_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

            output_ids, hidden_for_depth = self.model.sampling(
                logits=logits,
                hidden_states=backbone_hidden_states,
                requests=requests,
            )

            # Always use CUDA graphs with padding
            padded_batch_size = self._get_cuda_graph_batch_size(batch_size)

            self.nvtx_range_pop() # backbone_forward
            output_ids = self.run_lm_depth(output_ids, hidden_for_depth, requests, batch_size, padded_batch_size)

        else:
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.prefill_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

            self.nvtx_range_pop() # backbone_forward
            output_ids = self.model.sampling(
                logits=logits,
                requests=requests,
            )

        self.nvtx_range_pop() # lm_prefill

    def run_lm_decode(self, requests: List[Request]):
        """
        Override parent's run_lm_decode to add CUDA graph optimization with padding.
        """
        self.nvtx_range_push(f"lm_decode_bs{len(requests)}")
        lm_inputs = self._prepare_lm_inputs(requests)

        qo_indptr = lm_inputs["qo_indptr"]
        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]

        actual_batch_size = len(requests)
        padded_batch_size = self._get_cuda_graph_batch_size(actual_batch_size)

        # Pad inputs to match CUDA graph batch size
        if actual_batch_size < padded_batch_size:
            padding_size = padded_batch_size - actual_batch_size

            # Pad input tensors by repeating the last element
            input_ids.extend([input_ids[-1]] * padding_size)
            position_ids.extend([position_ids[-1]] * padding_size)

            if input_features[0] is not None:
                input_features.extend([input_features[-1]] * padding_size)
            if input_masks[0] is not None:
                input_masks.extend([input_masks[-1]] * padding_size)

            # Pad paged KV cache indices
            last_page_idx = paged_kv_indices[-1] if paged_kv_indices else 0
            last_page_len = paged_kv_last_page_len[-1] if paged_kv_last_page_len else 1

            for _ in range(padding_size):
                paged_kv_indices.append(last_page_idx)
                paged_kv_last_page_len.append(last_page_len)
                paged_kv_indptr.append(paged_kv_indptr[-1] + 1)

        self.logger.debug(f"Using CUDA graph with padded batch size {padded_batch_size} (actual: {actual_batch_size})")

        # Plan attention wrapper before CUDA graph
        paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr, device=self.device, dtype=torch.int32)
        paged_kv_indices_tensor = torch.tensor(paged_kv_indices, device=self.device, dtype=torch.int32)
        paged_kv_last_page_len_tensor = torch.tensor(paged_kv_last_page_len, device=self.device, dtype=torch.int32)

        self.decode_wrappers[padded_batch_size].plan(
            paged_kv_indptr_tensor,
            paged_kv_indices_tensor,
            paged_kv_last_page_len_tensor,
            torch.bfloat16,
        )
        torch.cuda.synchronize()

        graph = self.cuda_graphs_lm[padded_batch_size]

        self.cuda_graph_buffers["input_ids"][:padded_batch_size].copy_(
            torch.tensor(input_ids, device=self.device, dtype=torch.int32)
        )
        self.cuda_graph_buffers["position_ids"][:padded_batch_size].copy_(
            torch.tensor(position_ids, device=self.device, dtype=torch.int32)
        )

        # TODO: maybe model should have property about this?
        if input_masks[0] is not None:
            self.cuda_graph_buffers["input_masks"][:padded_batch_size].copy_(torch.cat(input_masks, dim=0))
        if input_features[0] is not None:
            self.cuda_graph_buffers["input_features"][:padded_batch_size].copy_(torch.cat(input_features, dim=0))

        # Replay the CUDA graph
        self.nvtx_range_push("cuda_graph_replay")
        graph.replay()
        self.nvtx_range_pop()

        # Copy output from buffer - only take the actual batch size, not padded
        logits = self.cuda_graph_buffers["logits"][:actual_batch_size]

        if self.has_depth_transformer:
            backbone_hidden_states = self.cuda_graph_buffers["backbone_hidden_states"][:actual_batch_size]

        # Sampling is not part of CUDA graph
        self.nvtx_range_push("sampling")
        if self.has_depth_transformer:
            output_ids, hidden_for_depth = self.model.sampling(
                logits=logits,
                hidden_states=backbone_hidden_states,
                requests=requests,
            )

            self.nvtx_range_pop() # sampling
            output_ids = self.run_lm_depth(output_ids, hidden_for_depth, requests, actual_batch_size, padded_batch_size)

        else:
            tick = time.time()
            output_ids = self.model.sampling(
                logits=logits,
                requests=requests,
            )
            self.nvtx_range_pop() # sampling

        self.nvtx_range_pop() # lm_decode

    def run_lm_depth(self, output_ids, hidden_for_depth, requests, actual_batch_size, padded_batch_size):
        """
        Shared depth transformer processing logic for both prefill and decode phases.
        Uses padding to make CUDA graphs always available.
        """
        self.nvtx_range_push(f"depth_transform_bs{actual_batch_size}")
        # Pad hidden_for_depth if necessary
        if actual_batch_size < padded_batch_size:
            padding_size = padded_batch_size - actual_batch_size
            # Pad by repeating the last hidden state
            last_hidden = hidden_for_depth[-1:].expand(padding_size, -1)
            hidden_for_depth = torch.cat([hidden_for_depth, last_hidden], dim=0)

        depth_position_ids = torch.tensor([0, 1] * padded_batch_size, device=self.device, dtype=torch.int32)
        depth_qo_indptr = torch.arange(padded_batch_size + 1, device=self.device, dtype=torch.int32) * 2
        depth_kv_indptr = torch.arange(padded_batch_size + 1, device=self.device, dtype=torch.int32)
        depth_kv_indices = torch.arange(padded_batch_size, device=self.device, dtype=torch.int32)
        depth_kv_last_page_len = torch.tensor([2] * padded_batch_size, device=self.device, dtype=torch.int32)
        self.depth_kv_cache.zero_()

        for i in range(1, self.model.depth_n_codebooks):
            if i > 1:
                self.depth_decode_wrappers[padded_batch_size].plan(
                    paged_kv_indptr=depth_kv_indptr,
                    paged_kv_indices=depth_kv_indices,
                    paged_kv_last_page_len=depth_kv_last_page_len,
                    dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()

                graph = self.cuda_graphs_depth_decode[padded_batch_size]

                self.cuda_graph_buffers["depth_hidden_states"][:padded_batch_size].copy_(hidden_for_depth)
                self.cuda_graph_buffers["depth_position_ids"][:padded_batch_size].copy_(depth_position_ids)

                self.nvtx_range_push("depth_decode_replay")
                graph.replay()
                self.nvtx_range_pop()

                # Only take outputs for actual batch size
                depth_logits = self.cuda_graph_buffers["depth_logits"][:actual_batch_size]

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

                # Re-pad hidden_for_depth for next iteration
                if actual_batch_size < padded_batch_size:
                    last_hidden = hidden_for_depth[-1:].expand(padded_batch_size - actual_batch_size, -1)
                    hidden_for_depth = torch.cat([hidden_for_depth, last_hidden], dim=0)

                depth_position_ids = torch.tensor([i + 1] * padded_batch_size, device=self.device, dtype=torch.int32)
                depth_qo_indptr = torch.arange(padded_batch_size + 1, device=self.device, dtype=torch.int32)
                depth_kv_last_page_len += 1

            else:
                self.depth_prefill_wrappers[padded_batch_size].plan(
                    qo_indptr=depth_qo_indptr,
                    paged_kv_indptr=depth_kv_indptr,
                    paged_kv_indices=depth_kv_indices,
                    paged_kv_last_page_len=depth_kv_last_page_len,
                    dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()

                graph = self.cuda_graphs_depth_prefill[padded_batch_size]

                self.cuda_graph_buffers["depth_hidden_states"][: 2 * padded_batch_size].copy_(hidden_for_depth)
                self.cuda_graph_buffers["depth_position_ids"][: 2 * padded_batch_size].copy_(depth_position_ids)

                self.nvtx_range_push("depth_prefill_replay")
                graph.replay()
                self.nvtx_range_pop()

                depth_logits = self.cuda_graph_buffers["depth_logits"][: 2 * padded_batch_size]
                # Get the actual batch size from the prefill outputs
                actual_qo_indptr = torch.arange(actual_batch_size + 1, device=self.device, dtype=torch.int32) * 2
                depth_logits = depth_logits[actual_qo_indptr[:-1] - 1]

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

                # Re-pad hidden_for_depth for next iteration
                if actual_batch_size < padded_batch_size:
                    last_hidden = hidden_for_depth[-1:].expand(padded_batch_size - actual_batch_size, -1)
                    hidden_for_depth = torch.cat([hidden_for_depth, last_hidden], dim=0)

                depth_position_ids = torch.tensor([i + 1] * padded_batch_size, device=self.device, dtype=torch.int32)
                depth_qo_indptr = torch.arange(padded_batch_size + 1, device=self.device, dtype=torch.int32)
                depth_kv_last_page_len += 1

        self.nvtx_range_pop() # depth_transform
        return output_ids

    def run_detokenize(self, requests: List[Request]):
        """
        Override parent's run_detokenize to add CUDA graph optimization with padding.
        """
        self.nvtx_range_push(f"detokenize_bs{len(requests)}")
        if len(requests) == 0:
            self.nvtx_range_pop()
            return

        actual_batch_size = len(requests)
        padded_batch_size = self._get_cuda_graph_batch_size(actual_batch_size)

        self.logger.debug(
            f"Using detokenization CUDA graph with padded batch size {padded_batch_size} (actual: {actual_batch_size})"
        )

        # Prepare token_ids the same way as parent method
        token_ids = []
        for req in requests:
            new_tokens = req.lm_output_audio_tokens[
                req.next_audio_decode_idx : req.next_audio_decode_idx + self.detokenize_interval
            ]

            if req.done_all:
                # exclude the last token since it is a stop token
                if len(new_tokens) > 1:
                    new_tokens = new_tokens[:-1]

            if len(new_tokens) < self.detokenize_interval:
                new_tokens.extend([new_tokens[-1]] * (self.detokenize_interval - len(new_tokens)))

            token_ids.append(new_tokens)

        # Pad token_ids to match CUDA graph batch size
        if actual_batch_size < padded_batch_size:
            padding_size = padded_batch_size - actual_batch_size
            # Pad by repeating the last token sequence
            token_ids.extend([token_ids[-1]] * padding_size)

        token_ids_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.int32)

        self.cuda_graph_buffers["detokenize_input"][:padded_batch_size].copy_(token_ids_tensor)

        graph = self.cuda_graphs_detokenization[padded_batch_size]

        self.nvtx_range_push("detokenize_replay")
        graph.replay()
        self.nvtx_range_pop()

        # Only take outputs for actual batch size
        audio_tensors = self.cuda_graph_buffers["detokenize_output"][:actual_batch_size]

        if self.needs_watermarking:
            for i in range(audio_tensors.shape[0]):
                audio_tensors[i, 0] = self.run_watermark(audio_tensors[i, 0], orig_sr=24000)

        # Process the audio the same way as parent method
        for i, req in enumerate(requests):
            audio = audio_tensors[i].detach().cpu().numpy()
            audio_int16 = (audio * 32767).astype(np.int16)

            last_chunk_len = len(
                req.lm_output_audio_tokens[
                    req.next_audio_decode_idx : req.next_audio_decode_idx + self.detokenize_interval
                ]
            )
            if last_chunk_len < self.detokenize_interval:
                # remove the padded audio
                audio_int16 = audio_int16[: int(audio_int16.shape[1] * last_chunk_len / self.detokenize_interval)]

            audio_bytes = audio_int16.tobytes()
            req.output_audio.put(audio_bytes)

            req.next_audio_decode_idx += self.detokenize_interval - self.detokenize_overlap

        self.nvtx_range_pop() # detokenize
        return
