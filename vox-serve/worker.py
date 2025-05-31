import time

import numpy as np 
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoTokenizer, EncodecFeatureExtractor

from .flashinfer_utils import FlashInferPrefillWrapper, FlashInferDecodeWrapper
from .model import OrpheusModel


class ModelWorker:

    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 8,
        top_p: float = 0.8,
        top_k: int = 2,
        temperature: float = 0.6,
        repetition_penalty: float = 1.3,
    ):

        # Load model
        self.model = OrpheusModel("canopylabs/orpheus-3b-0.1-ft", device="cuda")
        self.device = "cuda:0"

        # Store sampling and repetition parameters
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # Tensor to store offset values for each client
        self.offsets = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # Now, due to sliding window, only 1 page per client
        self.max_num_pages = max_batch_size
        self.page_size = 2048

        self.paged_kv_indptr_buffer = torch.zeros(max_batch_size + 1).to(self.device).to(torch.int32)
        self.paged_kv_indices_buffer = torch.zeros(self.max_num_pages).to(self.device).to(torch.int32)
        self.paged_kv_last_page_len_buffer = torch.zeros(max_batch_size).to(self.device).to(torch.int32)

        self.flashinfer_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.prefill_wrapper = FlashInferPrefillWrapper(
            attn_buffer=self.flashinfer_buffer,
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
        )
        self.decode_wrapper = FlashInferDecodeWrapper(
            attn_buffer=self.flashinfer_buffer,
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
            batch_size=max_batch_size,
            paged_kv_indptr_buffer=self.paged_kv_indptr_buffer,
            paged_kv_indices_buffer=self.paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=self.paged_kv_last_page_len_buffer,
        )

        self.kv_cache = torch.zeros(
            self.model.num_hidden_layers,
            self.max_num_pages,
            2, # K/V
            self.page_size,
            self.model.num_key_value_heads, # kv heads
            self.model.hidden_size // self.model.num_attention_heads, # head dim
            dtype=torch.bfloat16,
            device="cuda",
        )

        kv_cache_size = self.kv_cache.numel() * self.kv_cache.element_size()
        print(f"KV cache size: {kv_cache_size / 1024 / 1024:.2f} MB")

        self.qo_indptr = torch.arange(max_batch_size + 1).to(self.device).to(torch.int32)
        self.paged_kv_indptr = torch.arange(max_batch_size + 1).to(self.device).to(torch.int32)
        # initially, each batch is allocated one page
        self.paged_kv_indices = torch.arange(max_batch_size).to(self.device).to(torch.int32)
        self.paged_kv_last_page_len = torch.ones(max_batch_size).to(self.device).to(torch.int32)
        
        # self.graph_args = {
        #     "input_ids": torch.zeros(max_batch_size).to(self.device).to(torch.int32),
        #     "audio_input_ids": torch.zeros(self.num_audio_codebooks, max_batch_size).to(self.device).to(torch.int32),
        #     "position_ids": torch.zeros(max_batch_size).to(self.device).to(torch.int32),
        #     "exec_mask": torch.zeros(max_batch_size).to(self.device).to(torch.bool),
        #     "output": None,
        #     "mimi_input_values": torch.zeros(max_batch_size, 1, 1920).to(self.device).to(self.dtype),
        #     "mimi_exec_mask": torch.zeros(max_batch_size).to(self.device).to(torch.bool),
        #     "mimi_reset_mask": torch.zeros(max_batch_size).to(self.device).to(torch.bool),
        # }
        # # warm up
        # for _ in range(5):
        #     self.decode_wrapper.plan(
        #         # torch.arange(max_batch_size + 1, device=self.device, dtype=torch.int32),
        #         torch.arange(max_batch_size + 1, device=self.device, dtype=torch.int32),
        #         torch.arange(self.max_num_pages, device=self.device, dtype=torch.int32),
        #         torch.ones(max_batch_size, device=self.device, dtype=torch.int32),
        #         dtype=self.dtype,
        #     )
        #     torch.cuda.synchronize()
        #     self.model.forward(
        #         input_ids=self.graph_args["input_ids"],
        #         audio_input_ids=self.graph_args["audio_input_ids"],
        #         position_ids=self.graph_args["position_ids"],
        #         exec_mask=self.graph_args["exec_mask"],
        #         attn_wrapper=self.decode_wrapper,
        #         kv_cache=self.kv_cache,
        #         repetition_cache=self.repetition_cache,
        #         top_p=self.top_p,
        #         top_k=self.top_k,
        #         temperature=self.temperature,
        #         repetition_window=self.repetition_window,
        #         repetition_ngram=self.repetition_ngram,
        #         repetition_scale=self.repetition_scale,
        #     )
        #     self.mimi_model.encode(
        #         self.graph_args["mimi_input_values"],
        #         self.graph_args["mimi_exec_mask"],
        #         self.graph_args["mimi_reset_mask"],
        #     )
        
        # torch.cuda.synchronize()
        # self.decode_graph = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(self.decode_graph):
        #     self.graph_args["decode_outputs"] = self.model.forward(
        #         input_ids=self.graph_args["input_ids"],
        #         audio_input_ids=self.graph_args["audio_input_ids"],
        #         position_ids=self.graph_args["position_ids"],
        #         exec_mask=self.graph_args["exec_mask"],
        #         attn_wrapper=self.decode_wrapper,
        #         kv_cache=self.kv_cache,
        #         repetition_cache=self.repetition_cache,
        #         top_p=self.top_p,
        #         top_k=self.top_k,
        #         temperature=self.temperature,
        #         repetition_window=self.repetition_window,
        #         repetition_ngram=self.repetition_ngram,
        #         repetition_scale=self.repetition_scale,
        #     )
        # torch.cuda.synchronize()
        # self.decode_graph.replay()

        # torch.cuda.synchronize()
        # self.mimi_graph = torch.cuda.CUDAGraph()
        # with torch.cuda.graph(self.mimi_graph):
        #     self.graph_args["mimi_outputs"] = self.mimi_model.encode(
        #         self.graph_args["mimi_input_values"],
        #         self.graph_args["mimi_exec_mask"],
        #         self.graph_args["mimi_reset_mask"],
        #     )
        # self.mimi_graph.replay()
        # torch.cuda.synchronize()

        # self.kv_cache.zero_() 
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """
        Process a list of inference requests.
        
        Args:
            requests: A list of request objects, each containing:
                - audio_chunk: Audio data as numpy array
                - sample_rate: Sample rate of the audio
                - language: Target language for transcription
                - text_input_ids: Previous token IDs (optional)
                - is_first_chunk: Boolean indicating if this is the first chunk
                - Additional generation parameters may be included
                - control_type: Optional control message type ('join' or 'leave')
                
        Returns:
            Tensor containing the next token for each request
        """
        # prefill plan 
        input_ids, _ = self.model.preprocess(prompt)
        input_ids = input_ids.to(self.device).to(torch.int32)

        self.qo_indptr = torch.tensor([0, input_ids.shape[0]]).to(self.device).to(torch.int32)
        self.paged_kv_indptr = torch.tensor([0, 1]).to(self.device).to(torch.int32)
        # initially, each batch is allocated one page
        self.paged_kv_indices = torch.tensor([0]).to(self.device).to(torch.int32)
        self.paged_kv_last_page_len = torch.tensor([input_ids.shape[0]]).to(self.device).to(torch.int32)

        print(f"{self.qo_indptr=} {self.paged_kv_indptr=} {self.paged_kv_indices=} {self.paged_kv_last_page_len=}")

        self.prefill_wrapper.plan(
            self.qo_indptr, 
            self.paged_kv_indptr, 
            self.paged_kv_indices, 
            self.paged_kv_last_page_len, 
            torch.bfloat16
        )
        torch.cuda.synchronize()

        # prefill run 
        output_ids = self.model.forward(
            input_ids=input_ids,
            position_ids=torch.arange(input_ids.shape[0], device=self.device, dtype=torch.int32),
            attn_wrapper=self.prefill_wrapper,
            kv_cache=self.kv_cache,
        )
        print(f"{input_ids.shape=} {output_ids.shape=}")

        results = [output_ids[-1].item()]
        position_ids = torch.tensor([input_ids.shape[0] - 1], device=self.device, dtype=torch.int32)

        # decode for-loop
        for _ in range(512):
            # decode plan 
            self.paged_kv_last_page_len[0] += 1
            position_ids[0] += 1

            print(f"{self.qo_indptr=} {self.paged_kv_indptr=} {self.paged_kv_indices=} {self.paged_kv_last_page_len=}")

            self.decode_wrapper.plan(
                self.paged_kv_indptr, 
                self.paged_kv_indices, 
                self.paged_kv_last_page_len, 
                torch.bfloat16,
            )
            torch.cuda.synchronize()

            # decode run 
            output_ids = self.model.forward(
                input_ids=output_ids[-1:],
                position_ids=position_ids,
                attn_wrapper=self.decode_wrapper,
                kv_cache=self.kv_cache,
            )
            results.append(output_ids[-1].item())
        
        print(results)
        self.model.postprocess(results)


if __name__ == "__main__":
    worker = ModelWorker("canopylabs/orpheus-3b-0.1-ft", max_batch_size=1)
    start_time = time.time()
    worker("Man, the way social media has, um, completely changed how we interact is just wild, right?")
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")