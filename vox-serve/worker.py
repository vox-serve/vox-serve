import time
from typing import List

import numpy as np 
import torch
import torch.cuda.nvtx as nvtx

from .flashinfer_utils import FlashInferPrefillWrapper, FlashInferDecodeWrapper
from .model import OrpheusModel
from .requests import Request

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
    
    def _prepare_prefill(self):
        """Prepare LM prefill kernel"""
        pass 
    
    def _prepare_decode(self):
        """Prepare LM decode kernel"""
        pass 

    def _prepare_detokenize(self):
        """Prepare audio detokenizer"""
    
    def run_lm_prefill(self, requests: List[Request]):
        self._prepare_prefill()
        pass 
    
    def run_lm_decode(self, requests: List[Request]):
        self._prepare_decode()
        pass

    def run_detokenize(self, requests: List[Request]):
        self._prepare_detokenize()
        pass 
    
    def __call__(
        self,
        prompt: str,
    ):
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

        results = [output_ids[-1].item()]
        position_ids = torch.tensor([input_ids.shape[0] - 1], device=self.device, dtype=torch.int32)
        token_buffer = [self.model.postprocess([output_ids[-1].item()])]

        # decode for-loop
        for _ in range(512):
            # decode plan 
            self.paged_kv_last_page_len[0] += 1
            position_ids[0] += 1

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
            token_buffer.append(self.model.postprocess([output_ids[-1].item()]))

            if len(token_buffer) % 7 == 0 and len(token_buffer) > 27:
                audio_samples = self.model.convert_to_audio(token_buffer[-28:])
                if audio_samples is not None: 
                    yield audio_samples 


if __name__ == "__main__":
    import wave 

    worker = ModelWorker("canopylabs/orpheus-3b-0.1-ft", max_batch_size=1)
    start_time = time.time()
    prompt = "Man, the way social media has, um, completely changed how we interact is just wild, right?"
    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0
        for audio_chunk in worker(prompt):
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")