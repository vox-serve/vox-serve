import time
from typing import List, Dict
import queue

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
        self.paged_kv_indices = torch.arange(max_batch_size).to(self.device).to(torch.int32)
        self.paged_kv_last_page_len = torch.ones(max_batch_size).to(self.device).to(torch.int32)

        # Initialize empty pages
        self.empty_pages = queue.Queue()
        for i in range(self.max_num_pages):
            self.empty_pages.put(i)
    
    def _prepare_lm_inputs(self, requests: List[Request]):
        """Prepare inputs for the LM step."""
        qo_indptr = [0] 
        paged_kv_indptr = [0] 
        paged_kv_indices = [] 
        paged_kv_last_page_len = [] 

        input_ids = [] 
        position_ids = []

        for req in requests:
            if not req.done_lm_prefill:
                # prefill request
                req.input_tokens, _ = self.model.preprocess(req.prompt)
                
                n_pages_to_allocate = (len(req.input_tokens) + self.page_size - 1) // self.page_size
                req.kv_token_len = len(req.input_tokens)

                req.kv_pages = [self.empty_pages.get() for _ in range(n_pages_to_allocate)]
                req.kv_last_page_len = len(req.input_tokens) % self.page_size
                if req.kv_last_page_len == 0:
                    req.kv_last_page_len = self.page_size

                qo_indptr.append(qo_indptr[-1] + len(req.input_tokens))
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                input_ids.extend(req.input_tokens)
                position_ids.extend([i for i in range(len(req.input_tokens))])
            
                req.next_position_id = len(req.input_tokens) + 1
                req.done_lm_prefill = True

            else:
                # decode request
                next_input_token = req.lm_output_tokens[-1]

                req.kv_token_len += 1
                req.kv_last_page_len += 1
                if req.kv_last_page_len > self.page_size:
                    req.kv_pages.append(self.empty_pages.get())
                    req.kv_last_page_len = 1
                
                qo_indptr.append(qo_indptr[-1] + 1)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                input_ids.append(next_input_token)
                position_ids.append(req.next_position_id)

                req.next_position_id += 1
        
        return qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, input_ids, position_ids

    def _process_lm_outputs(
        self, 
        requests: List[Request], 
        output_ids: torch.Tensor, 
        qo_indptr: List[int] = None, 
        is_decode: bool = False,
    ):
        """
        Process the output IDs from the model and update the requests.
        """
        # output_ids = self.model.postprocess(output_ids)
        if not is_decode:
            # prefill
            assert qo_indptr is not None 
            for i, qo_idx in enumerate(qo_indptr[1:]):
                requests[i].lm_output_tokens.append(output_ids[qo_idx - 1].item())
        else:
            # decode
            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i].item())
        
        return
    
    def run_lm_prefill(self, requests: List[Request]):
        qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, input_ids, position_ids = (
            self._prepare_lm_inputs(requests)
        )

        input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.int32)
        position_ids = torch.tensor(position_ids, device=self.device, dtype=torch.int32)
        
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
        output_ids = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_wrapper=self.prefill_wrapper,
            kv_cache=self.kv_cache,
        )

        self._process_lm_outputs(requests, output_ids, qo_indptr, is_decode=False)
        return 
    
    def run_lm_decode(self, requests: List[Request]):
        qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, input_ids, position_ids = (
            self._prepare_lm_inputs(requests)
        )

        input_ids = torch.tensor(input_ids, device=self.device, dtype=torch.int32)
        position_ids = torch.tensor(position_ids, device=self.device, dtype=torch.int32)
        
        # qo_indptr_tensor = torch.tensor(qo_indptr, device=self.device, dtype=torch.int32)
        paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr, device=self.device, dtype=torch.int32)
        paged_kv_indices_tensor = torch.tensor(paged_kv_indices, device=self.device, dtype=torch.int32)
        paged_kv_last_page_len_tensor = torch.tensor(paged_kv_last_page_len, device=self.device, dtype=torch.int32)

        self.decode_wrapper.plan(
            paged_kv_indptr_tensor, 
            paged_kv_indices_tensor, 
            paged_kv_last_page_len_tensor, 
            torch.bfloat16,
        )
        torch.cuda.synchronize()

        # prefill run 
        output_ids = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_wrapper=self.decode_wrapper,
            kv_cache=self.kv_cache,
        )

        self._process_lm_outputs(requests, output_ids, is_decode=True)
        return 

    def run_detokenize(self, requests: List[Request]):
        # naive implementation for now 

        for req in requests:
            if len(req.lm_output_tokens) % 7 == 0 and len(req.lm_output_tokens) > 27:
                audio_samples = self.model.convert_to_audio(req.lm_output_tokens[-28:])
                if audio_samples is not None:
                    req.output_audio.append(audio_samples)
                    req.is_audio_available = True
                else:
                    req.is_audio_available = False
            else:
                req.is_audio_available = False
        
        return

    def free_kv_cache(self, request: Request):
        """
        Free the KV cache pages that was used by the request.
        """
        if hasattr(request, 'kv_pages') and request.kv_pages:
            # Return all allocated pages back to the empty pages queue
            for page_idx in request.kv_pages:
                self.empty_pages.put(page_idx)
            
            # Clear the request's page allocations
            request.kv_pages = []
            request.kv_token_len = 0
            request.kv_last_page_len = 0