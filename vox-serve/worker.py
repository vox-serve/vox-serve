import time
from typing import List, Dict
import queue

import numpy as np 
import torch
import torch.cuda.nvtx as nvtx

from .flashinfer_utils import FlashInferPrefillWrapper, FlashInferDecodeWrapper
from .model import load_model
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
        self.model = load_model(model_name, device="cuda")
        self.device = "cuda:0"

        # Store sampling and repetition parameters
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        
        # Tensor to store offset values for each client
        self.offsets = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

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

        self.has_depth_transformer = self.model.has_depth_transformer
        if self.has_depth_transformer:
            self.depth_attn_wrapper = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=self.model.depth_num_attention_heads,
                n_kv_head=self.model.depth_num_key_value_heads,
                n_state=self.model.depth_hidden_size,
                page_size=self.page_size,
            )
            self.depth_kv_cache = torch.zeros(
                self.model.depth_num_hidden_layers,
                self.max_num_pages,
                2, # K/V
                self.model.depth_n_codebooks,
                self.model.depth_num_key_value_heads, # kv heads
                self.model.depth_hidden_size // self.model.depth_num_attention_heads, # head dim
                dtype=torch.bfloat16,
                device="cuda",
            )
        else:
            self.depth_attn_wrapper = None
            self.depth_kv_cache = None

        # Initialize empty pages
        self.empty_pages = queue.Queue()
        for i in range(self.max_num_pages):
            self.empty_pages.put(i)
    
    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return self.model.detokenize_interval
    
    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return self.model.detokenize_overlap
    
    def _prepare_lm_inputs(self, requests: List[Request]):
        """Prepare inputs for the LM step."""
        # flashinfer inputs
        qo_indptr = [0] 
        paged_kv_indptr = [0] 
        paged_kv_indices = [] 
        paged_kv_last_page_len = [] 

        # necessary inference inputs
        input_ids = [] 
        position_ids = []

        # optional inference inputs 
        # TODO: these should be single tenosr, not list of tensors
        input_features = []
        input_masks = []

        # sampling inputs
        # TODO: sampling params, cfg scale
        repetition_cache = []

        for req in requests:
            if not req.done_lm_prefill:
                # prefill request
                preprocess_output = self.model.preprocess(req.prompt)
                req.input_tokens = preprocess_output.input_tokens

                if preprocess_output.input_features is not None:
                    req.input_features = preprocess_output.input_features
                
                if preprocess_output.input_masks is not None:
                    req.input_masks = preprocess_output.input_masks
                
                if preprocess_output.repetition_cache is not None:
                    req.repetition_cache = preprocess_output.repetition_cache
                
                input_features.append(req.input_features)
                input_masks.append(req.input_masks)
                repetition_cache.append(req.repetition_cache)
                
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

                input_features.append(req.input_features)
                input_masks.append(req.input_masks)
                repetition_cache.append(req.repetition_cache)

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
        
        return {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "input_features": input_features,
            "input_masks": input_masks,
        }
    
    def run_lm_prefill(self, requests: List[Request]):
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

            # assuming that the sequence length is 2 for the initial iteration of depth transformer. 
            # may need to change here for other models.
            depth_position_ids = torch.tensor([0, 1] * output_ids.shape[0], device=self.device)
            depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device) * 2
            depth_kv_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device)
            depth_kv_indices = torch.arange(output_ids.shape[0], device=self.device)
            depth_kv_last_page_len = torch.tensor([2] * output_ids.shape[0], device=self.device)
            self.depth_kv_cache.zero_()

            for i in range(1, self.model.depth_n_codebooks):
                self.depth_attn_wrapper.plan(
                    qo_indptr=depth_qo_indptr,
                    paged_kv_indptr=depth_kv_indptr, 
                    paged_kv_indices=depth_kv_indices, 
                    paged_kv_last_page_len=depth_kv_last_page_len,
                    dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()
                
                depth_logits = self.model.depth_forward(
                    input_ids=output_ids,
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=self.depth_attn_wrapper,
                    kv_cache=self.depth_kv_cache,
                )

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

                depth_position_ids = torch.tensor([i + 1] * output_ids.shape[0], device=self.device)
                depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device)
                depth_kv_last_page_len += 1
                
        else:
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.prefill_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

            output_ids = self.model.sampling(
                logits=logits,
                requests=requests,
            )

        for i, req in enumerate(requests):
            req.lm_output_tokens.append(output_ids[i].tolist())
        
        return 
    
    def run_lm_decode(self, requests: List[Request]):
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

        # decode run 
        if self.has_depth_transformer:
            logits, backbone_hidden_states = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.decode_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

            output_ids, hidden_for_depth = self.model.sampling(
                logits=logits,
                hidden_states=backbone_hidden_states,
                requests=requests,
            )

            depth_position_ids = torch.tensor([0, 1] * output_ids.shape[0], device=self.device)
            depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device) * 2
            depth_kv_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device)
            depth_kv_indices = torch.arange(output_ids.shape[0], device=self.device)
            depth_kv_last_page_len = torch.tensor([2] * output_ids.shape[0], device=self.device)
            self.depth_kv_cache.zero_()

            for i in range(1, self.model.depth_n_codebooks):
                self.depth_attn_wrapper.plan(
                    qo_indptr=depth_qo_indptr,
                    paged_kv_indptr=depth_kv_indptr, 
                    paged_kv_indices=depth_kv_indices, 
                    paged_kv_last_page_len=depth_kv_last_page_len,
                    dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()
                
                depth_logits = self.model.depth_forward(
                    input_ids=output_ids,
                    hidden_states=hidden_for_depth,
                    position_ids=depth_position_ids,
                    attn_wrapper=self.depth_attn_wrapper,
                    kv_cache=self.depth_kv_cache,
                )

                output_ids[:, i], hidden_for_depth = self.model.depth_sampling(
                    logits=depth_logits,
                    i_iteration=i,
                    requests=requests,
                )

                depth_position_ids = torch.tensor([i + 1] * output_ids.shape[0], device=self.device)
                depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device)
                depth_kv_last_page_len += 1
                
        else:
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.decode_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

            output_ids = self.model.sampling(
                logits=logits,
                requests=requests,
            )

        for i, req in enumerate(requests):
            req.lm_output_tokens.append(output_ids[i].tolist())

        return 

    def run_detokenize(self, requests: List[Request]):
        if len(requests) == 0:
            return

        token_ids = []
        for req in requests: 
            new_tokens = req.lm_output_tokens[req.next_audio_decode_idx : req.next_audio_decode_idx + self.detokenize_interval] 
            
            if req.done_all:
                # exclude the last token since it is a stop token
                if len(new_tokens) > 1:
                    new_tokens = new_tokens[:-1]

            if len(new_tokens) < self.detokenize_interval:
                new_tokens.extend([new_tokens[-1]] * (self.detokenize_interval - len(new_tokens)))
            
            token_ids.append(new_tokens)
        
        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.int32)

        audio_tensors = self.model.postprocess(token_ids)

        for i, req in enumerate(requests):
            audio = audio_tensors[i].detach().cpu().numpy()
            audio_int16 = (audio * 32767).astype(np.int16) 

            last_chunk_len = len(req.lm_output_tokens[req.next_audio_decode_idx : req.next_audio_decode_idx + self.detokenize_interval])
            if last_chunk_len < self.detokenize_interval:
                # remove the padded audio
                audio_int16 = audio_int16[:int(audio_int16.shape[1] * last_chunk_len / self.detokenize_interval)]

            audio_bytes = audio_int16.tobytes()
            req.output_audio.put(audio_bytes)

            req.next_audio_decode_idx += self.detokenize_interval - self.detokenize_overlap
        
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

    def is_finished(self, request: Request):
        # TODO: request-specific max_tokens
        return (
            self.model.is_stop_id(request.lm_output_tokens[-1]) 
            or request.next_position_id > self.model.max_tokens
        )