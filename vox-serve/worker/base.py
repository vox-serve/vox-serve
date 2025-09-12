import queue
from typing import Coroutine, List, Optional

import numpy as np
import torch
import torchaudio

from ..flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from ..model import load_model
from ..requests import LMInputs, Request
from ..utils import get_logger
from ..watermarker import silentcipher


class ModelWorker:
    def __init__(
        self,
        model_name: str,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_nvtx: bool = False,
    ):
        # Load model with sampling parameters
        self.model = load_model(
            model_name,
            device="cuda",
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            cfg_scale=cfg_scale,
            greedy=greedy,
        )
        self.device = "cuda:0"
        self.max_batch_size = max_batch_size
        self.logger = get_logger(__name__)

        # Set NVTX profiling based on parameter
        self.nvtx_enabled = enable_nvtx

        # Store sampling and repetition parameters
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.cfg_scale = cfg_scale

        # Tensor to store offset values for each client
        self.offsets = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)

        # Use CLI-provided values or defaults
        self.max_num_pages = max_num_pages
        self.page_size = page_size

        # Initialize empty pages
        self.empty_pages = queue.Queue()
        for i in range(self.max_num_pages):
            self.empty_pages.put(i)

        self.needs_watermarking = self.model.needs_watermarking
        if self.needs_watermarking:
            self.watermark_model = silentcipher.get_model(
                model_type="44.1k",
                device=self.device,
            )
            # TODO: This should be specified at server start time
            self.watermark_key = [11, 91, 60, 147, 209]

        self._prepare_attention_wrappers()

        # self.kv_cache.zero_()

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return self.model.detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return self.model.detokenize_overlap

    @property
    def supports_audio_input(self) -> bool:
        """Whether the model supports audio input."""
        return self.model.supports_audio_input

    @property
    def available_batch_sizes(self) -> Optional[List[int]]:
        """
        Return the available batch sizes supported by the model.
        For the base model worker, there is no restriction.
        """
        return None

    def _prepare_attention_wrappers(self):
        self.flashinfer_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.prefill_wrapper = FlashInferPrefillWrapper(
            attn_buffer=self.flashinfer_buffer,
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
            use_cuda_graph=False,
        )
        self.decode_wrapper = FlashInferDecodeWrapper(
            attn_buffer=self.flashinfer_buffer,
            n_qo_head=self.model.num_attention_heads,
            n_kv_head=self.model.num_key_value_heads,
            n_state=self.model.hidden_size,
            page_size=self.page_size,
            use_cuda_graph=False,
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
            self.depth_attn_wrapper = FlashInferPrefillWrapper(
                attn_buffer=self.flashinfer_buffer,
                n_qo_head=self.model.depth_num_attention_heads,
                n_kv_head=self.model.depth_num_key_value_heads,
                n_state=self.model.depth_hidden_size,
                page_size=self.page_size,
                max_batch_size=self.max_batch_size,
                use_cuda_graph=False,
            )
            self.depth_kv_cache = torch.zeros(
                self.model.depth_num_hidden_layers,
                self.max_num_pages,
                2,  # K/V
                self.model.depth_n_codebooks,
                self.model.depth_num_key_value_heads,  # kv heads
                self.model.depth_hidden_size // self.model.depth_num_attention_heads,  # head dim
                dtype=torch.bfloat16,
                device="cuda",
            )
        else:
            self.depth_attn_wrapper = None
            self.depth_kv_cache = None

    def prepare_lm_inputs(self, requests: List[Request]) -> Optional[LMInputs]:
        """Prepare inputs for the LM step."""
        if len(requests) == 0:
            return None

        # flashinfer inputs
        qo_indptr = [0]
        paged_kv_indptr = [0]
        paged_kv_indices = []
        paged_kv_last_page_len = []

        # necessary inference inputs
        input_ids_list = []
        position_ids_list = []

        # optional inference inputs
        input_features_list = []
        input_masks_list = []

        # sampling inputs
        repetition_cache_list = []

        # Determine if any request needs prefill
        is_prefill = any(not req.done_lm_prefill for req in requests)

        for req in requests:
            if not req.done_lm_prefill:
                # prefill request
                preprocess_output = self.model.preprocess(prompt=req.prompt, audio_path=req.audio_path)
                req.input_tokens = preprocess_output.input_tokens
                # Set input length based on prepared input tokens
                if req.input_tokens is not None:
                    req.input_length = req.input_tokens.shape[0]

                if preprocess_output.input_features is not None:
                    req.input_features = preprocess_output.input_features

                if preprocess_output.input_masks is not None:
                    req.input_masks = preprocess_output.input_masks

                if preprocess_output.repetition_cache is not None:
                    req.repetition_cache = preprocess_output.repetition_cache

                # input_ids.append(req.input_tokens.to(self.device, non_blocking=True)) # (seq, codebook)
                input_ids_list.append(req.input_tokens.to(self.device, non_blocking=True)) # (seq, codebook)
                position_ids_list.extend([i for i in range(len(req.input_tokens))])
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                n_pages_to_allocate = (len(req.input_tokens) + self.page_size - 1) // self.page_size
                req.kv_token_len = len(req.input_tokens)

                req.kv_pages = [self.empty_pages.get_nowait() for _ in range(n_pages_to_allocate)]
                req.kv_last_page_len = len(req.input_tokens) % self.page_size
                if req.kv_last_page_len == 0:
                    req.kv_last_page_len = self.page_size

                qo_indptr.append(qo_indptr[-1] + len(req.input_tokens))
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                req.next_position_id = len(req.input_tokens) + 1
                req.done_lm_prefill = True

            else:
                # decode request
                input_ids_list.append(req.input_tokens) # (1, codebook)
                input_features_list.append(req.input_features)
                input_masks_list.append(req.input_masks)
                repetition_cache_list.append(req.repetition_cache)

                req.kv_token_len += 1
                req.kv_last_page_len += 1
                if req.kv_last_page_len > self.page_size:
                    req.kv_pages.append(self.empty_pages.get_nowait())
                    req.kv_last_page_len = 1

                qo_indptr.append(qo_indptr[-1] + 1)
                paged_kv_indptr.append(paged_kv_indptr[-1] + len(req.kv_pages))
                paged_kv_indices.extend(req.kv_pages)
                paged_kv_last_page_len.append(req.kv_last_page_len)

                position_ids_list.append(req.next_position_id)

                req.next_position_id += 1
            
            req.audio_decode_idx = req.next_audio_decode_idx.copy()

        # Allocate tensors for GPU computation
        input_ids = torch.cat(input_ids_list, dim=0)
        position_ids = torch.tensor(position_ids_list, device=self.device, dtype=torch.int32)

        # Prepare input_masks and input_features as single tensors
        if self.model.needs_input_masks and input_masks_list:
            input_masks = torch.cat([mask for mask in input_masks_list if mask is not None], dim=0)
        else:
            input_masks = None

        if self.model.needs_input_features and input_features_list:
            input_features = torch.cat([features for features in input_features_list if features is not None], dim=0)
        else:
            input_features = None

        # Prepare repetition cache as stacked tensor if model uses repetition penalty
        if self.model.use_repetition_penalty and repetition_cache_list:
            repetition_cache = torch.stack([cache for cache in repetition_cache_list if cache is not None], dim=0)
        else:
            repetition_cache = None

        return {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "input_features": input_features,
            "input_masks": input_masks,
            "repetition_cache": repetition_cache,
            "is_prefill": is_prefill,
        }

    def run_lm_prefill(self, requests: List[Request], lm_inputs: LMInputs) -> Optional[Coroutine]:
        if len(requests) == 0:
            return None

        qo_indptr = lm_inputs["qo_indptr"]
        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]
        repetition_cache = lm_inputs["repetition_cache"]

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

            # select last token for each request for prefill
            if getattr(self.prefill_wrapper, "qo_indptr", None) is not None:
                logits = logits[self.prefill_wrapper.qo_indptr[:-1] - 1]
                backbone_hidden_states = backbone_hidden_states[self.prefill_wrapper.qo_indptr[:-1] - 1]

            output_ids, hidden_for_depth = self.model.sampling(
                logits=logits,
                hidden_states=backbone_hidden_states,
                requests=requests,
                repetition_cache=repetition_cache,
            )

            # assuming that the sequence length is 2 for the initial iteration of depth transformer.
            # may need to change here for other models.
            depth_position_ids = torch.tensor([0, 1] * output_ids.shape[0], device=self.device, dtype=torch.int32)
            depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32) * 2
            depth_kv_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32)
            depth_kv_indices = torch.arange(output_ids.shape[0], device=self.device, dtype=torch.int32)
            depth_kv_last_page_len = torch.tensor([2] * output_ids.shape[0], device=self.device, dtype=torch.int32)
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

                depth_position_ids = torch.tensor([i + 1] * output_ids.shape[0], device=self.device, dtype=torch.int32)
                depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32)
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

            # select last token for each request for prefill
            if getattr(self.prefill_wrapper, "qo_indptr", None) is not None:
                logits = logits[self.prefill_wrapper.qo_indptr[:-1] - 1]

            output_ids, task = self.model.sampling(
                logits=logits,
                requests=requests,
                repetition_cache=repetition_cache,
            )

            return task

    def run_lm_decode(self, requests: List[Request], lm_inputs: LMInputs) -> Optional[Coroutine]:
        """
        Run LM decode step for the given requests.
        Base implementation without CUDA graph optimization.
        """
        if len(requests) == 0:
            return None

        paged_kv_indptr = lm_inputs["paged_kv_indptr"]
        paged_kv_indices = lm_inputs["paged_kv_indices"]
        paged_kv_last_page_len = lm_inputs["paged_kv_last_page_len"]
        input_ids = lm_inputs["input_ids"]
        position_ids = lm_inputs["position_ids"]
        input_features = lm_inputs["input_features"]
        input_masks = lm_inputs["input_masks"]
        repetition_cache = lm_inputs["repetition_cache"]

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

        # Run decode
        if self.has_depth_transformer:
            logits, backbone_hidden_states = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.decode_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )
        else:
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_wrapper=self.decode_wrapper,
                kv_cache=self.kv_cache,
                input_features=input_features,
                input_masks=input_masks,
            )

        # Sampling
        if self.has_depth_transformer:
            output_ids, hidden_for_depth = self.model.sampling(
                logits=logits,
                hidden_states=backbone_hidden_states,
                requests=requests,
                repetition_cache=repetition_cache,
            )

            depth_position_ids = torch.tensor([0, 1] * output_ids.shape[0], device=self.device, dtype=torch.int32)
            depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32) * 2
            depth_kv_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32)
            depth_kv_indices = torch.arange(output_ids.shape[0], device=self.device, dtype=torch.int32)
            depth_kv_last_page_len = torch.tensor([2] * output_ids.shape[0], device=self.device, dtype=torch.int32)
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

                depth_position_ids = torch.tensor([i + 1] * output_ids.shape[0], device=self.device, dtype=torch.int32)
                depth_qo_indptr = torch.arange(output_ids.shape[0] + 1, device=self.device, dtype=torch.int32)
                depth_kv_last_page_len += 1

        else:
            output_ids, task = self.model.sampling(
                logits=logits,
                requests=requests,
                repetition_cache=repetition_cache,
            )

            return task

    def run_detokenize(self, requests: List[Request]):
        if len(requests) == 0:
            return

        # Collect all chunks from all requests
        token_ids = []
        request_chunk_mapping = []  # Track which request each chunk belongs to
        
        for req_idx, req in enumerate(requests):
            # Process multiple chunks from the same request if available
            for chunk_idx in range(len(req.audio_decode_idx)):
                decode_idx = req.audio_decode_idx[chunk_idx]
                new_tokens = req.lm_output_audio_tokens[
                    decode_idx : decode_idx + self.detokenize_interval
                ]

                if len(new_tokens) < self.detokenize_interval:
                    new_tokens.extend([new_tokens[-1]] * (self.detokenize_interval - len(new_tokens)))

                token_ids.append(new_tokens)
                request_chunk_mapping.append((req_idx, chunk_idx))

        if not token_ids:
            return

        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.int32)

        audio_tensors = self.model.postprocess(token_ids)
        self.logger.debug("Audio tensors: %s", audio_tensors)

        if self.needs_watermarking:
            for i in range(audio_tensors.shape[0]):
                audio_tensors[i, 0] = self.run_watermark(audio_tensors[i, 0], orig_sr=24000)

        # Process each chunk and assign to the corresponding request
        for i, (req_idx, chunk_idx) in enumerate(request_chunk_mapping):
            req = requests[req_idx]
            decode_idx = req.audio_decode_idx[chunk_idx]
            
            audio = audio_tensors[i].detach().cpu().numpy()
            audio_int16 = (audio * 32767).astype(np.int16)

            last_chunk_len = len(
                req.lm_output_audio_tokens[
                    decode_idx : decode_idx + self.detokenize_interval
                ]
            )
            if last_chunk_len < self.detokenize_interval:
                # remove the padded audio
                audio_int16 = audio_int16[: int(audio_int16.shape[1] * last_chunk_len / self.detokenize_interval)]

            audio_bytes = audio_int16.tobytes()
            req.output_audio.put(audio_bytes)

        # Check if any request is completely done
        for req in requests:
            if req.done_lm_generation:
                all_chunks_done = True
                for decode_idx in req.audio_decode_idx:
                    if decode_idx + self.detokenize_interval < len(req.lm_output_audio_tokens):
                        all_chunks_done = False
                        break
                if all_chunks_done:
                    req.done_all = True

        return

    def run_watermark(self, audio_tensor: torch.Tensor, orig_sr: int = 24000):
        """
        Run watermarking on the given audio array.
        """
        assert self.needs_watermarking

        audio_array_44khz = torchaudio.functional.resample(audio_tensor, orig_freq=orig_sr, new_freq=44100)

        # Run watermarking
        encoded, _ = self.watermark_model.encode_wav(
            audio_array_44khz, 44100, self.watermark_key, calc_sdr=False, message_sdr=36
        )

        encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=orig_sr)

        return encoded

    def nvtx_range_push(self, name: str):
        """
        Push an NVTX range with CUDA synchronization if profiling is enabled.
        Does nothing if NVTX profiling is disabled.

        Args:
            name: Name of the NVTX range
        """
        if self.nvtx_enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(name)

    def nvtx_range_pop(self):
        """
        Pop an NVTX range with CUDA synchronization if profiling is enabled.
        Does nothing if NVTX profiling is disabled.
        """
        if self.nvtx_enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

    def free_kv_cache(self, request: Request):
        """
        Free the KV cache pages that was used by the request.
        """
        if hasattr(request, "kv_pages") and request.kv_pages:
            # Return all allocated pages back to the empty pages queue
            for page_idx in request.kv_pages:
                self.empty_pages.put(page_idx)

            # Clear the request's page allocations
            request.kv_pages = []
            request.kv_token_len = 0
            request.kv_last_page_len = 0

    # def do_detokenize(self, request: Request):
    #     """
    #     Check if the request is ready for detokenization.
    #     """
    #     return len(request.lm_output_audio_tokens) - request.next_audio_decode_idx >= self.detokenize_interval
