from typing import Union

import flashinfer
import torch

from .utils import get_logger

logger = get_logger(__name__)


# Toggle: fuse the two scatter writes (K, V -> paged KV cache) into one Triton kernel.
# The cuDNN-default path is two index_elementwise_kernel launches per layer; this collapses
# to one. ~3% of decode GPU time at bs=64. Flip to False for the original behaviour.
FUSE_KV_SCATTER = True

# FlashInfer decode-wrapper params. Override via env vars to avoid recompiling.
#   VOXSERVE_FI_USE_TC: "1" enables tensor cores in the decode wrapper (current default)
#   VOXSERVE_FI_BACKEND: passthrough to BatchDecodeWithPagedKVCacheWrapper(backend=...).
#     Options: "auto" (default), "fa2", "trtllm-gen".  trtllm-gen requires kv_layout=HND
#     and tuple kv-cache layout — not currently wired up.
import os as _fi_os
FI_DECODE_USE_TC = bool(int(_fi_os.environ.get("VOXSERVE_FI_USE_TC", "1")))
FI_DECODE_BACKEND = _fi_os.environ.get("VOXSERVE_FI_BACKEND", "auto")


_kv_scatter_kernel = None


def _get_kv_scatter_kernel():
    global _kv_scatter_kernel
    if _kv_scatter_kernel is not None:
        return _kv_scatter_kernel if _kv_scatter_kernel else None
    try:
        import triton
        import triton.language as tl
    except ImportError:
        _kv_scatter_kernel = False
        return None

    @triton.jit
    def _kernel(
        k_ptr, v_ptr, kv_cache_ptr,
        pages_ptr, positions_ptr,
        N, page_size, two_p_size_pH_pD, p_size_pH_pD, pH_pD,
        BLOCK: tl.constexpr,
    ):
        # one program per (token, head*head_dim block)
        tid = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < pH_pD

        # source offsets: token-contiguous k,v of shape (N, n_heads, head_dim) -> flat
        src = tid * pH_pD + offs
        k = tl.load(k_ptr + src, mask=mask)
        v = tl.load(v_ptr + src, mask=mask)

        page = tl.load(pages_ptr + tid)
        pos = tl.load(positions_ptr + tid)

        # kv_cache layout: (n_pages, 2, page_size, n_heads, head_dim)
        # element at [page, c, pos, h, d] = ((page*2 + c) * page_size + pos) * pH_pD + h*head_dim+d
        base_k = (page * two_p_size_pH_pD) + 0 * p_size_pH_pD + pos * pH_pD
        base_v = (page * two_p_size_pH_pD) + 1 * p_size_pH_pD + pos * pH_pD
        tl.store(kv_cache_ptr + base_k + offs, k, mask=mask)
        tl.store(kv_cache_ptr + base_v + offs, v, mask=mask)

    def launch(kv_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               pages: torch.Tensor, positions: torch.Tensor):
        # k, v: (N, n_heads, head_dim). kv_cache: (n_pages, 2, page_size, n_heads, head_dim).
        N, n_heads, head_dim = k.shape
        page_size = kv_cache.shape[2]
        pH_pD = n_heads * head_dim
        p_size_pH_pD = page_size * pH_pD
        two_p_size_pH_pD = 2 * p_size_pH_pD
        BLOCK = 1024
        if pH_pD <= 1024:
            BLOCK = triton.next_power_of_2(pH_pD)
        grid = (N, triton.cdiv(pH_pD, BLOCK))
        _kernel[grid](
            k.contiguous(), v.contiguous(), kv_cache,
            pages.to(torch.int32).contiguous(), positions.to(torch.int32).contiguous(),
            N, page_size, two_p_size_pH_pD, p_size_pH_pD, pH_pD,
            BLOCK=BLOCK,
        )

    _kv_scatter_kernel = launch
    return launch


class FlashInferPrefillWrapper:
    def __init__(
        self,
        attn_buffer: torch.Tensor,
        n_qo_head: int,
        n_kv_head: int,
        n_state: int,
        page_size: int,
        batch_size: int = None,
        max_seq_len: int = None,
        device: torch.device = torch.device("cuda"),
        qo_indptr_buffer: torch.Tensor = None,
        paged_kv_indptr_buffer: torch.Tensor = None,
        paged_kv_indices_buffer: torch.Tensor = None,
        paged_kv_last_page_len_buffer: torch.Tensor = None,
        use_cuda_graph: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        if self.use_cuda_graph:
            assert self.batch_size is not None, "batch_size must be specified for cuda graph optimization"
            assert max_seq_len is not None, "max_seq_len must be specified for cuda graph optimization"
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                attn_buffer,
                "NHD",
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=qo_indptr_buffer,
                paged_kv_indptr_buf=paged_kv_indptr_buffer,
                paged_kv_indices_buf=paged_kv_indices_buffer,
                paged_kv_last_page_len_buf=paged_kv_last_page_len_buffer,
            )
            self.token_to_page = torch.zeros(max_seq_len, dtype=torch.long, device=device)
            self.token_to_cache = torch.zeros(max_seq_len, dtype=torch.long, device=device)

        else:
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                attn_buffer,
                "NHD",
            )

        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.n_state = n_state
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size

    def plan(
        self,
        qo_indptr: torch.Tensor,  # [n_req + 1]
        paged_kv_indptr: torch.Tensor,  # [n_req + 1]
        paged_kv_indices: torch.Tensor,  # [# active pages]
        paged_kv_last_page_len: torch.Tensor,  # [n_req]
        dtype: torch.dtype = torch.float16,
    ):
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.n_qo_head,
            num_kv_heads=self.n_kv_head,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        self.qo_indptr = qo_indptr
        self.paged_kv_indptr = paged_kv_indptr
        self.paged_kv_indices = paged_kv_indices
        self.paged_kv_last_page_len = paged_kv_last_page_len

        n_req = qo_indptr.shape[0] - 1

        # Per-request counts
        starts = qo_indptr[:-1].to(torch.int32)                     # [n_req]
        lens   = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)   # [n_req]
        total_tokens = int(lens.sum().item())

        # Pages/lengths AFTER append
        num_pages_after = (paged_kv_indptr[1:] - paged_kv_indptr[:-1]).to(torch.int32) # [n_req]
        kv_len_after = (num_pages_after - 1) * self.page_size + paged_kv_last_page_len # [n_req], int32

        # Flatten to per-token
        seg = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int32, device=lens.device), lens)  # [T]
        intra = (
            torch.arange(
            total_tokens,
            dtype=torch.int32,
            device=lens.device,
            )
            - torch.repeat_interleave(starts, lens)
        )

        # Starting index of the newly appended run (per request), then absolute index per token
        start_new = kv_len_after[seg] - lens[seg]                # [T], int32
        g = start_new + intra                                    # [T], int32

        # Map to page + offset using the (post-append) page table
        page_off = torch.div(g, self.page_size, rounding_mode='floor').to(torch.int32)
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = (paged_kv_indptr[:-1])[seg] + page_off    # [T], int32

        if self.use_cuda_graph:
            self.token_to_page[:total_tokens] = paged_kv_indices[abs_page_ptr].to(self.device)
            self.token_to_cache[:total_tokens] = off_in_page.to(self.device)
            self.token_to_page[total_tokens:] = -1
            self.token_to_cache[total_tokens:] = -1
        else:
            self.token_to_page = paged_kv_indices[abs_page_ptr].to(self.device)
            self.token_to_cache = off_in_page.to(self.device)


    @torch.compiler.disable
    def run(self, q, kv_cache):
        if not self.use_cuda_graph:
            if q.isnan().any() or kv_cache.isnan().any():
                logger.warning("NaN detected in input tensors!")
        return self.attn_wrapper.run(q, kv_cache)

    def set_kv_cache(self, kv_cache, k, v):
        """
        kv_cache : torch.Tensor, shape = (n_pages, 2, page_size, n_heads, head_dim)
        k, v   : torch.Tensor, shape = (n_token, n_heads, head_dim)
        """
        page_idx = self.token_to_page  # (total_tokens,)
        cache_idx = self.token_to_cache  # (total_tokens,)

        if FUSE_KV_SCATTER:
            fn = _get_kv_scatter_kernel()
            if fn is not None:
                fn(kv_cache, k, v, page_idx, cache_idx)
                return
        kv_cache[page_idx, 0, cache_idx] = k  # keys
        kv_cache[page_idx, 1, cache_idx] = v  # values



class FlashInferDecodeWrapper:
    def __init__(
        self,
        attn_buffer: torch.Tensor,
        n_qo_head: int,
        n_kv_head: int,
        n_state: int,
        page_size: int,
        batch_size: int = None,
        device: torch.device = torch.device("cuda"),
        paged_kv_indptr_buffer: torch.Tensor = None,
        paged_kv_indices_buffer: torch.Tensor = None,
        paged_kv_last_page_len_buffer: torch.Tensor = None,
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = True,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.batch_size = batch_size

        # Honour the env-var overrides for quick A/B testing without rebuilding.
        _use_tc = FI_DECODE_USE_TC if use_tensor_cores else False
        _backend = FI_DECODE_BACKEND
        logger.info(f"FlashInfer decode wrapper: backend={_backend}, use_tensor_cores={_use_tc}")
        self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            attn_buffer,
            "NHD",
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=_use_tc,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
            backend=_backend,
        )

        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.n_state = n_state
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size

        if self.use_cuda_graph:
            assert self.batch_size is not None, "batch_size must be specified for cuda graph optimization"
            self.kv_cache_locations = torch.zeros((self.batch_size, 2), dtype=torch.long, device=self.device)

    def plan(
        self,
        # qo_indptr: torch.Tensor, # [n_req + 1]
        paged_kv_indptr: torch.Tensor,  # [n_req + 1]
        paged_kv_indices: torch.Tensor,  # [# active pages]
        paged_kv_last_page_len: torch.Tensor,  # [n_req]
        dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = paged_kv_indptr.shape[0] - 1

        self.attn_wrapper.plan(
            indptr=paged_kv_indptr,
            indices=paged_kv_indices,
            last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.n_qo_head,
            num_kv_heads=self.n_kv_head,
            head_dim=self.head_dim,
            page_size=self.page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        # self.qo_indptr = qo_indptr
        self.paged_kv_indptr = paged_kv_indptr
        self.paged_kv_indices = paged_kv_indices
        self.paged_kv_last_page_len = paged_kv_last_page_len

        # Vectorize the computation of page indices:
        # For each request i, the page index is given by the last element in the page pointer range:
        page_idx = paged_kv_indices[paged_kv_indptr[1:] - 1]  # shape: (n_req,)
        # And the corresponding position index is just the last token's position:
        pos_idx = paged_kv_last_page_len - 1  # shape: (n_req,)

        # Store as a single tensor of shape (n_req, 2) [page_idx, pos_idx]
        if self.use_cuda_graph:
            self.kv_cache_locations[: self.batch_size].copy_(torch.stack([page_idx, pos_idx], dim=1))
        else:
            self.kv_cache_locations = torch.stack([page_idx, pos_idx], dim=1).to(self.device)


    @torch.compiler.disable
    def run(self, q, kv_cache):
        return self.attn_wrapper.run(q, kv_cache)

    def set_kv_cache(self, kv_cache, k, v):
        """
        kv_cache : torch.Tensor, shape = (n_pages, 2, page_size, n_heads, head_dim)
        k, v   : torch.Tensor, shape = (n_req, n_heads, head_dim)
        """
        pages = self.kv_cache_locations[: self.batch_size, 0].long()
        positions = self.kv_cache_locations[: self.batch_size, 1].long()

        if FUSE_KV_SCATTER:
            fn = _get_kv_scatter_kernel()
            if fn is not None:
                fn(kv_cache, k, v, pages, positions)
                return
        kv_cache[pages, 0, positions] = k
        kv_cache[pages, 1, positions] = v



FlashInferWrapper = Union[FlashInferPrefillWrapper, FlashInferDecodeWrapper]


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Wrapper for FlashInfer RMSNorm operation.

    Args:
        hidden_states: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Epsilon value for numerical stability

    Returns:
        Normalized tensor of the same shape as hidden_states
    """
    return flashinfer.norm.rmsnorm(
        input=hidden_states,
        weight=weight,
        eps=eps,
    )


def apply_rope_pos_ids(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0,
    interleave: bool = False,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for FlashInfer RoPE application with position IDs.

    Args:
        query_states: Query states tensor
        key_states: Key states tensor
        position_ids: Position IDs tensor
        rope_scale: Rope scaling factor
        rope_theta: Rope theta parameter
        interleave: Whether to interleave the RoPE
        **kwargs: Additional parameters for specific RoPE variants

    Returns:
        Tuple of (rotated_query_states, rotated_key_states)
    """
    # Filter out kwargs that are meant for specific RoPE variants
    llama31_params = {}
    other_params = {}

    for key, value in kwargs.items():
        if key in ['low_freq_factor', 'high_freq_factor', 'old_context_len']:
            llama31_params[key] = value
        else:
            other_params[key] = value

    # Use LLaMA 3.1 variant if specific parameters are provided
    if llama31_params:
        return flashinfer.rope.apply_llama31_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            interleave=interleave,
            **llama31_params
        )
    else:
        return flashinfer.rope.apply_rope_pos_ids(
            query_states,
            key_states,
            pos_ids=position_ids,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            interleave=interleave,
            **other_params
        )
