import torch 
import flashinfer
from typing import Union

class FlashInferPrefillWrapper():
    def __init__(
        self, 
        attn_buffer: torch.Tensor,
        n_qo_head: int, 
        n_kv_head: int,
        n_state: int,
        page_size: int,
        max_batch_size: int = None,
        device: torch.device = torch.device("cuda"),
        qo_indptr_buf: torch.Tensor = None,
        paged_kv_indptr_buf: torch.Tensor = None,
        paged_kv_indices_buf: torch.Tensor = None,
        paged_kv_last_page_len_buf: torch.Tensor = None,
        use_cuda_graph: bool = False, 
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        # self.attn_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        # self.seq_len = seq_len # number of tokens, not number of batches

        if self.use_cuda_graph:
            # used for depth transformer
            assert max_batch_size is not None, "max_batch_size must be specified for cuda graph optimization"
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                attn_buffer, "NHD",
                use_cuda_graph=use_cuda_graph,
                qo_indptr_buf=qo_indptr_buf,
                paged_kv_indptr_buf=paged_kv_indptr_buf,
                paged_kv_indices_buf=paged_kv_indices_buf,
                paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
            )
            # TODO: support the sequence length per request of >1, such as the first step in depth transformer
            # or transformers inside Mimi
            self.token_to_page = torch.zeros(max_batch_size, dtype=torch.long, device=device)
            self.token_to_cache = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        else:
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                attn_buffer, "NHD",
            )

        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.n_state = n_state
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size
    
    def plan(
        self, 
        qo_indptr: torch.Tensor, # [n_req + 1] 
        paged_kv_indptr: torch.Tensor, # [n_req + 1]
        paged_kv_indices: torch.Tensor, # [# active pages]
        paged_kv_last_page_len: torch.Tensor, # [n_req]
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
        
        # Precompute KV cache update locations as a single tensor
        # Format: [start_idx, end_idx, page_idx, cache_start, cache_end, n_new_token]
        n_req = qo_indptr.shape[0] - 1

        # 1) compute per-request starts, lengths, page indices, cache starts/ends
        starts = qo_indptr[:-1]  # shape [n_req]
        ends = qo_indptr[1:]
        lengths = ends - starts  # shape [n_req]
        page_idxs = paged_kv_indices[paged_kv_indptr[:-1]]  # shape [n_req]
        cache_ends = paged_kv_last_page_len  # shape [n_req]
        cache_starts = cache_ends - lengths  # shape [n_req]

        # 2) now build the flat mappings
        total_tokens = int(lengths.sum().item())  # total across all requests

        # for each new token, which request (0…n_req-1) it belongs to
        segment_ids = torch.repeat_interleave(torch.arange(n_req, device=self.device), lengths)  # shape [total_tokens]

        # global token positions 0…total_tokens-1
        token_pos = torch.arange(total_tokens, device=self.device)

        # 3) fill the two output tensors in one shot
        if self.use_cuda_graph:
            self.token_to_page[:] = page_idxs[segment_ids]
            self.token_to_cache[:] = cache_starts[segment_ids] + (token_pos - starts[segment_ids])
        else:
            self.token_to_page = page_idxs[segment_ids]
            self.token_to_cache = cache_starts[segment_ids] + (token_pos - starts[segment_ids])

        return 
    
    def run(self, q, kv_cache):
        if not self.use_cuda_graph:
            if q.isnan().any() or kv_cache.isnan().any(): 
                print("[WARNING] NaN detected in input tensors!")
        return self.attn_wrapper.run(q, kv_cache)
    
    def set_kv_cache(self, kv_cache, k, v):
        """
        kv_cache : torch.Tensor, shape = (n_pages, 2, n_ctx, n_heads, head_dim)
            the KV cache for cross-attention
            n_cts is either 1500 or 448
        k, v   : torch.Tensor, shape = (n_token, n_heads, head_dim)
        """
        # these were created in `plan()`
        page_idx = self.token_to_page  # (total_tokens,)
        cache_idx = self.token_to_cache  # (total_tokens,)

        # two pure‐tensor assignments—no Python loop, no .item():
        kv_cache[page_idx, 0, cache_idx] = k  # keys
        kv_cache[page_idx, 1, cache_idx] = v  # values
        
        return 
    

class FlashInferDecodeWrapper():
    def __init__(
        self, 
        attn_buffer: torch.Tensor,
        n_qo_head: int, 
        n_kv_head: int,
        n_state: int,
        page_size: int,
        max_batch_size: int,
        device: torch.device = torch.device("cuda"),
        paged_kv_indptr_buffer: torch.Tensor = None,
        paged_kv_indices_buffer: torch.Tensor = None,
        paged_kv_last_page_len_buffer: torch.Tensor = None,
        use_cuda_graph: bool = False, 
        use_tensor_cores: bool = True,
    ):
        self.device = device
        # self.attn_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.max_batch_size = max_batch_size

        self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            attn_buffer, "NHD", 
            use_cuda_graph=use_cuda_graph, 
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        )

        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.n_state = n_state
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size

        self.kv_cache_locations = torch.zeros((max_batch_size, 2), dtype=torch.long, device=self.device)
    
    def plan(
        self, 
        # qo_indptr: torch.Tensor, # [n_req + 1] 
        paged_kv_indptr: torch.Tensor, # [n_req + 1]
        paged_kv_indices: torch.Tensor, # [# active pages]
        paged_kv_last_page_len: torch.Tensor, # [n_req]
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
        pos_idx = paged_kv_last_page_len - 1                # shape: (n_req,)

        # Store as a single tensor of shape (n_req, 2) [page_idx, pos_idx]
        self.kv_cache_locations[:self.batch_size].copy_(torch.stack([page_idx, pos_idx], dim=1))

        return 
    
    def run(self, q, kv_cache):
        return self.attn_wrapper.run(q, kv_cache)
    
    def set_kv_cache(self, kv_cache, k, v):
        """
        kv_cache : torch.Tensor, shape = (n_pages, 2, page_size, n_heads, head_dim)
            the KV cache for cross-attention
            n_cts is either 1500 or 448
        k, v   : torch.Tensor, shape = (n_req, n_heads, head_dim)
        """
        # Assuming self.kv_cache_locations is a tensor of shape (batch_size, 2)
        # with the first column being page indices and the second column being pos indices.
        pages = self.kv_cache_locations[:self.batch_size, 0].long()
        positions = self.kv_cache_locations[:self.batch_size, 1].long()

        # Vectorized assignment replaces the loop:
        kv_cache[pages, 0, positions] = k
        kv_cache[pages, 1, positions] = v
        
        return 


FlashInferWrapper = Union[FlashInferPrefillWrapper, FlashInferDecodeWrapper]