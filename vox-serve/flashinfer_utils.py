import torch 
import flashinfer
from typing import Union

class FlashInferPrefillWrapper():
    def __init__(
        self, 
        attn_buffer: torch.Tensor,
        n_qo_head=24, 
        n_kv_head=8,
        n_state=3072, 
        page_size=16,
        seq_len=16,
        device=torch.device("cuda"),
        qo_indptr_buf=None,
        paged_kv_indptr_buf=None,
        paged_kv_indices_buf=None,
        paged_kv_last_page_len_buf=None,
    ):
        self.device = device
        # self.attn_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        # self.seq_len = seq_len # number of tokens, not number of batches

        self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            attn_buffer, "NHD",
            # use_cuda_graph=True,
            # qo_indptr_buf=qo_indptr_buf,
            # paged_kv_indptr_buf=paged_kv_indptr_buf,
            # paged_kv_indices_buf=paged_kv_indices_buf,
            # paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
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
        self.kv_cache_locations = torch.zeros((n_req, 5), dtype=torch.long, device=self.device)
        
        for i in range(n_req):
            n_new_token = qo_indptr[i + 1] - qo_indptr[i]
            page_idx = paged_kv_indices[paged_kv_indptr[i]]
            last_len = paged_kv_last_page_len[i]
            
            # Store all location parameters in a single tensor row
            self.kv_cache_locations[i, 0] = qo_indptr[i]                # start_idx
            self.kv_cache_locations[i, 1] = qo_indptr[i + 1]            # end_idx
            self.kv_cache_locations[i, 2] = page_idx                    # page_idx
            self.kv_cache_locations[i, 3] = last_len - n_new_token      # cache_start
            self.kv_cache_locations[i, 4] = last_len                    # cache_end

        return 
    
    def run(self, q, kv_cache):
        if q.isnan().any() or kv_cache.isnan().any(): 
            # somehow we need this for robust initialization
            # this is okay since prefill is not cuda-graph optimized
            print("[WARNING] NaN detected in input tensors!")
        return self.attn_wrapper.run(q, kv_cache)
    
    def set_kv_cache(self, kv_cache, k, v):
        """
        kv_cache : torch.Tensor, shape = (n_pages, 2, n_ctx, n_heads, head_dim)
            the KV cache for cross-attention
            n_cts is either 1500 or 448
        k, v   : torch.Tensor, shape = (n_token, n_heads, head_dim)
        """
        # Use precomputed tensor locations to update KV cache
        for i in range(len(self.kv_cache_locations)):
            # Extract location data from tensor
            start_idx = self.kv_cache_locations[i, 0].item()
            end_idx = self.kv_cache_locations[i, 1].item()
            page_idx = self.kv_cache_locations[i, 2].item()
            cache_start = self.kv_cache_locations[i, 3].item()
            cache_end = self.kv_cache_locations[i, 4].item()
            
            # Extract slices based on precomputed indices
            k_ = k[start_idx:end_idx]  # (n_new_token, n_heads, head_dim)
            v_ = v[start_idx:end_idx]  # (n_new_token, n_heads, head_dim)
            
            # Copy data to the precomputed locations in cache
            kv_cache[page_idx, 0, cache_start:cache_end].copy_(k_)
            kv_cache[page_idx, 1, cache_start:cache_end].copy_(v_)
        
        return 
    

class FlashInferDecodeWrapper():
    def __init__(
        self, 
        attn_buffer: torch.Tensor,
        n_qo_head=24, 
        n_kv_head=8,
        n_state=3072, 
        page_size=16,
        batch_size=1,
        device=torch.device("cuda"),
        paged_kv_indptr_buffer=None,
        paged_kv_indices_buffer=None,
        paged_kv_last_page_len_buffer=None,
    ):
        self.device = device
        # self.attn_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.batch_size = batch_size

        self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            attn_buffer, "NHD", 
            use_cuda_graph=True, 
            use_tensor_cores=True,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        )

        self.n_qo_head = n_qo_head
        self.n_kv_head = n_kv_head
        self.n_state = n_state
        self.head_dim = n_state // n_qo_head
        self.page_size = page_size

        self.kv_cache_locations = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)
    
    def plan(
        self, 
        # qo_indptr: torch.Tensor, # [n_req + 1] 
        paged_kv_indptr: torch.Tensor, # [n_req + 1]
        paged_kv_indices: torch.Tensor, # [# active pages]
        paged_kv_last_page_len: torch.Tensor, # [n_req]
        dtype: torch.dtype = torch.float16,
    ):
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
        self.kv_cache_locations.copy_(torch.stack([page_idx, pos_idx], dim=1))

        # print(f"{self.kv_cache_locations=}")

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
        pages = self.kv_cache_locations[:, 0].long()
        positions = self.kv_cache_locations[:, 1].long()

        # Vectorized assignment replaces the loop:
        kv_cache[pages, 0, positions] = k
        kv_cache[pages, 1, positions] = v
        
        return 


FlashInferWrapper = Union[FlashInferPrefillWrapper, FlashInferDecodeWrapper]