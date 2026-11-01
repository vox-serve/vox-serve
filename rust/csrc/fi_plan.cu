// C ABI around FlashInfer's host-side prefill scheduler, so the Rust runtime can re-plan the
// attention kernels captured in CUDA graphs without going through Python.
#include <cstring>
#include <exception>

#include <flashinfer/attention/scheduler.cuh>

extern "C" int vox_fi_prefill_plan(void* float_buffer, size_t float_size, void* int_buffer, void* pinned_int_buffer,
                                   size_t int_size, const int32_t* qo_indptr_h, const int32_t* kv_indptr_h,
                                   uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads,
                                   uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size,
                                   int enable_cuda_graph, cudaStream_t stream, int64_t* plan_info_out, char* err,
                                   size_t err_len) {
  try {
    flashinfer::PrefillPlanInfo info;
    cudaError_t st = flashinfer::PrefillPlan<int32_t>(
        float_buffer, float_size, int_buffer, pinned_int_buffer, int_size, info, const_cast<int32_t*>(qo_indptr_h),
        const_cast<int32_t*>(kv_indptr_h), total_num_rows, batch_size, num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, enable_cuda_graph != 0, /*sizeof_dtype_o=*/2, stream);
    if (st != cudaSuccess) {
      std::strncpy(err, cudaGetErrorString(st), err_len - 1);
      err[err_len - 1] = 0;
      return 1;
    }
    std::vector<int64_t> v = info.ToVector();
    for (size_t i = 0; i < v.size(); ++i) plan_info_out[i] = v[i];
    return 0;
  } catch (const std::exception& e) {
    std::strncpy(err, e.what(), err_len - 1);
    err[err_len - 1] = 0;
    return 2;
  }
}
