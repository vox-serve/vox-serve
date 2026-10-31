//! Host-side FlashInfer planning (FlashInfer's own `PrefillPlan`, linked in via `csrc/fi_plan.cu`).

use std::ffi::{c_char, c_int, c_void, CStr};

use anyhow::{bail, Result};

use crate::cuda::{Pinned, Stream};
use crate::spec::AttentionSpec;

extern "C" {
    fn vox_fi_prefill_plan(
        float_buffer: *mut c_void,
        float_size: usize,
        int_buffer: *mut c_void,
        pinned_int_buffer: *mut c_void,
        int_size: usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        total_num_rows: u32,
        batch_size: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        enable_cuda_graph: c_int,
        stream: Stream,
        plan_info_out: *mut i64,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
}

/// Re-plans one captured attention wrapper. In CUDA-graph mode FlashInfer keeps the plan layout
/// fixed (padded batch, split-kv, workspace offsets) for a given (total rows, batch size), so the
/// captured kernels stay valid; we still verify this on every plan since a mismatch would be silent
/// memory corruption.
pub struct Planner {
    spec: AttentionSpec,
    pinned: Pinned,
}

impl Planner {
    pub fn new(spec: AttentionSpec) -> Result<Self> {
        let pinned = Pinned::new(spec.int_ws_size)?;
        Ok(Self { spec, pinned })
    }

    pub fn batch_size(&self) -> usize {
        self.spec.batch_size
    }

    /// `qo_indptr` / `kv_indptr` (in pages) cover exactly `batch_size` requests. The device copy
    /// of the planning metadata is enqueued on `stream`; the pinned staging buffer is owned by this
    /// planner and must not be re-planned before that copy completes.
    pub fn plan(&mut self, qo_indptr: &[i32], kv_indptr: &[i32], stream: Stream) -> Result<()> {
        let s = &self.spec;
        debug_assert_eq!(qo_indptr.len(), s.batch_size + 1);
        debug_assert_eq!(kv_indptr.len(), s.batch_size + 1);
        let mut info = [0i64; 15];
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            vox_fi_prefill_plan(
                s.float_ws_ptr as *mut c_void,
                s.float_ws_size,
                s.int_ws_ptr as *mut c_void,
                self.pinned.as_mut_ptr() as *mut c_void,
                s.int_ws_size,
                qo_indptr.as_ptr(),
                kv_indptr.as_ptr(),
                s.total_num_rows as u32,
                s.batch_size as u32,
                s.num_qo_heads,
                s.num_kv_heads,
                s.head_dim,
                s.page_size,
                1,
                stream,
                info.as_mut_ptr(),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy();
            bail!("FlashInfer plan failed: {msg}");
        }
        if info[..] != s.plan_info[..] {
            bail!("FlashInfer plan layout changed: captured {:?}, got {:?}", s.plan_info, info);
        }
        Ok(())
    }
}
