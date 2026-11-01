# Vendored FlashInfer headers

The headers `csrc/fi_plan.cu` needs to compile FlashInfer's host-side prefill planner
(`flashinfer/attention/scheduler.cuh` and what it includes), copied unchanged from
`flashinfer-python` (version in `version.txt`; Apache-2.0, see `LICENSE` and `NOTICE`).

The Rust runtime plans attention with this code and the captured graphs run the kernels of the
installed `flashinfer-python`, so the two must be the same version: `pyproject.toml` pins it exactly
and `vox_serve.rt.build_engine` refuses to start on a mismatch. After changing the pin, run
`update.sh` with `VOX_PY` pointing at a Python that has the new version installed.

`build.rs` uses these headers unless `FLASHINFER_INCLUDE` names another include directory.
