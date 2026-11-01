"""Kernel fusion for the models' programs, used by both runtimes.

The programs (``vox_serve/model/<name>/program.py``) switch these on from their config, which both the Rust
runtime and the Python runtime set from the same flags:

- ``--fuse-lm``: ``projections`` merges sibling q/k/v and gate/up projections into single GEMMs.
- ``--fuse-layers``: ``decoder_stack`` runs Llama / Qwen-style decoder layers with merged GEMMs and fused
  add + RMSNorm, q/k-norm + RoPE + KV-write and SiLU-and-mul kernels.
- ``--fuse-codec``: ``dac`` (Higgs Audio DAC decoder) and ``hift`` (CosyVoice3 HiFT vocoder, channels-last),
  plus the CosyVoice3 flow kernels of ``dit``.
- ``--fuse-depth`` / ``--fuse-talker`` (Qwen3-TTS) and ``--compile-detok`` live in the programs themselves.

``dit`` and ``seg_attention`` also hold kernels CosyVoice3's flow decoder always uses.
"""
