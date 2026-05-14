"""FastAPI server that hosts Baseten's ``orpheus-best-performance`` Model class
locally on our own GPU. The Baseten inference handler is imported VERBATIM
from the cloned truss-examples repo; only the engine handle and the HTTP
shell are ours.

Layout assumption (set via env or CLI flags):

  ${TRUSS_EXAMPLES_REPO}/orpheus-best-performance/model/model.py
  ${ENGINE_DIR}/...               # trtllm-build output
  ${ORPHEUS_HF_DIR}/...           # baseten/orpheus-3b-0.1-ft @ pinned revision
  ${SNAC_DIR}/...                 # hubertsiuzdak/snac_24khz
  ${DATA_DIR}/tokenization/...    # symlinks to tokenizer files (from setup script)

Run:

  python server.py \\
    --engine-dir   "$ENGINE_DIR" \\
    --tokenizer    "$ORPHEUS_HF_DIR" \\
    --data-dir     "$DATA_DIR" \\
    --snac-dir     "$SNAC_DIR" \\
    --truss-repo   "$TRUSS_EXAMPLES_REPO" \\
    --host 0.0.0.0 --port 8000

The benchmark client (goodput_baseten.py) talks to POST /predict with the
same JSON schema Baseten's managed endpoint expects:

  {"prompt": "...", "voice": "tara", "max_tokens": 4096,
   "stop_token_ids": [128258, 128009], "request_id": "<uuid>"}

Response is a streaming raw-PCM audio body (24 kHz mono int16), identical to
what Baseten's hosted endpoint returns.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import fastapi
import uvicorn

from engine_adapter import LocalTRTLLMEngine


def _patch_snac_loader(snac_dir: str) -> None:
    """Redirect Baseten's hardcoded ``/app/snac_24khz`` to our local SNAC dir.

    Baseten's ``model/model.py`` calls ``SNAC.from_pretrained("/app/snac_24khz")``,
    which only exists inside their Docker image. We can't add a /app symlink
    without root on a shared cluster, so we monkey-patch ``SNAC.from_pretrained``
    to swap that one specific path before importing their module.

    This is the ONLY behavioral change we make to Baseten's stack — and it's a
    path redirect, not a model-logic change.
    """
    from snac import SNAC

    _orig = SNAC.from_pretrained.__func__  # classmethod -> underlying func

    def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "/app/snac_24khz":
            pretrained_model_name_or_path = snac_dir
        return _orig(cls, pretrained_model_name_or_path, *args, **kwargs)

    SNAC.from_pretrained = classmethod(_patched)
    print(f"[server] SNAC.from_pretrained patched: /app/snac_24khz -> {snac_dir}")


def _import_baseten_model_module(truss_repo: str):
    """Import truss-examples/orpheus-best-performance/model/model.py verbatim."""
    model_py = Path(truss_repo) / "orpheus-best-performance" / "model" / "model.py"
    if not model_py.is_file():
        raise FileNotFoundError(f"Could not find Baseten Model handler at {model_py}")

    model_dir = str(model_py.parent)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location("baseten_orpheus_model", str(model_py))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_app(args) -> fastapi.FastAPI:
    _patch_snac_loader(args.snac_dir)

    print(f"[server] loading TRT-LLM engine from {args.engine_dir}")
    engine = LocalTRTLLMEngine(
        engine_dir=args.engine_dir,
        tokenizer_dir=args.tokenizer,
        kv_cache_free_gpu_mem_fraction=args.kv_cache_fraction,
        enable_chunked_context=True,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
    )

    print(f"[server] importing Baseten Model handler from {args.truss_repo}")
    baseten_module = _import_baseten_model_module(args.truss_repo)

    # Baseten's Model expects keyword args: trt_llm, secrets, data_dir.
    model = baseten_module.Model(
        trt_llm={"engine": engine},
        secrets={"hf_access_token": os.environ.get("HF_TOKEN")},
        data_dir=args.data_dir,
    )
    model.load()
    print("[server] Baseten Model handler ready")

    app = fastapi.FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/predict")
    async def predict(payload: dict, request: fastapi.Request):
        return await model.predict(payload, request)

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine-dir", required=True,
                   help="trtllm-build output dir (see 01_build_engine.sh).")
    p.add_argument("--tokenizer", required=True,
                   help="Path to Orpheus HF dir (used as TRT-LLM tokenizer).")
    p.add_argument("--data-dir", required=True,
                   help="Dir containing tokenization/ subfolder (see setup script).")
    p.add_argument("--snac-dir", required=True,
                   help="Local SNAC weights dir; will be symlinked to /app/snac_24khz.")
    p.add_argument("--truss-repo", required=True,
                   help="Cloned basetenlabs/truss-examples repo root.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--kv-cache-fraction", type=float, default=0.90,
                   help="Matches Baseten's kv_cache_free_gpu_mem_fraction.")
    p.add_argument("--max-batch-size", type=int, default=256)
    p.add_argument("--max-num-tokens", type=int, default=16384)
    args = p.parse_args()

    # Single worker so SnacModelBatched (module-level singleton) batches across
    # all concurrent requests in one process. Don't crank workers > 1.
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
