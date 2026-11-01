"""Records the engine ``meta`` of models for ``tests/test_prompt_parity.py``. Needs a GPU and the weights.

    python tests/prompt_meta/record.py orpheus csm chatterbox cosyvoice2 glm step qwen3-tts voxtral-tts higgs-tts

Writes ``tests/prompt_meta/<model>.json`` with the program class and its ``meta`` exactly as the engine
builds it, except for local file paths: tokenizer files become ``"{tokenizer}"`` (the test substitutes a
small tokenizer of its own, so both prompt builders read the same file) and any other path is dropped.
Re-record after changing what a program puts in ``meta``. Each model is built in its own process.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _scrub(value, dropped, key=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            v = _scrub(v, dropped, f"{key}.{k}" if key else k)
            if v is not _DROP:
                out[k] = v
        return out
    if isinstance(value, list):
        return [_scrub(v, dropped, key) for v in value]
    if isinstance(value, str) and os.path.isabs(value) and os.path.exists(value):
        if os.path.basename(value).endswith("tokenizer.json"):
            return "{tokenizer}"
        dropped.append(key)
        return _DROP
    return value


_DROP = object()


def record(model: str) -> None:
    from vox_serve.rt import core

    core.build_engine(json.dumps({"model": model, "max_batch_size": 2, "max_context": 2048}))
    program, builder = core._ENGINE_KEEPALIVE[-1]
    dropped = []
    meta = _scrub(builder.meta, dropped)
    cls = type(program)
    out = {"program": f"{cls.__module__}:{cls.__name__}", "meta": meta}
    path = os.path.join(HERE, f"{model}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=None, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(f"{model}: {path} ({os.path.getsize(path)} bytes); dropped paths: {dropped or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        record(sys.argv[2])
    else:
        failed = [m for m in sys.argv[1:] if subprocess.call([sys.executable, __file__, "--one", m]) != 0]
        sys.exit(f"failed: {failed}" if failed else 0)
