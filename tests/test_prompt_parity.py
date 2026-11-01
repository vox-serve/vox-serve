"""Each model's Rust prompt builder produces exactly the rows of its program's ``preprocess``.

Both builders read only the engine ``meta`` (recorded per model in ``tests/prompt_meta``, see
``record.py`` there) and its tokenizer file. The test substitutes a small byte-level BPE tokenizer for
the model's, so it runs on CPU without downloading anything: what it checks is everything around the
tokenizer (templates, special ids, speakers / voices / languages, row layout, masks, text normalization,
input-streaming prompts). Text-only requests: audio inputs need the model's encoders.

Needs the Rust extension (``vox_serve._rt``); skipped without it unless ``VOX_REQUIRE_RT=1``.
"""

import glob
import importlib
import json
import os
import re

import pytest

if os.environ.get("VOX_REQUIRE_RT") == "1":
    from vox_serve import _rt
else:
    _rt = pytest.importorskip("vox_serve._rt", reason="the Rust extension is not built")

META_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_meta")
MODELS = sorted(os.path.basename(p)[: -len(".json")] for p in glob.glob(os.path.join(META_DIR, "*.json")))

T = "Hey there, this is a short benchmark sentence for the vox serve runtime written in rust."
TEXTS = [
    T,
    "I have 3 apples and 1999 pears; ok... “quoted” — x",
    "Use {voice} and {text} braces.",
    "  spaced 42  ",
    "hello   world",
    "wait... what; ok",
    "Done!",
    "<|endofprompt|>keep 12",
    "élan — “quoted” : x",
    "你好，世界。今天是2024年。",
    "",
]
CORPUS = TEXTS + [
    "the quick brown fox jumps over the lazy dog " * 4,
    "system human assistant user you are a helpful assistant speak happily some transcript",
    "0123456789 one two three hundred thousand point percent",
] + ["<|im_start|>assistant\n<|im_start|>user\n<|im_start|>system\nhuman\n"] * 50  # whole-word role names
# Tokenizers of these model families add no BOS token (the Qwen3-TTS builders check the role header shape).
NO_BOS = {"qwen3_tts"}


def _kwargs_cases(meta):
    """Model-specific request options, as (model_kwargs, first streamed text) pairs."""
    kind = meta.get("prompt_builder") or meta.get("kind")
    cases = []
    if kind == "template":
        choices = (meta["prompt_template"].get("kwargs") or {}).get("voice", {}).get("choices") or []
        cases += [{"voice": v} for v in choices[:2]] + [{"voice": "nobody"}, {"voice": 3}]
    elif kind == "csm":
        cases += [{"speaker": "1"}, {"speaker": 2}, {"speaker": True}]
    elif kind == "qwen3_tts" and meta.get("tts_model_type") == "custom_voice":
        speakers = sorted(meta["speakers"])
        cases += [
            {"speaker": speakers[1]},
            {"speaker": speakers[-1].upper()},
            {"speaker": "NoSuchSpeaker"},
            {"language": "chinese"},
            {"language": "auto"},
            {"language": "klingon"},
            {"instruct": "Speak happily."},
            {"speaker": speakers[-1], "language": "auto"},
        ]
    elif kind == "voxtral_tts":
        voices = sorted(meta["voices"])
        cases += [{"voice": voices[-1]}, {"voice": [voices[0]]}, {"voice": "nobody"}]
    return [(k, None) for k in cases]


def _tokenizer(path, meta):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    specials = ["<|bos|>", "<|endofprompt|>", "<|im_start|>", "<|im_end|>", "[SPACE]"]
    specials += sorted(set(re.findall(r"<\|[A-Za-z_]+\|>", json.dumps(meta))) - set(specials))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400 + len(specials),
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS + [json.dumps(meta)[:20000]], trainer)
    if meta.get("kind") not in NO_BOS:
        # A BOS token with add_special_tokens, so builders that disagree on that flag differ.
        tok.post_processor = processors.TemplateProcessing(
            single="<|bos|> $A", pair="<|bos|> $A $B", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>"))]
        )
    tok.save(path)


def _substitute(value, path):
    if isinstance(value, dict):
        return {k: _substitute(v, path) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, path) for v in value]
    return path if value == "{tokenizer}" else value


@pytest.fixture(scope="module", params=MODELS)
def model(request, tmp_path_factory):
    with open(os.path.join(META_DIR, f"{request.param}.json")) as f:
        rec = json.load(f)
    path = str(tmp_path_factory.mktemp(request.param) / "tokenizer.json")
    _tokenizer(path, rec["meta"])
    meta = _substitute(rec["meta"], path)
    module, name = rec["program"].split(":")
    program = getattr(importlib.import_module(module), name).for_prompts(meta)
    return request.param, meta, program


def _python(program, text, kwargs, first):
    try:
        if first is None:
            p = program.preprocess(text, None, kwargs)
        else:
            p = program.preprocess_streaming(text, None, kwargs, first)
        return (list(p.tokens), None if p.mask is None else list(p.mask)), None
    except Exception as e:  # noqa: BLE001  both builders must reject the same requests
        return None, f"{type(e).__name__}: {e}"


def _rust(meta, text, kwargs, first):
    try:
        tokens, mask = _rt.build_prompt(json.dumps(meta), text, None, json.dumps(kwargs), first)
        return (list(tokens), None if mask is None else list(mask)), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def test_prompt_parity(model):
    name, meta, program = model
    cases = [({}, None, t) for t in TEXTS] + [(k, f, T) for k, f in _kwargs_cases(meta)]
    if meta.get("text_stream"):  # input streaming: the prompt holds the first streamed text token
        first = program.tokenize_stream_text("Hello")[0]
        cases += [({}, first, T), ({}, first, "")] + [(k, first, T) for k, _ in _kwargs_cases(meta)[:2]]
    failures = []
    for kwargs, first, text in cases:
        py, py_err = _python(program, text, kwargs, first)
        rs, rs_err = _rust(meta, text, kwargs, first)
        where = f"{name} text={text[:24]!r} kwargs={kwargs} first={first}"
        if (py_err is None) != (rs_err is None):
            failures.append(f"{where}: python {py_err or 'ok'} / rust {rs_err or 'ok'}")
        elif py != rs:
            failures.append(f"{where}: rows differ\n  python {py}\n  rust   {rs}")
    assert not failures, "\n".join(failures)
    # The fixture is not trivially empty: most requests build a prompt.
    assert sum(_python(program, t, {}, None)[0] is not None for t in TEXTS) >= len(TEXTS) // 2
