"""TRT-LLM executor adapter exposing the interface that Baseten's
``orpheus-best-performance/model/model.py`` expects from ``self._engine``.

Baseten's ``Model.predict`` calls::

    token_gen = await self._engine.predict(model_input, request)
    if isinstance(token_gen, StreamingResponse):
        token_gen = token_gen.body_iterator
    async for token_sim in token_gen:
        for tok_str in split_custom_tokens(token_sim):  # regex over the string
            ...

So ``engine.predict`` must return either a ``StreamingResponse`` whose
``body_iterator`` yields strings, or an async iterator yielding strings.
Each yielded string is regex-scanned for ``<custom_token_NNNN>`` patterns;
incremental yields are fine (regex still extracts every token).

We back this with TRT-LLM's high-level ``LLM`` API + ``generate_async`` in
streaming mode. ``generate_async`` yields cumulative ``output.outputs[0].text``;
we compute the delta and yield it.
"""

import asyncio
from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse


class LocalTRTLLMEngine:
    """Local replacement for Baseten's Briton-injected ``trt_llm['engine']``."""

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        kv_cache_free_gpu_mem_fraction: float = 0.90,
        enable_chunked_context: bool = True,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
    ):
        # Import lazily so module-level import doesn't trigger CUDA init on
        # machines that just want to lint/parse this file.
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        kv_cfg = KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_free_gpu_mem_fraction,
        )
        # NB: scheduler_policy and chunked-context flags are runtime options.
        # Their exact import paths shift across TRT-LLM versions; we let the
        # build-time flag (use_paged_context_fmha=enable) carry the chunked
        # support and only set free-GPU memory at runtime. If your TRT-LLM
        # version exposes more knobs, adjust here.
        self.llm = LLM(
            model=engine_dir,
            tokenizer=tokenizer_dir,
            kv_cache_config=kv_cfg,
        )

    async def predict(self, model_input: dict, request: Any) -> StreamingResponse:
        from tensorrt_llm import SamplingParams

        prompt = model_input["prompt"]
        sp = SamplingParams(
            max_tokens=int(model_input.get("max_tokens", 4096)),
            temperature=float(model_input.get("temperature", 0.6)),
            top_p=float(model_input.get("top_p", 0.8)),
            repetition_penalty=float(model_input.get("repetition_penalty", 1.1)),
            end_id=int(model_input.get("end_id", 128258)),
            stop_token_ids=list(model_input.get("stop_token_ids", []) or []),
        )

        llm = self.llm

        async def token_iter() -> AsyncIterator[bytes]:
            # generate_async with streaming=True returns an async iterator
            # whose elements carry cumulative text. We compute deltas.
            result_iter = llm.generate_async(
                prompt,
                sampling_params=sp,
                streaming=True,
            )
            prev_len = 0
            async for partial in _aiter(result_iter):
                # The output object is RequestOutput-like; access .outputs[0].text.
                text = ""
                try:
                    text = partial.outputs[0].text
                except (AttributeError, IndexError):
                    text = getattr(partial, "text", "")
                if len(text) > prev_len:
                    delta = text[prev_len:]
                    prev_len = len(text)
                    yield delta.encode("utf-8") if isinstance(delta, str) else delta

        return StreamingResponse(token_iter(), media_type="text/plain")


async def _aiter(it):
    """Adapt sync or async iterator into an async iterator uniformly.

    TRT-LLM's generate_async return value has been either an asynchronous
    iterator or a sync iterable depending on version. Handle both.
    """
    if hasattr(it, "__aiter__"):
        async for x in it:
            yield x
        return
    if hasattr(it, "__await__"):
        it = await it
    if hasattr(it, "__aiter__"):
        async for x in it:
            yield x
        return
    # Fallback: sync iterable — run iteration on a thread to avoid blocking
    # the event loop.
    loop = asyncio.get_event_loop()
    sync_iter = iter(it)
    sentinel = object()
    while True:
        nxt = await loop.run_in_executor(None, next, sync_iter, sentinel)
        if nxt is sentinel:
            return
        yield nxt
