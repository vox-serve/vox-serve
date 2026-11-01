Adding a Model
==============

A model is implemented once, in Python, as a *program*, and both runtimes (:doc:`runtimes`) serve it.
Adding one touches no scheduler, worker or engine code:

.. list-table::
   :header-rows: 1
   :widths: 34 50 16

   * - File
     - Contents
     - Needed
   * - ``vox_serve/model/<name>/model.py``
     - PyTorch modules, weight loading, codec, model constants (a ``BaseLM`` subclass)
     - usually
   * - ``vox_serve/model/<name>/program.py``
     - the program: how a step runs on the GPU, what sampled tokens mean, prompts, ``meta``
     - yes
   * - ``vox_serve/model/__init__.py``
     - the model names that select the program
     - yes
   * - ``rust/src/models/<name>.rs``
     - a prompt builder in Rust, for high request rates on the Rust runtime
     - optional
   * - ``README.md``, ``docs/models.rst``, ``docs/usage/<name>.md``
     - documentation
     - yes

The shortest complete example is Orpheus (Llama-3.2-3B with the SNAC codec), whose whole program is shown
in :ref:`step 2 <adding-a-model:2. the program>`. The other programs cover multi-codebook models
(``zonos``, ``higgs_tts``), depth transformers (``csm``, ``qwen3_tts``), flow-matching codecs (``cosyvoice2``,
``chatterbox``), audio input (``step_audio_2``, ``glm_voice``), input streaming (``qwen3_tts``) and a codec that
decodes the LM's hidden states, carried to it in the token rows (``xtts``).

How a program runs
------------------

At start-up, the program allocates static buffers (KV cache, per-request sampling state) and declares
*arenas*: named input and output fields in one device buffer each. It then captures every stage of a step
as CUDA graphs, one per batch size:

- ``lm_prefill``: prompt rows, one graph per row bucket (``256``, ``512``, ``1024``, ``2048``, up to
  ``--max-context``);
- ``lm_decode``: one row per request, for batch sizes 1, 2, 4, ... up to ``--max-batch-size``;
- ``detok``: the codec, which turns a chunk of ``detok_interval`` frames into ``detok_samples`` audio samples.

Each step, a runtime writes the inputs (tokens, positions, KV page tables, sampling parameters, ...) into the
input arena, replays the graphs and reads the output arena: the sampled tokens (``out_tok``) and their
kind (``out_kind``: audio frame, stop, or other). The runtime never looks inside the model, and the program
never sees requests, only rows and batch entries. ``LMProgram`` (``vox_serve/rt/program.py``) implements all
of this; a program supplies the model-specific pieces. The conventions (rows fed back on the GPU, the
scratch KV page, padding entries) are described at the top of ``vox_serve/rt/program.py``.

Code that runs inside a captured graph, which is everything a program's hooks run except ``load``,
``preprocess`` and ``meta``, must follow the rules of CUDA graph capture:

- **No host synchronization.** No ``.item()``, ``.cpu()``, ``.tolist()``, printing tensors, or Python
  ``if`` on tensor values; use ``torch.where`` and masks.
- **Static shapes.** Shapes may depend on the batch size or row bucket being captured, not on data.
- **State in buffers.** Anything kept between steps or chunks lives in a buffer allocated in
  ``allocate_state`` and indexed by the request's slot (``arena["slot"]``), never in Python attributes.
- **Padding is real work.** Unused batch entries (``active == 0``) point at a dummy slot and KV page 0 and
  compute garbage that is ignored; they must not fail or write into other requests' state.
- **Randomness through the sampler.** ``sample_into`` draws from per-request seeds, so a request samples the
  same tokens whatever batch it lands in.

1. The model code
-----------------

``model.py`` holds the PyTorch side, often adapted from the reference implementation:

- **Attention** goes through FlashInfer's paged KV cache. Each layer receives ``attn_wrapper`` and the paged
  cache, writes its keys and values with ``attn_wrapper.set_kv_cache(kv_cache, k, v)`` and attends with
  ``attn_wrapper.run(q, kv_cache)``. The cache is laid out ``[num_layers, num_pages, 2, page_size,
  num_kv_heads, head_dim]``. ``vox_serve/flashinfer_utils.py`` has RoPE and RMSNorm kernels. Start from a
  model with the same backbone: Orpheus (Llama), CosyVoice2 (Qwen2), Qwen3-TTS (Qwen3).
- **The codec** maps token ids ``[batch, frames, codebooks]`` to audio. Shared codecs live in
  ``vox_serve/tokenizer/`` (SNAC, Mimi, DAC, HiFT, ...) and audio encoders in ``vox_serve/encoder/``.
- **Constants**: a ``BaseLM`` subclass (``vox_serve/model/base.py``) reports the layer and head counts,
  ``detokenize_interval`` (frames per audio chunk), ``detokenize_overlap`` (frames consecutive chunks
  share), ``output_audio_length`` (samples per chunk), ``max_tokens`` and ``default_sampling_config``
  (a ``vox_serve.sampling.SamplingConfig``).

The model class does not run steps, sample or build prompts; the program does.

2. The program
--------------

``program.py`` subclasses ``LMProgram``. The Orpheus program, complete:

.. literalinclude:: ../vox_serve/model/orpheus/program.py
   :language: python

Required pieces:

``load()``
  Builds the model on ``self.device`` (the codec on ``self.detok_device``, a second GPU with
  ``--enable-disaggregation``) and sets ``n_codebooks``, ``vocab_size``, ``num_layers``, ``num_qo_heads``,
  ``num_kv_heads``, ``head_dim``, ``detok_interval`` and ``detok_samples``. ``self.config`` holds the
  command-line options (``model``, ``max_batch_size``, ``fuse_lm``, ...); map the short model name to a
  Hugging Face id here.

``embed(ids, arena, rows)``, ``backbone(hidden, pos, attn)``, ``head(hidden)``
  Token ids ``[rows, n_codebooks]`` to hidden states, the decoder stack over paged attention, and hidden
  states ``[N, H]`` to logits ``[N, n_codebooks, vocab]``. ``self.kv_cache`` is the paged cache.

``detokenize(tokens)``
  ``[Bd, detok_interval, n_codebooks]`` token ids to float audio in ``[-1, 1]`` with ``Bd * detok_samples``
  values. Consecutive chunks start ``detok_interval - detok_overlap`` frames apart, and each returns only the
  samples it adds. ``detok_step`` converts the audio to int16 PCM with ``pcm16`` (``rt/program.py``), which
  clamps samples outside ``[-1, 1]``; programs that override ``detok_step`` convert with it too.

``stop_tokens()``, ``audio_tokens()``
  Which codebook-0 tokens end generation, and the ``[lo, hi)`` range that is audio (other tokens, such as
  interleaved text, are fed back but not decoded). Override ``token_kind(tok)`` for anything else.

``meta()``
  Model constants for the runtimes (table below).

Prompts
  ``preprocess``, or the ``template`` prompt builder (see :ref:`step 4 <adding-a-model:4. prompts>`).

Optional pieces (look at the programs named for examples):

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Override / attribute
     - Use
   * - ``decode_tokens(hidden, arena, out_tok, n_req)``
     - Custom sampling: depth transformers (``csm``), delay patterns (``zonos``), classifier-free guidance
       (``voxtral_tts``). Sample with ``self.sample_into``, with a different ``salt`` per sampling call.
   * - ``allocate_state()``
     - Extra per-slot buffers (call ``super().allocate_state()`` first).
   * - ``uses_input_mask = True`` + ``meta["decode_mask"]``
     - A per-column input mask next to the tokens (e.g. which columns of a row are text or audio).
   * - ``fb_columns`` + ``meta["decode_row"]``
     - Only the first ``fb_columns`` columns of decode rows are fed back from the last sample; the
       others come from ``decode_row``.
   * - ``uses_repetition_penalty = False``
     - The model has no repetition penalty (or applies its own in ``decode_tokens``).
   * - ``detok_stateful = True``
     - The codec keeps per-request state across chunks; ``arena["dt_slot"]`` / ``arena["dt_reset"]`` index
       and reset it.
   * - ``detok_step`` + ``dt_len`` / ``dt_last`` detok inputs (``declare``)
     - Chunks with look-ahead (``cosyvoice3``, ``higgs_tts``): ``arena["dt_len"]`` is the number of valid
       frames in the window and ``arena["dt_last"]`` is 1 on a request's last chunk, the one no other chunk
       follows. A chunk may then hold audio back for the next one: the runtimes finish a request only after
       its last chunk has been decoded.
   * - ``feature_pool_rows``, ``feature_dim``
     - Audio input: encoded audio rows that prompts reference through the input mask (``step_audio_2``).
   * - ``tokenize_stream_text``, ``preprocess_streaming`` + ``meta["text_stream"]``
     - Input streaming: text arriving while audio is generated (``qwen3_tts``).
   * - ``supports_urgent_lane = True``
     - The codec can run on a second lane at the same time (no shared scratch state between calls).
   * - ``decode_head`` / ``decode_depth`` (``supports_depth_split``), ``backbone_range``
       (``supports_layer_split``)
     - The LM step can be captured in segments (``--lm-segments``).
   * - ``fuse_lm_modules()``, ``fused_stack_spec()``, ``compile_detok_targets()``
     - Kernel fusion with ``--fuse-lm``, ``--fuse-layers`` and ``--compile-detok`` (``vox_serve/fusion/``).

``meta()`` keys (``LMProgram`` adds the common ones: batch sizes, page size, codebooks, ...):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``kind``
     - The model's identifier; also selects the Rust prompt builder unless ``prompt_builder`` is set.
   * - ``prompt_builder``
     - ``python`` (the program's ``preprocess``), ``template``, or a Rust builder's name.
   * - ``max_tokens``
     - Per-request token limit, prompt included (required).
   * - ``sampling``
     - Default ``temperature``, ``top_k``, ``top_p``, ``min_p``, ``repetition_penalty``; the ``--temperature``
       etc. flags override them.
   * - ``sample_rate``
     - Output sample rate in Hz (default 24000).
   * - ``detok_overlap``
     - Frames consecutive audio chunks share (default 0).
   * - ``preseed_frames``
     - Zero frames placed before the first audio frame, so the first chunk is decoded earlier; their
       samples are trimmed.
   * - ``decode_mask``, ``decode_row``, ``text_stream``
     - See the optional pieces above.
   * - ``chunk_samples``, ``audio_len_in_row``
     - Codecs whose chunks vary in length: samples a chunk advances, and the valid length stored in the last
       sample of each audio row.
   * - anything else
     - Inputs of the prompt builders (tokenizer paths, special token ids, voices, ...).

3. Registering the model
------------------------

Add the names that select the program to ``_PROGRAMS`` in ``vox_serve/model/__init__.py``, lower case, as
``(package, class)``. ``--model`` is matched exactly first, then as a substring, so list the short name and the
Hugging Face organization prefix:

.. code-block:: python

   "my-tts": ("my_tts", "MyTTSProgram"),
   "my-org/my-tts": ("my_tts", "MyTTSProgram"),

4. Prompts
----------

``preprocess(text, audio_path, model_kwargs)`` returns a ``Prompt`` (``vox_serve/rt/prompt.py``): the
prompt's rows flattened to ``[rows * n_codebooks]`` token ids, and with ``uses_input_mask`` a mask of the
same length. ``model_kwargs`` holds the request's model-specific options (``voice``, ``speaker``,
``language``, ``ref_text``, ``instruct``, ``x_vector_only_mode``); a new option has to be added to both
servers' ``/generate`` form handling (``vox_serve/launch.py`` and ``rust/src/server.rs``). Raise
``ValueError`` for unsupported input, such as audio for a model without audio input: the request then fails
with that message.

The Python runtime always calls ``preprocess``. The Rust runtime uses the builder named in ``meta``:

``python``
  Calls ``preprocess`` for every request, holding the Python interpreter lock while it runs. Start here: it
  needs no Rust, and it is fast enough unless requests arrive at high rates. It does not support prompts
  that use the feature pool (audio input) or input streaming.

``template``
  For prompts that are fixed token ids around a tokenized text template, configured in
  ``meta["prompt_template"]`` (Orpheus above). Runs in Rust; no ``preprocess`` needed.

A Rust builder
  ``rust/src/models/<name>.rs`` implements the ``PromptBuilder`` trait (``preprocess``, and
  ``tokenize_stream_text`` / ``preprocess_streaming`` for input streaming), reads its constants from
  ``meta`` in its constructor, and is registered under the model's ``kind`` in ``load`` in
  ``rust/src/models/mod.rs``. Steps that need PyTorch (encoding reference audio, phonemization) can stay
  in Python as hooks: ``Hooks::call("name", args)`` runs the program's ``hook_<name>`` (see ``zonos.rs``).
  The builder must produce exactly the rows ``preprocess`` does. ``tests/test_prompt_parity.py`` checks
  that for text-only requests on CPU, from the model's recorded ``meta``: record it once with
  ``python tests/prompt_meta/record.py <name>`` (on a GPU) and commit ``tests/prompt_meta/<name>.json``.
  Text-only prompts should therefore read only ``self.spec_meta`` (``LMProgram.for_prompts`` builds a
  program without weights for the test). For other requests, compare with
  ``vox_serve._rt.build_prompt(meta_json, text, audio_path, model_kwargs_json)``, or serve with
  ``--prompt-builder python`` and compare greedy outputs.

5. Validating
-------------

The tests in ``tests/`` run on CPU (:ref:`development:tests`) and check the schedulers and prompt parity,
not models: there is no GPU test suite, so validate a model by serving it.

.. code-block:: bash

   # Debug model code eagerly (Python exceptions, prints), then with CUDA graphs, in the Python runtime.
   vox-serve --runtime python --model my-tts --disable-cuda-graph --port 8000
   vox-serve --runtime python --model my-tts --port 8000
   curl -X POST localhost:8000/generate -F "text=Hello world" -F "streaming=true" -o out.wav

   # The Rust runtime: concurrent requests without HTTP, greedy, audio written to a directory.
   vox-serve --runtime rust --model my-tts --bench-requests 4 --bench-out /tmp/my-tts --greedy
   python rust/scripts/asr_check.py /tmp/my-tts/bench_*.wav   # Whisper transcripts of the outputs

Check that:

- the audio is intelligible and complete (the ASR transcript matches the text, the end is not cut off);
- streaming and non-streaming requests both work, in both runtimes;
- several concurrent requests give the same quality as one (bugs in per-slot state or padding entries only
  show with batching) and greedy outputs do not depend on the batch;
- with a Rust prompt builder, its prompts equal ``preprocess``'s (``python -m pytest tests/test_prompt_parity.py``
  with the extension built, after recording the model's ``meta``).

Then measure throughput and latency under load with ``benchmark/``.

6. Documentation
----------------

Add the model to the table in ``README.md`` and :doc:`models`, and add a usage page (``docs/usage/<name>.md``,
listed in ``docs/usage/index.rst``) with its request options.
