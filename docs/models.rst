Supported Models
================

VoxServe supports multiple TTS and STS model families. The tables below summarize the
current families, their shorthand codes, and a representative Hugging Face identifier.

Text-to-speech (TTS)
--------------------

.. list-table:: TTS model families
   :header-rows: 1
   :widths: 12 30 32

   * - Family code
     - Example Hugging Face model ID
     - Description
   * - ``chatterbox``
     - ``ResembleAI/chatterbox`` (`link <https://huggingface.co/ResembleAI/chatterbox>`_)
     - TTS model developed by Resemble AI. Using 0.5B LLM with flow matching + HiFT vocoder. Supports audio input for voice cloning.
   * - ``cosyvoice2``
     - ``FunAudioLLM/CosyVoice2-0.5B`` (`link <https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>`_)
     - TTS model developed by Alibaba. Using 0.5B LLM with flow matching + HiFT vocoder. Supports audio input for voice cloning.
   * - ``csm``
     - ``sesame/csm-1b`` (`link <https://huggingface.co/sesame/csm-1b>`_)
     - TTS model developed by Sesame. Using 1B LLM and depth-wise model with Mimi detokenizer.
   * - ``orpheus``
     - ``canopylabs/orpheus-3b-0.1-ft`` (`link <https://huggingface.co/canopylabs/orpheus-3b-0.1-ft>`_)
     - TTS model developed by Canopy Labs. Using 3B LLM with SNAC detokenizer.
   * - ``qwen3-tts``
     - | ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``
       | ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``
       | ``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign``
       | (`link <https://huggingface.co/collections/Qwen/qwen3-tts>`_)
     - TTS model developed by Alibaba Qwen team. Using 1.7B LLM with depth transformer and Mimi-like detokenizer.
   * - ``zonos``
     - ``Zyphra/Zonos-v0.1-transformer`` (`link <https://huggingface.co/Zyphra/Zonos-v0.1-transformer>`_)
     - TTS model developed by Zyphra. Using 1B LLM with DAC detokenizer.

Speech-to-speech (STS)
----------------------

.. list-table:: STS model families
   :header-rows: 1
   :widths: 12 30 32

   * - Family code
     - Example Hugging Face model ID
     - Description
   * - ``glm``
     - ``zai-org/glm-4-voice-9b`` (`link <https://huggingface.co/zai-org/glm-4-voice-9b>`_)
     - STS model developed by Z.ai. Using 9B LLM with flow matching + HiFT vocoder.
   * - ``step``
     - ``stepfun-ai/Step-Audio-2-mini`` (`link <https://huggingface.co/stepfun-ai/Step-Audio-2-mini>`_)
     - STS model developed by StepFun. Using 8B LLM with flow matching + HiFT vocoder.

Notes
-----

- The examples above are representative model IDs. You can use local paths or other
  compatible variants within each family.
- Some families support audio input (STS). Refer to the model card for input requirements.
