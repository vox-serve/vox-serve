"""The models: one package per model. ``<name>/program.py`` is the model's implementation as both runtimes
run it (GPU stages captured as CUDA graphs, token kinds, prompts, ``meta``), built on the PyTorch modules
of ``<name>/model.py``.
"""

import importlib
from typing import Optional, Tuple, Type

# Model name (lower case; exact match, else substring) -> (package, program class).
_PROGRAMS = {
    "orpheus": ("orpheus", "OrpheusProgram"),
    "canopylabs/orpheus": ("orpheus", "OrpheusProgram"),
    "chatterbox": ("chatterbox", "ChatterboxProgram"),
    "cosyvoice2": ("cosyvoice2", "CosyVoice2Program"),
    "funaudiollm/cosyvoice2": ("cosyvoice2", "CosyVoice2Program"),
    "cosyvoice3": ("cosyvoice3", "CosyVoice3Program"),
    "funaudiollm/fun-cosyvoice3": ("cosyvoice3", "CosyVoice3Program"),
    "csm": ("csm", "CSMProgram"),
    "sesame/csm": ("csm", "CSMProgram"),
    "glm": ("glm_voice", "GLMVoiceProgram"),
    "zai-org/glm-4-voice": ("glm_voice", "GLMVoiceProgram"),
    "higgs-tts": ("higgs_tts", "HiggsTTSProgram"),
    "bosonai/higgs-tts": ("higgs_tts", "HiggsTTSProgram"),
    "higgs-audio-v2": ("higgs_audio_v2", "HiggsAudioV2Program"),
    "bosonai/higgs-audio-v2": ("higgs_audio_v2", "HiggsAudioV2Program"),
    "qwen3-tts": ("qwen3_tts", "Qwen3TTSProgram"),
    "qwen/qwen3-tts": ("qwen3_tts", "Qwen3TTSProgram"),
    "step": ("step_audio_2", "StepAudio2Program"),
    "stepfun-ai/step-audio-2": ("step_audio_2", "StepAudio2Program"),
    "zonos": ("zonos", "ZonosProgram"),
    "zyphra/zonos": ("zonos", "ZonosProgram"),
    "voxtral-tts": ("voxtral_tts", "VoxtralTTSProgram"),
    "mistralai/voxtral-4b-tts": ("voxtral_tts", "VoxtralTTSProgram"),
    "xtts": ("xtts", "XTTSProgram"),
    "coqui/xtts-v2": ("xtts", "XTTSProgram"),
}


def find_program(model_name: str) -> Optional[Tuple[str, str]]:
    """(package, class) of the model's program, or None."""
    name = model_name.lower()
    if name in _PROGRAMS:
        return _PROGRAMS[name]
    return next((target for pattern, target in _PROGRAMS.items() if pattern in name), None)


def get_program(model_name: str) -> Type:
    match = find_program(model_name)
    if match is None:
        raise ValueError(f"unsupported model '{model_name}'. Known: {list(_PROGRAMS)}")
    return getattr(importlib.import_module(f".{match[0]}.program", __name__), match[1])
