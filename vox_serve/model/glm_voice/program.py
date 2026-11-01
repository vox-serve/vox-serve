"""GLM-4-Voice (ChatGLM 9B with interleaved text / audio tokens + flow-matching decoder) program."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...rt.core import RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from .model import GLMVoiceModel


def export_chatglm_tokenizer(slow, path: str) -> None:
    """Converts the tiktoken-based ChatGLM4 tokenizer into a `tokenizers` JSON (checked for parity)."""
    from tokenizers import AddedToken, processors
    from transformers.convert_slow_tokenizer import TikTokenConverter

    tok = TikTokenConverter(
        vocab_file=slow.vocab_file, pattern=slow.tokenizer._pat_str, additional_special_tokens=[]
    ).converted()
    added = sorted(slow.added_tokens_decoder.items())
    if added[0][0] != tok.get_vocab_size():
        raise RuntimeError("unexpected ChatGLM added-token layout")
    tok.add_special_tokens([AddedToken(t.content, special=True, normalized=False) for _, t in added])
    gmask, sop = slow.convert_tokens_to_ids("[gMASK]"), slow.convert_tokens_to_ids("<sop>")
    tok.post_processor = processors.TemplateProcessing(
        single="[gMASK] <sop> $A", special_tokens=[("[gMASK]", gmask), ("<sop>", sop)]
    )
    for probe in (
        "<|system|>\nsys prompt.<|user|>\nHello, how are you? 12,345 héllo 你好<|assistant|>streaming_transcription\n",
        "<|begin_of_audio|><|audio_12|><|audio_3|><|end_of_audio|>  spaces \n\n",
    ):
        if slow(probe).input_ids != tok.encode(probe).ids:
            raise RuntimeError("converted ChatGLM tokenizer disagrees with the slow tokenizer")
    tok.save(path)


class GLMVoiceProgram(LMProgram):
    uses_repetition_penalty = False

    def load(self) -> None:
        name = self.config.get("model", "zai-org/glm-4-voice-9b")
        if name == "glm":
            name = "zai-org/glm-4-voice-9b"
        self.model_name = name
        self.model = GLMVoiceModel(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        self.lm = self.model.model
        self.n_codebooks = 1
        self.vocab_size = self.lm.transformer.output_layer.out_features
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = self.model.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

    def embed(self, ids: torch.Tensor, arena, rows: int) -> torch.Tensor:
        return self.lm.embed_tokens(ids[:, 0])

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.lm.transformer.encoder(hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.transformer.output_layer(hidden)[:, None, :]

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.postprocess(tokens)

    def hook_encode_audio(self, path: str) -> List[int]:
        return [int(x) for x in self.model._extract_speech_token(path)[0]]

    @staticmethod
    def _text_system_prompt() -> str:
        return (
            "User will provide you with a text instruction. Do it step by step. First, think about the "
            "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        )

    @staticmethod
    def _speech_system_prompt() -> str:
        return (
            "User will provide you with a speech instruction. Do it step by step. First, think about the "
            "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
        )

    def stop_tokens(self) -> List[int]:
        return list(self.model.stop_token_ids)

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """The chat template around the text, or around the ``<|audio_N|>`` tokens of the input clip."""
        m = self.spec_meta
        if audio_path is not None:
            codes = self.hook_encode_audio(audio_path)
            user = "<|begin_of_audio|>" + "".join(f"<|audio_{int(c)}|>" for c in codes) + "<|end_of_audio|>"
            s = m["speech_prompt_template"].replace("{prompt}", user, 1)
        else:
            s = m["text_prompt_template"].replace("{prompt}", text, 1)
        return Prompt(self._encode(s))

    def audio_tokens(self) -> Optional[Tuple[int, int]]:
        return (self.model.audio_offset, self.vocab_size)  # text tokens below: fed back, not vocoded

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "glm_voice_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        export_chatglm_tokenizer(self.model.text_tokenizer, tok_path)
        prompt = self.model._format_prompt("text", "{prompt}", None)
        if self._text_system_prompt() not in prompt:
            raise RuntimeError("GLM-4-Voice prompt format changed; update the rt program")
        return {
            "kind": "glm_voice",
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 22050,
            "detok_overlap": 0,
            "max_tokens": self.model.max_tokens,
            "text_prompt_template": prompt,
            "speech_prompt_template": prompt.replace(self._text_system_prompt(), self._speech_system_prompt()),
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": 1.0,
            },
        }
