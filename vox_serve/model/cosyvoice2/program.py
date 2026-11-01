"""CosyVoice2-0.5B (Qwen2 LM over speech tokens + flow / HiFT decoder) graph program.

Prompt rows are ``[sos] ref_text text [task] ref_speech``; the per-row kind (input mask column 0)
selects the embedding table: 0 ``llm_embedding`` (sos / task id), 1 text ``embed_tokens``,
2 ``speech_embedding`` (reference speech and every generated token). The decoder runs in the
reference's shared-prompt-cache mode, so detokenization is stateless.
"""

import os
from typing import Any, Dict, List, Optional

import torch

from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from .model import CosyVoice2Model

SPECIAL, TEXT, SPEECH = 0, 1, 2


_INFLECT = None


def text_normalize(text: str) -> str:
    """``CosyVoice2Model._text_normalize(text, split=False)``: strip and spell out (ASCII) digit runs
    (also in rust/src/models/cosyvoice2.rs)."""
    global _INFLECT
    if ("<|" in text and "|>" in text) or not text:
        return text
    if _INFLECT is None:
        import inflect

        _INFLECT = inflect.engine()
    text = text.strip()
    out: List[str] = []
    start = None
    for i, c in enumerate(text):
        if "0" <= c <= "9":
            if start is None:
                start = i
            continue
        if start is not None:
            out.append(_INFLECT.number_to_words(text[start:i]))
            start = None
        out.append(c)
    if start is not None:
        out.append(_INFLECT.number_to_words(text[start:]))
    return "".join(out)


class CosyVoice2Program(LMProgram):
    uses_input_mask = True
    uses_repetition_penalty = False

    def load(self) -> None:
        name = self.config.get("model", "FunAudioLLM/CosyVoice2-0.5B")
        if name == "cosyvoice2":
            name = "FunAudioLLM/CosyVoice2-0.5B"
        self.model_name = name
        self.model = CosyVoice2Model(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        if self.model.use_detokenizer_cache:
            raise NotImplementedError("rt runtime uses CosyVoice2's shared prompt cache mode")
        m = self.model.model
        self.n_codebooks = 1
        self.vocab_size = m.llm_decoder.out_features
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = self.model.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model.model
        kind = arena["mask"][:rows, :1]
        tok = ids[:, 0]
        special = m.llm_embedding(tok.clamp(0, m.llm_embedding.num_embeddings - 1))
        text = m.embed_tokens(tok.clamp(0, m.llm.model.model.embed_tokens.num_embeddings - 1))
        speech = m.embed_tokens_speech(tok.clamp(0, m.speech_embedding.num_embeddings - 1))
        h = torch.where(kind == SPECIAL, special, text)
        return torch.where(kind == SPEECH, speech, h)

    def fuse_lm_modules(self):
        return [self.model.model.llm]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.llm(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.model.llm_decoder(hidden)[:, None, :]

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.postprocess(tokens)

    def stop_tokens(self) -> List[int]:
        return list(self.model.stop_token_ids)

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """``[sos] ref_text text [task] ref_speech`` with the default (LJSpeech) reference."""
        if audio_path is not None:
            raise ValueError("audio_path is not supported yet for this model")
        m = self.spec_meta
        tokens = [m["sos"]] + list(m["ref_text_ids"]) + self._encode(text_normalize(text))
        mask = [SPECIAL] + [TEXT] * (len(tokens) - 1)
        tokens.append(m["task_id"])
        mask.append(SPECIAL)
        tokens += m["ref_speech_tokens"]
        mask += [SPEECH] * len(m["ref_speech_tokens"])
        return Prompt(tokens, mask)

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        ref = self.model.default_speaker_ref_dict
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "cosyvoice2_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        self.model.text_tokenizer.tokenizer.backend_tokenizer.save(tok_path)
        return {
            "kind": "cosyvoice2",
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "sos": self.model.sos,
            "task_id": self.model.task_id,
            "ref_text_ids": ref["ref_text_ids"][0].tolist(),
            "ref_speech_tokens": ref["prompt_speech_token"][0].tolist(),
            "row_kinds": {"special": SPECIAL, "text": TEXT, "speech": SPEECH},
            "decode_mask": [SPEECH],
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": 1.0,
            },
        }
