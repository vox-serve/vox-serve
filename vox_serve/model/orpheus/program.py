"""Orpheus (Llama-3.2-3B + SNAC 24 kHz) graph program."""

from typing import Any, Dict, List

import torch

from ...rt.core import RtAttention
from ...rt.program import LMProgram
from .model import OrpheusModel


class OrpheusProgram(LMProgram):
    supports_urgent_lane = True

    def load(self) -> None:
        name = self.config.get("model", "canopylabs/orpheus-3b-0.1-ft")
        if name == "orpheus":
            name = "canopylabs/orpheus-3b-0.1-ft"
        self.model_name = name
        self.model = OrpheusModel(
            name, device=str(self.device), audio_decoder_device=str(self.detok_device), tokenizer_path=name
        )
        self.lm = self.model.model
        cfg = self.lm.config
        self.n_codebooks = 1
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = self.model.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

    def embed(self, ids: torch.Tensor, arena, rows: int) -> torch.Tensor:
        return self.lm.model.embed_tokens(ids[:, 0])

    def fuse_lm_modules(self):
        return [self.lm.model]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.lm.model(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.lm_head(hidden)[:, None, :]

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.postprocess(tokens)

    def stop_tokens(self) -> List[int]:
        return [self.model.stop_token_id]

    def meta(self) -> Dict[str, Any]:
        from huggingface_hub import hf_hub_download

        sc = self.model.default_sampling_config
        return {
            "kind": "orpheus",
            "model_name": self.model_name,
            # `{voice}: {text}` between Orpheus' start-of-human / end-of-human-and-start-of-AI ids.
            "prompt_builder": "template",
            "prompt_template": {
                "tokenizer_json": hf_hub_download(self.model_name, "tokenizer.json"),
                "prefix_ids": [128259],
                "suffix_ids": [128009, 128260, 128261, 128257],
                "text": "{voice}: {text}",
                "kwargs": {"voice": {"default": "tara", "choices": list(self.model.available_voices)}},
            },
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
                "repetition_window": sc.repetition_window if sc.repetition_window is not None else -1,
            },
        }
