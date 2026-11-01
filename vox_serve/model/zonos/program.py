"""Zonos-v0.1-transformer (9 delayed DAC codebooks, prefix conditioning) graph program.

The prefix conditioner is row-wise (``norm(project(cat(conds)))``), so every prefix row is a lookup
into a table precomputed at load time: one row per phoneme symbol, followed by the constant rows
of the default conditioning (speaker, emotion, fmax, pitch_std, speaking_rate, language_id).
Prefix rows (input mask column 0 == ``PREFIX``) carry their table index in column 0; code rows
embed the sum of the 9 codebook embeddings. Phonemization (espeak) runs in a per-request hook.

Sampling applies the repetition penalty over a window of the last ``window`` frames (per
codebook), then the delay pattern masks codebooks ``j > step``.
"""

from typing import Any, Dict, List, Optional

import torch

from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from .model import ZonosModel, ZonosUtils

CODES, PREFIX = 0, 1


class ZonosProgram(LMProgram):
    supports_urgent_lane = True
    uses_input_mask = True
    uses_repetition_penalty = False  # windowed penalty implemented in decode_tokens

    def load(self) -> None:
        name = self.config.get("model", "Zyphra/Zonos-v0.1-transformer")
        if name == "zonos":
            name = "Zyphra/Zonos-v0.1-transformer"
        self.model_name = name
        self.model = ZonosModel(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        m = self.model.model
        self.n_codebooks = self.model.n_codebooks
        self.vocab_size = self.model.vocab_size
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = self.model.hidden_size // self.num_qo_heads
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length
        self.mask_id = self.model.masked_token_id
        sc = self.model.default_sampling_config
        self.window = max(1, int(sc.repetition_window or 1))

        pc = m.prefix_conditioner
        espeak, rest = pc.conditioners[0], pc.conditioners[1:]
        if espeak.name != "espeak":
            raise RuntimeError("Zonos prefix layout changed; update the rt program")
        with torch.no_grad():
            cond = self.model._make_cond_dict(text="", device=self.device)
            const = torch.cat([c(cond.get(c.name)).reshape(1, -1) for c in rest])
            phon = espeak.project(espeak.phoneme_embedder.weight)
            self.prefix_table = pc.norm(pc.project(torch.cat([phon, const.to(phon.dtype)]))).contiguous()
        self.n_phonemes = phon.shape[0]
        self.n_const = const.shape[0]

    def allocate_state(self) -> None:
        super().allocate_state()
        # Last `window` frames of every slot; `vocab_size` marks an empty entry.
        self.rep_hist = torch.full(
            (self.max_slots + 1, self.window, self.n_codebooks), self.vocab_size, dtype=torch.int64, device=self.device
        )

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        kind = arena["mask"][:rows, :1]
        prefix = self.prefix_table[ids[:, 0].clamp(0, self.prefix_table.shape[0] - 1)]
        codes = self.model.model.embed_tokens(ids.clamp(0, self.mask_id))
        return torch.where(kind == PREFIX, prefix, codes)

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.backbone(
            inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache
        )

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([h(hidden) for h in self.model.model.heads], dim=1)
        return logits + self.model.logit_bias

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        V = self.vocab_size
        logits = self.head(hidden)  # [N, 9, V]
        slot = arena["slot"][:n_req].long()
        first = arena["first"][:n_req] != 0
        hist = torch.where(first[:, None, None], V, self.rep_hist.index_select(0, slot))
        seen = torch.zeros(n_req, self.n_codebooks, V + 1, dtype=torch.bool, device=logits.device)
        seen.scatter_(2, hist.transpose(1, 2).clamp(0, V), True)
        seen = seen[:, :, :V]
        pen = arena["rep_pen"][:n_req].to(logits.dtype)[:, None, None]
        logits = torch.where(seen & (logits > 0), logits / pen, logits)
        logits = torch.where(seen & (logits <= 0), logits * pen, logits)
        self.sample_into(logits.contiguous(), arena, out_tok, n_req, self.LM_SALT)

        sampled = out_tok.long()
        self.rep_hist.index_copy_(0, slot, torch.cat([hist[:, 1:], sampled[:, None, :]], dim=1))
        # Delay pattern: codebook j only starts at step j.
        cols = torch.arange(self.n_codebooks, device=out_tok.device)
        delayed = cols[None, :] > arena["step"][:n_req, None]
        out_tok.copy_(torch.where(delayed, self.mask_id, sampled).to(out_tok.dtype))

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.postprocess(tokens)

    def hook_phonemize(self, text: str) -> List[int]:
        phonemes = ZonosUtils.phonemize([text], ["en-us"])
        ids, _ = ZonosUtils.tokenize_phonemes(phonemes)
        return [int(x) for x in ids[0]]

    def stop_tokens(self) -> List[int]:
        return [self.model.eos_token_id]

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """Phoneme rows and the constant conditioning rows (indices into the prefix table), then an
        all-MASK code row."""
        if audio_path is not None:
            raise ValueError("audio_path is not supported yet for this model")
        C = self.n_codebooks
        tokens: List[int] = []
        mask: List[int] = []

        def push(row: List[int], kind: int) -> None:
            tokens.extend(row)
            mask.extend([kind] + [0] * (C - 1))

        for i in self.hook_phonemize(text):
            push([min(max(int(i), 0), self.n_phonemes - 1)] + [0] * (C - 1), PREFIX)
        for k in range(self.n_const):
            push([self.n_phonemes + k] + [0] * (C - 1), PREFIX)
        push([self.mask_id] * C, CODES)
        return Prompt(tokens, mask)

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        return {
            "kind": "zonos",
            "model_name": self.model_name,
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "decode_mask": [CODES] + [0] * (self.n_codebooks - 1),
            "mask_token": self.mask_id,
            "n_phonemes": self.n_phonemes,
            "n_const_rows": self.n_const,
            "row_kinds": {"codes": CODES, "prefix": PREFIX},
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
            },
        }
