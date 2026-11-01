"""Step-Audio-2-mini (Qwen2.5-7B LM over text + audio tokens, flow-matching + HiFT token2wav).

Prompt rows whose input mask is ``m > 0`` take row ``m - 1`` of the feature pool (encoded input
audio, written by ``hook_encode_audio`` + ``hook_features_commit``) instead of the token embedding.

The token2wav decoder is stateful and starts from the preset-voice cache rather than zeros: a
detok reset copies the initial template into the request's state slot.
"""

from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram, pcm16
from ...rt.prompt import Prompt
from ...tokenizer.step_audio_2 import StepAudio2DecoderCache
from .model import StepAudio2Model

AUDIO_START = 151688
SEGMENTS = [
    "<|BOT|>system\nYou are a helpful assistant.<|EOT|>",
    "<|BOT|>human\n{prompt}<|EOT|>",
    "<|BOT|>assistant\n<tts_start>",
]


def _cache_tensors(cache: StepAudio2DecoderCache) -> List[torch.Tensor]:
    return [getattr(cache, f.name) for f in fields(cache)]


class StepAudio2Program(LMProgram):
    uses_input_mask = True
    detok_stateful = True
    feature_pool_rows = 4096

    def load(self) -> None:
        name = self.config.get("model", "stepfun-ai/Step-Audio-2-mini")
        if name == "step":
            name = "stepfun-ai/Step-Audio-2-mini"
        self.model_name = name
        self.model = StepAudio2Model(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        self.n_codebooks = 1
        self.vocab_size = self.model.vocab_size
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = self.model.hidden_size // self.num_qo_heads
        self.feature_dim = self.model.hidden_size
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

    def allocate_state(self) -> None:
        super().allocate_state()
        init = self.model._audio_decoder_initial_cache
        self._init_t = [t.contiguous() for t in _cache_tensors(init)]
        n = self.detok_slots + 1
        self.dec_state = [t.repeat(n, *([1] * (t.dim() - 1))).contiguous() for t in self._init_t]
        bd = self.max_detok_batch_size
        self.dec_work = StepAudio2DecoderCache(
            *[t.repeat(bd, *([1] * (t.dim() - 1))).contiguous() for t in self._init_t]
        )

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = arena["mask"][:rows, :1]
        text = self.model.model.embed_tokens(ids[:, 0])
        feat = self.feature_pool[(m[:, 0] - 1).clamp(0, self.feature_pool.shape[0] - 1).long()]
        return torch.where(m > 0, feat, text)

    def fuse_lm_modules(self):
        return [self.model.model.model]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.model(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.model.lm_head(hidden)[:, None, :]

    def detok_step(self, arena: Arena, out: Arena, bd: int) -> None:
        slots = arena["dt_slot"][:bd].long()
        reset = arena["dt_reset"][:bd] != 0
        work = self.dec_work[:bd]
        work_t = _cache_tensors(work)
        for s, init, w in zip(self.dec_state, self._init_t, work_t, strict=True):
            r = reset.view(-1, *([1] * (s.dim() - 1)))
            w.copy_(torch.where(r, init, s.index_select(0, slots)))
        tokens = arena["dt_tok"][:bd, :, 0].long() - self.model.audio_offset
        audio, new = self.model.audio_decoder(tokens.clamp(min=0), work)
        for s, n in zip(self.dec_state, _cache_tensors(new), strict=True):
            s.index_copy_(0, slots, n.to(s.dtype))
        if audio.shape[-1] != self.detok_samples:
            raise RuntimeError(f"Step-Audio-2 chunk has {audio.shape[-1]} samples, expected {self.detok_samples}")
        out["audio"][:bd].copy_(pcm16(audio.reshape(bd, -1)))

    def hook_encode_audio(self, path: str) -> Dict[str, Any]:
        """Full prompt for an audio-input request plus its encoded features (staged on device)."""
        prompt_ids, mels, mel_lengths = self.model._format_prompt("audio", None, path)
        ids = prompt_ids.view(-1)
        out, feat_lens = self.model.model.encoder(mels, mel_lengths)
        out = self.model.model.adapter(out)
        feat_lens = ((feat_lens - 1) // 2 + 1).tolist()
        starts = (torch.nonzero(ids == AUDIO_START)[:, 0] + 1).tolist()
        spans = [[int(s), int(n)] for s, n in zip(starts, feat_lens, strict=True)]
        feats = torch.cat([out[i, :n] for i, n in enumerate(feat_lens)])
        return {"ids": ids.tolist(), "spans": spans, "key": self._stage_features(feats), "rows": int(feats.shape[0])}

    def stop_tokens(self) -> List[int]:
        return [self.model.stop_token_id]

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """The chat-formatted text prompt, or (audio input) the prompt of ``hook_encode_audio`` whose
        audio-patch rows read the encoded features from the feature pool (mask = pool row + 1)."""
        if audio_path is not None:
            return self._audio_prompt(audio_path)
        tokens: List[int] = []
        for seg in SEGMENTS:
            tokens += self._encode(seg.replace("{prompt}", text, 1))
        return Prompt(tokens, [0] * len(tokens))

    def _audio_prompt(self, path: str) -> Prompt:
        r = self.hook_encode_audio(path)
        try:
            lease = self._features().alloc(int(r["rows"]))
            self.hook_features_commit(r["key"], lease.offset)
            tokens = [int(x) for x in r["ids"]]
            mask = [0] * len(tokens)
            row = lease.offset
            for start, n in r["spans"]:
                for j in range(start, min(start + n, len(mask))):
                    mask[j] = row + 1
                    row += 1
            return Prompt(tokens, mask, features=lease)
        except Exception:
            self.hook_features_discard(r["key"])
            raise

    def audio_tokens(self) -> Optional[Tuple[int, int]]:
        return (self.model.audio_offset, self.vocab_size)  # text tokens below: fed back, not vocoded

    def meta(self) -> Dict[str, Any]:
        from huggingface_hub import hf_hub_download

        sc = self.model.default_sampling_config
        return {
            "kind": "step_audio_2",
            "model_name": self.model_name,
            "tokenizer_json": hf_hub_download(self.model_name, "tokenizer.json"),
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "prompt_segments": SEGMENTS,
            "decode_mask": [0],
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
            },
        }
