"""CSM-1B (Llama backbone + depth decoder + Mimi) graph program.

One LM graph per step covers the backbone, the codebook-0 head and all depth-decoder iterations
(dense SDPA KV per batch entry), so a frame of 32 codebooks costs a single graph launch. The
SilentCipher watermark is applied to every detokenized chunk inside the detok graph.
"""

import os
from typing import Any, Dict, List, Optional

import torch

from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from ...rt.watermark import SilentCipherBatched
from .model import CSMModel


class CSMProgram(LMProgram):
    supports_urgent_lane = True
    uses_input_mask = True
    uses_repetition_penalty = False

    def load(self) -> None:
        name = self.config.get("model", "sesame/csm-1b")
        if name == "csm":
            name = "sesame/csm-1b"
        self.model_name = name
        self.tokenizer_path = self.config.get("tokenizer_path", "unsloth/Llama-3.2-1B")
        self.model = CSMModel(
            name,
            device=str(self.device),
            audio_decoder_device=str(self.detok_device),
            tokenizer_path=self.tokenizer_path,
        )
        m = self.model.model
        cfg = m.config
        dcfg = cfg.depth_decoder_config
        self.n_codebooks = self.model.n_codebooks  # 32 audio + text
        self.depth_n = dcfg.num_codebooks
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = self.model.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

        depth_attn = m.depth_decoder.model.layers[0].self_attn
        self.depth_kv = torch.zeros(
            dcfg.num_hidden_layers,
            self.max_batch_size,
            2,
            self.depth_n,
            dcfg.num_key_value_heads,
            depth_attn.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        n = self.max_batch_size
        self.depth_pos = [
            torch.tensor([[0, 1]] if i == 1 else [[i]], dtype=torch.int32, device=self.device).repeat(n, 1)
            for i in range(self.depth_n)
        ]
        self.watermark = bool(self.config.get("watermark", True))
        self.silentcipher = (
            SilentCipherBatched(24000, self.detok_samples, self.detok_device) if self.watermark else None
        )

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model.model
        text = m.embed_text_tokens(ids[:, -1:])
        audio = m.backbone_model.embed_tokens(ids[:, :-1])
        mask = arena["mask"][:rows].to(text.dtype)
        return (torch.cat([audio, text], dim=1) * mask[:, :, None]).sum(dim=1)

    def fuse_lm_modules(self):
        return [self.model.model.backbone_model, self.model.model.depth_decoder]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.backbone_model(hidden, pos, attn, self.kv_cache)

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        m = self.model.model
        N = n_req
        logits0 = m.lm_head(hidden)[:, None, :]
        self.sample_into(logits0, arena, out_tok[:, 0:1], N, salt=0)
        x = torch.stack([hidden, self.model.embed_audio_tokens_single(out_tok[:, 0].long(), 0)], dim=1)
        logits = m.forward_depth_unrolled(x, self.depth_pos[1][:N], self.depth_kv, 0)
        self.sample_into(logits[:, -1:, :], arena, out_tok[:, 1:2], N, salt=1)
        for i in range(2, self.depth_n):
            x = self.model.embed_audio_tokens_single(out_tok[:, i - 1].long(), i - 1)[:, None, :]
            logits = m.forward_depth_unrolled(x, self.depth_pos[i][:N], self.depth_kv, i)
            self.sample_into(logits, arena, out_tok[:, i : i + 1], N, salt=i)
        # Text column: unused in decode (masked out) and dropped by the detokenizer.
        out_tok[:, self.depth_n :].zero_()

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        audio = self.model.postprocess(tokens)
        if self.silentcipher is not None:
            audio = self.silentcipher(audio.reshape(audio.shape[0], -1)).reshape(audio.shape)
        return audio

    def stop_tokens(self) -> List[int]:
        return [self.model.stop_token_id]

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """The default two-speaker context, then ``[speaker]text`` in the text column."""
        if audio_path is not None:
            raise ValueError("CSM does not accept audio input")
        sp = model_kwargs.get("speaker")
        if isinstance(sp, str):
            speaker = sp
        elif isinstance(sp, (int, float)) and not isinstance(sp, bool):
            speaker = str(sp)
        else:
            speaker = "0"
        c = self.spec_meta["n_codebooks"]
        tokens, mask = list(self.spec_meta["context_tokens"]), list(self.spec_meta["context_mask"])
        for t in self._encode(f"[{speaker}]{text}"):
            tokens += [0] * (c - 1) + [t]
            mask += [0] * (c - 1) + [1]
        return Prompt(tokens, mask)

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        ctx = torch.cat(self.model.default_context["tokens"], dim=0).int().cpu()
        ctx_mask = torch.cat(self.model.default_context["tokens_mask"], dim=0).int().cpu()
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "csm_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        # Saved with the BOS/EOS template post-processor that CSM installs.
        self.model.text_tokenizer._tokenizer.save(tok_path)
        decode_mask = [1] * self.depth_n + [0] * (self.n_codebooks - self.depth_n)
        return {
            "kind": "csm",
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "context_tokens": ctx.flatten().tolist(),
            "context_mask": ctx_mask.flatten().tolist(),
            "decode_mask": decode_mask,
            "watermark": "silentcipher" if self.watermark else None,
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": 1.0,
            },
        }
