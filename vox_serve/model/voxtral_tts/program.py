"""Voxtral-4B-TTS (Mistral backbone + flow-matching acoustic head + 37-codebook codec) program.

Rows carry 37 columns. Prompt rows embed column 0 as a text token, except voice rows (input mask
column 0 == ``VOICE``) which take row ``tok[:, 1]`` of the concatenated voice-embedding table.
Decode rows (``AUDIO``) embed the previous frame: the sum of the 37 codebook embeddings.

The inline audio head runs inside the LM graph: semantic argmax, then a 7-step Euler ODE with
classifier-free guidance, starting from Philox noise keyed by (request seed, step) so that it does
not depend on graph replay counts. The codec keeps a 25-frame left context per request (stateful
detok slot).
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download

from ...rt.core import Arena, RtAttention
from ...rt.kernels import ops
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from ...utils import get_logger
from .model import (
    _AUDIO_SPECIAL_TOKENS,
    _EMPTY_AUDIO_TOKEN_ID,
    _END_AUDIO_TOKEN_ID,
    VoxtralTTSModel,
)

TEXT, VOICE, AUDIO = 0, 1, 2
NOISE_SALT = 1
LEFT_CONTEXT = 25


class _Model(VoxtralTTSModel):
    # mistral-common is only needed for prompt construction, which lives in Rust.
    def _load_tokenizer(self, model_name: str):
        return None


def export_tekken(path: str, out: str) -> Dict[str, int]:
    """Converts ``tekken.json`` into a `tokenizers` BPE whose ids match mistral-common's Tekkenizer.

    Returns the control-token ids by name.
    """
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    with open(path) as f:
        d = json.load(f)
    cfg = d["config"]
    n_special = cfg["default_num_special_tokens"]
    ranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in d["vocab"][: cfg["default_vocab_size"] - n_special]}
    enc = bytes_to_unicode()

    def s(b: bytes) -> str:
        return "".join(enc[x] for x in b)

    # Control tokens are never produced from text; placeholders keep the id space contiguous.
    vocab = {f"<SPECIAL_{i}>": i for i in range(n_special)}
    merges = []
    for tok, rank in ranks.items():
        vocab[s(tok)] = rank + n_special
        if len(tok) == 1:
            continue
        local = [(tok[:i], tok[i:], rank) for i in range(1, len(tok)) if tok[:i] in ranks and tok[i:] in ranks]
        local.sort(key=lambda x: (ranks[x[0]], ranks[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda x: x[2])
    t = Tokenizer(BPE(vocab, [(s(a), s(b)) for a, b, _ in merges], fuse_unk=False))
    t.model.ignore_merges = True
    t.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(cfg["pattern"]), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    t.decoder = decoders.ByteLevel()
    t.save(out)
    return {v["token_str"]: v["rank"] for v in d["special_tokens"]}


class VoxtralTTSProgram(LMProgram):
    uses_input_mask = True
    uses_repetition_penalty = False
    detok_stateful = True

    def load(self) -> None:
        name = self.config.get("model", "mistralai/Voxtral-4B-TTS-2603")
        if name == "voxtral-tts":
            name = "mistralai/Voxtral-4B-TTS-2603"
        self.model_name = name
        self.model = _Model(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        cfg = self.model.config
        at = self.model.acoustic_transformer
        self.n_codebooks = self.model.n_codebooks
        self.vocab_size = at.semantic_codebook_output.out_features
        self.num_layers = cfg.num_hidden_layers
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length
        self.first_chunk_frames = self.model.first_chunk_frames
        self.samples_per_frame = self.model.downsample_factor

        self.n_special = len(_AUDIO_SPECIAL_TOKENS)
        self.semantic_end = self.n_special + cfg.semantic_codebook_size
        self.levels = at.acoustic_embeddings_levels
        self.offsets = self.model._audio_codebook_offsets.long()
        self.cfg_alpha = float(self.config.get("cfg_alpha", self.model.default_sampling_config.cfg_scale or 1.2))

        self.voice_index: Dict[str, Dict[str, int]] = {}
        table = []
        n = 0
        for voice, emb in self.model.voice_to_embedding.items():
            emb = emb.reshape(-1, cfg.hidden_size)
            self.voice_index[voice] = {"offset": n, "len": emb.shape[0]}
            table.append(emb)
            n += emb.shape[0]
        if not table:
            raise RuntimeError("Voxtral: no voice embeddings loaded")
        self.voice_table = torch.cat(table).to(self.device, self.dtype)

    def allocate_state(self) -> None:
        super().allocate_state()
        n = self.max_batch_size
        self.noise = torch.zeros(n, self.n_codebooks - 1, dtype=torch.float32, device=self.device)
        self.cfg_buf = torch.full((n, 1), self.cfg_alpha, dtype=self.dtype, device=self.device)
        self.left_ctx = torch.zeros(
            self.detok_slots + 1,
            LEFT_CONTEXT,
            self.n_codebooks,
            dtype=torch.int64,
            device=self.model.audio_decoder_device,
        )

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model.model
        kind = arena["mask"][:rows, :1]
        text = m.embed_tokens(ids[:, 0].clamp(0, m.embed_tokens.num_embeddings - 1))
        voice = self.voice_table[ids[:, 1].clamp(0, self.voice_table.shape[0] - 1)]
        shifted = (ids + self.offsets[None, :]).clamp(0, m.audio_codebook_embeddings.num_embeddings - 1)
        # Bag-sum avoids a [rows, 37, H] intermediate in the large prefill buckets.
        audio = torch.nn.functional.embedding_bag(shifted, m.audio_codebook_embeddings.weight, mode="sum")
        h = torch.where(kind == VOICE, voice, text)
        return torch.where(kind == AUDIO, audio, h)

    def fuse_lm_modules(self):
        return [self.model.model, self.model.acoustic_transformer]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        at = self.model.acoustic_transformer
        B = hidden.shape[0]
        dtype = hidden.dtype

        logits = at.semantic_codebook_output(hidden).float()
        logits[:, _EMPTY_AUDIO_TOKEN_ID] = -float("inf")
        logits[:, self.semantic_end :] = -float("inf")
        semantic = logits.argmax(dim=-1)

        noise = self.noise[:B]
        ops().randn(noise, arena["seed"][:n_req], arena["step"][:n_req], NOISE_SALT)
        x = (at._noise_scale * noise).to(dtype)
        cfg = self.cfg_buf[:B]
        hidden_zero = torch.zeros_like(hidden)
        llm_batched = torch.cat([hidden, hidden_zero], dim=0)
        timesteps = at._timesteps.to(dtype=dtype)
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = at.time_embedding(t.view(-1, 1).repeat(B, 1)).to(dtype)
            v_all = at._predict_velocity(
                x_t=torch.cat([x, x], dim=0), llm_output=llm_batched, t_emb=torch.cat([t_emb, t_emb], dim=0)
            )
            v_t, uncond = v_all[:B], v_all[B:]
            x = x + (cfg * v_t + (1 - cfg) * uncond) * dt

        scaled = ((torch.clamp(x, -1, 1) + 1) / 2) * (self.levels - 1)
        acoustic = scaled.round().long()
        acoustic = torch.where((semantic == _END_AUDIO_TOKEN_ID)[:, None], _EMPTY_AUDIO_TOKEN_ID, acoustic)
        out_tok[:, 0].copy_(semantic)
        out_tok[:, 1:].copy_(acoustic + self.n_special)

    def detok_step(self, arena: Arena, out: Arena, bd: int) -> None:
        slots = arena["dt_slot"][:bd].long()
        keep = (arena["dt_reset"][:bd] == 0).long()[:, None, None]
        ctx = self.left_ctx.index_select(0, slots) * keep
        full = torch.cat([ctx, arena["dt_tok"][:bd].long()], dim=1)
        self.left_ctx.index_copy_(0, slots, full[:, -LEFT_CONTEXT:].contiguous())
        codes = (full - self.n_special).clamp(min=0)
        audio = self.model.audio_tokenizer.decode(codes.transpose(1, 2), dtype=self.dtype)
        audio = audio[..., LEFT_CONTEXT * self.samples_per_frame :].reshape(bd, self.detok_samples)
        out["audio"][:bd].copy_((audio.float() * 32767).to(torch.int16))

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """``[BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]``;
        column 1 of the ``[AUDIO]`` (voice) rows indexes the voice-embedding table."""
        if audio_path is not None:
            raise ValueError("audio_path is not supported yet for this model")
        m = self.spec_meta
        voices, default = m["voices"], m["default_voice"]
        v = model_kwargs.get("voice")
        requested = v if isinstance(v, str) else (v[0] if isinstance(v, list) and v and isinstance(v[0], str) else None)
        if requested is not None and requested not in voices:
            get_logger(__name__).warning("Voxtral voice %r not in registry; falling back to %r", requested, default)
            requested = None
        voice = voices[requested or default]
        C = self.n_codebooks
        tokens: List[int] = []
        mask: List[int] = []

        def push(tok: int, kind: int, voice_row: int = 0) -> None:
            tokens.extend([tok, voice_row] + [0] * (C - 2))
            mask.extend([kind] + [0] * (C - 1))

        push(m["bos"], TEXT)
        push(m["begin_audio"], TEXT)
        for k in range(int(voice["len"])):
            push(m["audio_token"], VOICE, int(voice["offset"]) + k)
        push(m["text_to_audio"], TEXT)
        for t in self._encode(text, add_special_tokens=False):
            push(t, TEXT)
        push(m["audio_to_text"], TEXT)
        push(m["begin_audio"], TEXT)
        return Prompt(tokens, mask)

    def stop_tokens(self) -> List[int]:
        return [_END_AUDIO_TOKEN_ID]

    def meta(self) -> Dict[str, Any]:
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "voxtral_tts_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        special = export_tekken(hf_hub_download(self.model_name, "tekken.json"), tok_path)
        cfg = self.model.config
        if special["[AUDIO]"] != cfg.audio_token_id or special["[BEGIN_AUDIO]"] != cfg.begin_audio_token_id:
            raise RuntimeError("Voxtral tekken.json special tokens disagree with the model config")
        return {
            "kind": "voxtral_tts",
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 24000,
            "detok_overlap": 0,
            "max_tokens": self.model.max_tokens,
            "bos": special["<s>"],
            "audio_token": cfg.audio_token_id,
            "begin_audio": cfg.begin_audio_token_id,
            # mistral-common's text_to_audio / audio_to_text tokens.
            "text_to_audio": special["[NEXT_AUDIO_TEXT]"],
            "audio_to_text": special["[REPEAT_AUDIO_TEXT]"],
            "voices": self.voice_index,
            "default_voice": next(iter(self.voice_index)),
            "row_kinds": {"text": TEXT, "voice": VOICE},
            "decode_mask": [AUDIO] + [0] * (self.n_codebooks - 1),
            "preseed_frames": self.detok_interval - self.first_chunk_frames,
            "cfg_alpha": self.cfg_alpha,
            "sampling": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0},
        }
