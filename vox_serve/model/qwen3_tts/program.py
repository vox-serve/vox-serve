"""Qwen3-TTS (talker + code predictor + streaming 12 Hz codec) graph program.

Columns: ``[c0..c15, text]``. The talker embeds ``text_projection(text)`` plus, per the input mask,
codebook 0 (``codec_embedding``) and codebooks 1..15 (``code_predictor.codec_embedding[i-1]``), so
the next decode step's input is the full previous frame. Only the audio columns are fed back from
``last_tok``; the text column always comes from the host (``tts_pad`` or a streamed text token).

Base (voice clone) models: the speaker-embedding row reads its embedding from the feature pool
(input mask of the text column = pool row + 1, written by ``hook_encode_voice``); ICL reference-code
rows are ordinary rows with all 16 codebook columns unmasked.

The codec decoder is causal and stateful: each request owns a detok state slot holding its conv /
sliding-window attention caches, gathered into the batch before and scattered back after a chunk.
"""

import os
from dataclasses import fields
from typing import Any, Dict, List, Optional

import flashinfer
import torch
import torch.nn.functional as F

from ...flashinfer_utils import apply_rope_pos_ids
from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from ...utils import download_audio_from_url, get_logger
from .model import Qwen3TTSModel

logger = get_logger(__name__)

_ALIASES = {
    "qwen3-tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3-tts-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen3-tts-voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
DEFAULT_REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)
# Decoder cache members that carry state across chunks (the rest are scratch work buffers).
_STATE_FIELDS = (
    "attention_cache",
    "position_offset",
    "pre_conv_cache",
    "upsample_conv_caches",
    "decoder_conv_caches",
    "transconv_caches",
)


def _tensors(cache, names) -> List[torch.Tensor]:
    out = []
    for f in fields(cache):
        if f.name not in names:
            continue
        v = getattr(cache, f.name)
        out.extend(v if isinstance(v, list) else [v])
    return out


@torch.compiler.disable
def _fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    # Kept out of Dynamo: the FlashInfer op's fake kernel touches data pointers. (With
    # --enable-torch-compile the talker loop graph-breaks at attention and runs eagerly anyway.)
    flashinfer.norm.fused_add_rmsnorm(x, residual, weight, eps)


class Qwen3TTSProgram(LMProgram):
    uses_input_mask = True
    detok_stateful = True
    supports_depth_split = True
    supports_layer_split = True
    supports_urgent_lane = True

    def load(self) -> None:
        name = self.config.get("model", "qwen3-tts")
        name = _ALIASES.get(name, name)
        self.model_name = name
        self.model = Qwen3TTSModel(
            name,
            device=str(self.device),
            audio_decoder_device=str(self.detok_device),
            detokenize_interval=self.config.get("detok_interval"),
        )
        self.tts_type = self.model.tts_model_type
        if self.tts_type not in ("custom_voice", "base", "voice_design"):
            raise NotImplementedError(f"unknown Qwen3-TTS model type {self.tts_type}")
        if self.tts_type == "base":
            # One speaker-embedding row per request between preprocessing and prefill.
            self.feature_pool_rows = 256
            self.feature_dim = self.model.hidden_size
        cfg = self.model.config
        tcfg = cfg.talker_config
        self.depth_n = self.model.depth_n_codebooks
        self.n_codebooks = self.model.n_codebooks
        self.fb_columns = self.depth_n
        self.vocab_size = self.model.vocab_size
        self.num_layers = self.model.num_hidden_layers
        self.hidden_size = self.model.hidden_size
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = self.model.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

        talker = self.model.model.talker
        self.talker = talker
        self.cp = talker.code_predictor
        cp_attn = self.cp.model.layers[0].self_attn
        self.depth_kv = torch.zeros(
            len(self.cp.model.layers),
            self.max_batch_size,
            2,
            self.depth_n,
            cp_attn.num_key_value_heads,
            cp_attn.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        n = self.max_batch_size
        self.depth_pos = [
            torch.tensor([[0, 1]] if i == 1 else [[i]], dtype=torch.int32, device=self.device).repeat(n, 1)
            for i in range(self.depth_n)
        ]
        if self.config.get("fuse_depth"):
            self._prepare_fused_depth()
        if self.config.get("fuse_talker"):
            self._prepare_fused_talker()
        self.suppress = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        self.suppress[self.model.suppress_tokens] = True
        self.tts_pad = cfg.tts_pad_token_id

    # ---- forward pieces -----------------------------------------------------------------------

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        mask = arena["mask"][:rows]
        text = self.talker.text_projection(self.talker.model.text_embedding(ids[:, -1]))
        c0 = self.talker.model.codec_embedding(ids[:, 0])
        h = torch.where(mask[:, :1] != 0, text + c0, text)
        # Codebooks 1..15, accumulated in the same order as the reference's per-request features.
        acc = torch.zeros_like(h)
        for i in range(1, self.depth_n):
            e = self.cp.model.codec_embedding[i - 1](ids[:, i])
            acc = acc + e * mask[:, i : i + 1].to(e.dtype)
        h = h + acc
        if self.feature_pool_rows:
            m = mask[:, -1:]
            feat = self.feature_pool[(m[:, 0] - 1).clamp(0, self.feature_pool.shape[0] - 1).long()]
            h = h + torch.where(m > 0, feat, torch.zeros_like(feat))
        return h

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        if self.config.get("fuse_talker"):
            return self.talker.model.norm(self._talker_layers(hidden, pos, attn, 0, self.num_layers))
        return self.talker.model(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def backbone_range(self, hidden, pos, attn, l0, l1, last):
        tm = self.talker.model
        if self.config.get("fuse_talker"):
            hidden = self._talker_layers(hidden, pos, attn, l0, l1)
        else:
            for i in range(l0, l1):
                hidden = tm.layers[i](
                    hidden_states=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache[i]
                )
        return tm.norm(hidden) if last else hidden

    # ---- fused talker -------------------------------------------------------------------------

    def _prepare_fused_talker(self) -> None:
        """Merged Q/K/V and gate/up weights of the talker's layers, used by ``_talker_layers``."""
        self._ft = []
        for layer in self.talker.model.layers:
            a, mlp = layer.self_attn, layer.mlp
            qkv_w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
            qkv_b = None
            if a.q_proj.bias is not None:
                qkv_b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0).contiguous()
            gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
            self._ft.append((qkv_w, qkv_b, gu_w))
        logger.info("fused talker: merged QKV and gate/up projections, fused residual add + RMSNorm")

    def _talker_layers(self, h: torch.Tensor, pos: torch.Tensor, attn: RtAttention, l0: int, l1: int):
        """Talker layers [l0, l1) with merged QKV / gate-up GEMMs and the post-attention residual add
        fused into its RMSNorm (FlashInfer ``fused_add_rmsnorm``). Same math as the module forward."""
        tm = self.talker.model
        a0 = tm.layers[0].self_attn
        nq, nkv, hd = a0.num_attention_heads, a0.num_key_value_heads, a0.head_dim
        h = h.contiguous()
        for li in range(l0, l1):
            layer = tm.layers[li]
            a = layer.self_attn
            qkv_w, qkv_b, gu_w = self._ft[li]
            q, k, v = F.linear(layer.input_layernorm(h), qkv_w, qkv_b).split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            q = a.q_norm(q.reshape(-1, hd)).view(-1, nq, hd)
            k = a.k_norm(k.reshape(-1, hd)).view(-1, nkv, hd)
            q, k = apply_rope_pos_ids(q, k, pos, interleave=False, rope_theta=a.rope_theta)
            attn.set_kv_cache(self.kv_cache[li], k, v.reshape(-1, nkv, hd))
            o = a.o_proj(attn.run(q, self.kv_cache[li]).reshape(h.shape[0], -1))
            post = layer.post_attention_layernorm
            # In place: h += o; o = rmsnorm(h) * weight.
            _fused_add_rmsnorm(o, h, post.weight, post.variance_epsilon)
            g, u = F.linear(o, gu_w).chunk(2, dim=-1)
            h = h + layer.mlp.down_proj(F.silu(g) * u)
        return h

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        self.decode_head(hidden, arena, out_tok, n_req)
        self.decode_depth(hidden, arena, out_tok, n_req)

    def decode_head(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        """Codebook 0 from the talker's codec head."""
        logits0 = self.talker.codec_head(hidden)
        logits0 = logits0.masked_fill(self.suppress, torch.finfo(logits0.dtype).min)[:, None, :]
        self.sample_into(logits0, arena, out_tok[:, 0:1], n_req, salt=0, penalty=True)

    # ---- fused depth transformer --------------------------------------------------------------

    def _prepare_fused_depth(self) -> None:
        """Merged Q/K/V and gate/up weights of the code predictor's layers (one GEMM each instead of
        three / two at tiny M), used by ``_depth_forward``."""
        self._fd = []
        for layer in self.cp.model.layers:
            a, mlp = layer.self_attn, layer.mlp
            qkv_w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
            qkv_b = None
            if a.q_proj.bias is not None:
                qkv_b = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0).contiguous()
            gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
            self._fd.append((qkv_w, qkv_b, gu_w))
        logger.info("fused depth transformer: merged QKV and gate/up projections")

    def _depth_forward(self, x: torch.Tensor, pos: torch.Tensor, kv_cache: torch.Tensor, cache_pos: int):
        """Same math as ``forward_depth_unrolled`` with fewer kernels: merged QKV and gate/up GEMMs,
        grouped-query SDPA without materializing repeated KV heads, and the step's output head taken
        as a static view instead of an ``index_select`` copy."""
        cp = self.cp
        h = cp.small_to_mtp_projection(x)
        bs, sl, H = h.shape
        a0 = cp.model.layers[0].self_attn
        nq, nkv, hd = a0.num_attention_heads, a0.num_key_value_heads, a0.head_dim
        flat_pos = pos.reshape(-1)
        for li, layer in enumerate(cp.model.layers):
            a, mlp = layer.self_attn, layer.mlp
            qkv_w, qkv_b, gu_w = self._fd[li]
            res = h
            hn = layer.input_layernorm(h.view(-1, H))
            q, k, v = F.linear(hn, qkv_w, qkv_b).split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            q = a.q_norm(q.reshape(-1, hd)).view(bs * sl, nq, hd)
            k = a.k_norm(k.reshape(-1, hd)).view(bs * sl, nkv, hd)
            q, k = apply_rope_pos_ids(q, k, flat_pos, rope_theta=a.rope_theta, interleave=False)
            kv_cache[li, :bs, 0, cache_pos : cache_pos + sl] = k.view(bs, sl, nkv, hd)
            kv_cache[li, :bs, 1, cache_pos : cache_pos + sl] = v.reshape(bs, sl, nkv, hd)
            kf = kv_cache[li, :bs, 0, : cache_pos + sl].transpose(1, 2)
            vf = kv_cache[li, :bs, 1, : cache_pos + sl].transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q.view(bs, sl, nq, hd).transpose(1, 2), kf, vf, is_causal=sl > 1, enable_gqa=nq != nkv
            )
            h = res + a.o_proj(o.transpose(1, 2).reshape(bs, sl, nq * hd))
            res = h
            g, u = F.linear(layer.post_attention_layernorm(h.view(-1, H)), gu_w).chunk(2, dim=-1)
            h = res + mlp.down_proj(F.silu(g) * u).view(bs, sl, H)
        h = cp.model.norm(h.view(-1, H)).view(bs, sl, H)
        head = max(cache_pos + sl - 1, 1) - 1
        return h @ cp.lm_head_weight[head].t()

    def decode_depth(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        """Codebooks 1..15 from the depth transformer (code predictor)."""
        m = self.model.model
        fwd = self._depth_forward if self.config.get("fuse_depth") else m.forward_depth_unrolled
        N = n_req
        x = torch.stack([hidden, self.talker.model.codec_embedding(out_tok[:, 0].long())], dim=1)
        logits = fwd(x, self.depth_pos[1][:N], self.depth_kv, 0)
        self.sample_into(logits[:, -1:, :], arena, out_tok[:, 1:2], N, salt=1)
        for i in range(2, self.depth_n):
            x = self.cp.model.codec_embedding[i - 2](out_tok[:, i - 1].long())[:, None, :]
            logits = fwd(x, self.depth_pos[i][:N], self.depth_kv, i)
            self.sample_into(logits, arena, out_tok[:, i : i + 1], N, salt=i)
        out_tok[:, self.depth_n :].fill_(self.tts_pad)

    # ---- stateful codec -----------------------------------------------------------------------

    def compile_detok_targets(self):
        # Only the decoder: compiling the whole step would functionalize the in-place scatter into the
        # per-slot state store into a copy of the entire store on every call.
        return [(self.model.audio_decoder, "decode_chunk")]

    def allocate_state(self) -> None:
        super().allocate_state()
        dec = self.model.audio_decoder
        dev = self.model.audio_decoder_device
        # Normalize the RVQ codebooks once, before any graph (or torch.compile trace) sees them.
        with torch.no_grad():
            for mod in dec.modules():
                if hasattr(mod, "_embedding") and hasattr(mod, "embedding_sum"):
                    mod._embedding = mod.embedding_sum / mod.cluster_usage.clamp(min=mod.epsilon)[:, None]
        self.dec_state = dec.init_cache(self.detok_slots + 1, dev, self.dtype, self.detok_interval)
        self.dec_work = dec.init_cache(self.max_detok_batch_size, dev, self.dtype, self.detok_interval)
        if self.urgent_lane:
            # Separate scratch state for the urgent lane, which runs concurrently with lane 0.
            self.dec_work_u = dec.init_cache(max(self._urgent_sizes()), dev, self.dtype, self.detok_interval)
        # Only the persistent members of the per-slot store are needed.
        for f in fields(self.dec_state):
            if f.name not in _STATE_FIELDS:
                setattr(self.dec_state, f.name, None)
        self._state_t = _tensors(self.dec_state, _STATE_FIELDS)

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        slots = arena["dt_slot"][:bd].long()
        keep = arena["dt_reset"][:bd] == 0
        work = (self.dec_work if lane == 0 else self.dec_work_u)[:bd]
        work_t = _tensors(work, _STATE_FIELDS)
        for s, w in zip(self._state_t, work_t, strict=True):
            g = s.index_select(0, slots)
            w.copy_(g * keep.view(-1, *([1] * (g.dim() - 1))).to(g.dtype))
        codes = arena["dt_tok"][:bd, :, : self.depth_n].long().transpose(1, 2)
        codes = codes.clamp(0, self.model.depth_vocab_size - 1)
        audio, _ = self.model.audio_decoder.decode_chunk(codes, decoder_cache=work)
        for s, w in zip(self._state_t, work_t, strict=True):
            s.index_copy_(0, slots, w)
        audio = audio.reshape(bd, self.detok_samples)
        out["audio"][:bd].copy_((audio.float() * 32767).to(torch.int16))

    # ---- voice cloning (Base) ------------------------------------------------------------------

    def hook_encode_voice(self, path: Optional[str], ref_text: Optional[str], x_vector_only: bool) -> Dict[str, Any]:
        """Speaker embedding (staged for the feature pool) and, for ICL, the reference codes.

        Without reference audio (or, for ICL, without its transcript) the default reference is used.
        """
        if path is None or (ref_text is None and not x_vector_only):
            path, ref_text = str(download_audio_from_url(DEFAULT_REF_AUDIO)), DEFAULT_REF_TEXT
        m = self.model
        audio, sr = m._load_audio_to_np(path)
        sr_spk = m.speaker_encoder_sample_rate
        audio_spk = audio
        if sr != sr_spk:
            import librosa
            import numpy as np

            audio_spk = librosa.resample(y=audio.astype(np.float32), orig_sr=int(sr), target_sr=sr_spk)
        spk = m._extract_speaker_embedding(audio=audio_spk, sr=sr_spk)
        out: Dict[str, Any] = {"key": self._stage_features(spk.reshape(1, -1).to(self.dtype)), "rows": 1}
        if not x_vector_only:
            out["codes"] = m._encode_audio_to_codes(audio, sr).tolist()
            out["ref_text"] = ref_text
        return out

    # ---- prompt: rows ``[c0..c15, text]``; mask column i < 16 adds codebook i's embedding to the text
    # embedding (prompt rows use column 0, ICL reference rows and decode rows all 16); on Base models a
    # nonzero text-column mask adds feature-pool row ``m - 1`` (the speaker embedding).

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        ids = self._encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
        if len(ids) < 8:
            raise ValueError("unexpected tokenization of the prompt template")
        return self._build(ids[3:-5], audio_path, model_kwargs, streaming=False)

    def tokenize_stream_text(self, text: str) -> List[int]:
        return self._encode(text, add_special_tokens=False)

    def preprocess_streaming(
        self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any], first: int
    ) -> Prompt:
        """``<|im_start|>assistant\\n{text}`` without the closing template: the last text row carries
        ``codec_bos`` and there is no ``tts_eos`` row; the remaining text arrives on decode rows."""
        return self._build([first], audio_path, model_kwargs, streaming=True)

    def _build(self, text: List[int], audio_path: Optional[str], kwargs: Dict[str, Any], streaming: bool) -> Prompt:
        m = self.spec_meta
        tts_type = m["tts_model_type"]
        if audio_path is not None and tts_type != "base":
            raise ValueError("this Qwen3-TTS model does not accept audio input (voice cloning needs the Base model)")
        voice = self._encode_voice(audio_path, kwargs, streaming) if tts_type == "base" else None
        log = get_logger(__name__)
        languages, speakers = m["languages"], m["speakers"]
        lang = kwargs.get("language")
        language = (lang if isinstance(lang, str) else m["default_language"]).lower()
        if language != "auto" and language not in languages:
            log.warning("language %s not supported, using auto", language)
            language = "auto"
        speaker = None
        if tts_type == "custom_voice":
            sp = kwargs.get("speaker")
            speaker = (sp if isinstance(sp, str) else m["default_speaker"]).lower()
            if speaker not in speakers:
                log.warning("speaker %s not supported, using %s", speaker, m["default_speaker"])
                speaker = m["default_speaker"]
        instruct = kwargs.get("instruct")
        instruct = instruct if isinstance(instruct, str) and instruct and m.get("supports_instruct") else None
        language_id = None if language == "auto" else languages[language]
        if speaker is not None and language in ("chinese", "auto") and speaker in m["speaker_dialect"]:
            language_id = languages.get(m["speaker_dialect"][speaker], language_id)

        c = m["n_codebooks"]
        tokens: List[int] = []
        mask: List[int] = []

        def row(t: int, codec: Optional[int]) -> None:
            tokens.extend([0 if codec is None else codec] + [0] * (c - 2) + [t])
            mask.extend([int(codec is not None)] + [0] * (c - 1))

        if instruct is not None:
            for t in self._encode(f"<|im_start|>user\n{instruct}<|im_end|>\n"):
                row(t, None)
        for t in self._encode("<|im_start|>assistant\n"):
            row(t, None)
        if language_id is None:
            prefix = [m["codec_nothink"], m["codec_think_bos"], m["codec_think_eos"]]
        else:
            prefix = [m["codec_think"], m["codec_think_bos"], language_id, m["codec_think_eos"]]
        for t in prefix:
            row(m["tts_pad"], t)
        if speaker is not None:
            row(m["tts_pad"], speakers[speaker])
        if voice is not None:
            row(m["tts_pad"], None)  # text embedding + speaker embedding (no codec embedding)
            mask[-1] = voice["lease"].offset + 1
        row(m["tts_bos"], m["codec_pad"])
        icl = voice["icl"] if voice is not None else None
        if icl is not None:
            for t in icl[0]:
                row(t, m["codec_pad"])
        for k, t in enumerate(text):
            row(t, m["codec_bos"] if streaming and k + 1 == len(text) else m["codec_pad"])
        if not streaming:
            row(m["tts_eos"], m["codec_pad"])
            row(m["tts_pad"], m["codec_bos"])
        if icl is not None:
            for frame in icl[1]:
                tokens.extend(list(frame) + [m["tts_pad"]])
                mask.extend([1] * (c - 1) + [0])
        return Prompt(tokens, mask, features=voice["lease"] if voice is not None else None)

    def _encode_voice(self, audio_path: Optional[str], kwargs: Dict[str, Any], streaming: bool) -> Dict[str, Any]:
        """Runs ``hook_encode_voice`` and commits the speaker embedding to a feature-pool row."""
        x_vector_only = kwargs.get("x_vector_only_mode") is True
        if streaming and not x_vector_only:
            raise ValueError(
                "Input streaming is not supported with ICL mode (voice cloning with reference audio). "
                "Please use x_vector_only_mode=True or disable input streaming."
            )
        ref_text = kwargs.get("ref_text")
        r = self.hook_encode_voice(audio_path, ref_text if isinstance(ref_text, str) else None, x_vector_only)
        try:
            lease = self._features().alloc(1)
            self.hook_features_commit(r["key"], lease.offset)
            icl = None
            if isinstance(r.get("codes"), list) and isinstance(r.get("ref_text"), str):
                ids = self._encode(f"<|im_start|>assistant\n{r['ref_text']}<|im_end|>\n")
                if len(ids) < 5:
                    raise ValueError("unexpected tokenization of the reference text")
                codes = [[int(x) for x in f] for f in r["codes"]]
                if any(len(f) != self.n_codebooks - 1 for f in codes):
                    raise ValueError("encode_voice: malformed reference codes")
                icl = (ids[3:-2], codes)
            return {"lease": lease, "icl": icl}
        except Exception:
            self.hook_features_discard(r["key"])
            raise

    def stop_tokens(self) -> List[int]:
        return [self.model.stop_token_id]

    def meta(self) -> Dict[str, Any]:
        cfg = self.model.config
        t = cfg.talker_config
        sc = self.model.default_sampling_config
        tok_dir = os.path.join(os.environ.get("VOX_RT_CACHE", "/tmp/vox_rt_cache"), "qwen3_tts_tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok_path = os.path.join(tok_dir, "tokenizer.json")
        self.model.text_tokenizer.backend_tokenizer.save(tok_path)
        return {
            "kind": "qwen3_tts",
            "tts_model_type": self.tts_type,
            "model_name": self.model_name,
            "tokenizer_json": tok_path,
            "sample_rate": 24000,
            "detok_overlap": 0,
            "max_tokens": self.model.max_tokens,
            "decode_mask": [1] * self.depth_n + [0],
            "decode_row": [0] * self.depth_n + [self.tts_pad],
            # Input streaming: the text column of decode rows carries the next streamed text token.
            "text_stream": {"column": self.n_codebooks - 1, "pad": cfg.tts_pad_token_id, "eos": cfg.tts_eos_token_id},
            "tts_pad": cfg.tts_pad_token_id,
            "tts_bos": cfg.tts_bos_token_id,
            "tts_eos": cfg.tts_eos_token_id,
            "codec_pad": t.codec_pad_id,
            "codec_bos": t.codec_bos_id,
            "codec_think": t.codec_think_id,
            "codec_nothink": t.codec_nothink_id,
            "codec_think_bos": t.codec_think_bos_id,
            "codec_think_eos": t.codec_think_eos_id,
            "languages": dict(t.codec_language_id),
            "speakers": dict(t.spk_id),
            "speaker_dialect": {k: v for k, v in t.spk_is_dialect.items() if v},
            "default_speaker": "ryan",
            "default_language": "english",
            "supports_instruct": self.model.tts_model_size not in "0b6",
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
            },
        }
