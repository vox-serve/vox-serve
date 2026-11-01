"""Chatterbox (T3 Llama-520M + S3Gen, Perth watermark) graph program, with zero-shot voice cloning.

Rows are ``tok = [id, 0]`` with mask ``[kind, voice slot]``; the kind selects the embedding:
  0 conditioning row ``id`` of the voice's ``cond_enc`` output (speaker, perceiver prompt, emotion),
  1 text row: ``text_emb(id) + text_pos_emb(pos - cond_len)``,
  2 speech-start row: ``speech_emb(id) + speech_pos_emb(0)``,
  3 generated speech (decode rows): ``speech_emb(id) + speech_pos_emb(i + 1)`` for the i-th generated
    token (0-based), as the reference's ``T3.inference``; the program counts each slot's tokens.
Output rows are ``[token, voice slot]``, so the detokenizer knows each window's voice.

Voice slot 0 holds the default voice (``conds.pt``); reference clips are encoded into the others on
demand (``hook_encode_voice``): T3 conditioning from the clip (voice-encoder embedding, the first 6 s as
speech-token prompt) and S3Gen's prompt caches from its first 10 s (``ChatterboxModel.prepare_conditionals``).
S3Gen is CosyVoice2's flow + HiFT decoder (``FlowHiFTDetok``). The Perth watermark is applied to every
detokenized chunk inside the detok graph.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from ...encoder.chatterbox import T3Cond
from ...rt.core import Arena, EngineSpecBuilder, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt, VoiceSlots
from ...rt.watermark import istft
from ...tokenizer.cosyvoice2 import PromptCacheTable
from ..cosyvoice2.program import FlowHiFTDetok
from .model import ChatterboxModel

COND, TEXT, SPEECH_START, SPEECH = 0, 1, 2, 3


def punc_norm(text: str) -> str:
    """``ChatterboxModel._punc_norm`` (also in rust/src/models/chatterbox.rs)."""
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith((".", "!", "?", "-", ",")):
        text += "."
    return text


class PerthBatched(torch.nn.Module):
    """Batched, graph-capturable equivalent of ``PerthImplicitWatermarker.encode_wav``."""

    def __init__(self, sr: int, device: torch.device):
        super().__init__()
        from ...watermarker.perth import PerthImplicitWatermarker, magphase_to_cx

        wm = PerthImplicitWatermarker(device=device)
        self.net = wm.perth_net
        self.to_cx = magphase_to_cx
        self.up = torchaudio.transforms.Resample(sr, wm.sr).to(device)
        self.down = torchaudio.transforms.Resample(wm.sr, sr).to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        ap = self.net.ap
        x = self.up(audio.float())
        mag, phase = ap.signal_to_magphase(x)
        wm_mag, _ = self.net.encoder(mag)
        inv = ap.inv_spectrogram
        y = istft(self.to_cx(ap.hp, wm_mag, phase), inv.n_fft, inv.hop_length, inv.window)
        return self.down(y)


class ChatterboxProgram(FlowHiFTDetok, LMProgram):
    uses_input_mask = True
    fb_columns = 1  # column 1 of fed-back rows (the voice) is not read

    def load(self) -> None:
        name = self.config.get("model", "ResembleAI/chatterbox")
        if name == "chatterbox":
            name = "ResembleAI/chatterbox"
        self.model_name = name
        self.model = ChatterboxModel(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        m = self.model.model
        cfg = m.config
        self.n_codebooks = 2
        self.vocab_size = cfg.speech_tokens_dict_size
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_overlap = self.model.detokenize_overlap
        self.detok_samples = self.model.output_audio_length + 1  # + the audio length
        self.decoder = self.model.audio_decoder
        self.n_voices = int(self.config.get("n_voices", 4))

        default = self.model.default_conds
        cond = self._cond_emb(default.t3)
        self.cond_len = cond.shape[0]
        self.voice_cond = cond.new_zeros(self.n_voices, *cond.shape)
        self.voice_cond[0] = cond
        cache, embedding = self._prompt_cache(default.gen)
        self.prompts = PromptCacheTable(cache, embedding, self.n_voices)
        self.prompts.write(0, cache, embedding)
        self.watermark = bool(self.config.get("watermark", True))
        self.perth = PerthBatched(24000, self.detok_device) if self.watermark else None

    # ---- voices ---------------------------------------------------------------------------------

    @torch.no_grad()
    def _cond_emb(self, t3: T3Cond) -> torch.Tensor:
        """The conditioning rows (cond_len, H) of a voice: ``T3.prepare_conditioning``."""
        m = self.model.model
        t3 = t3.to(device=self.device, dtype=self.dtype)
        if t3.cond_prompt_speech_tokens is not None and t3.cond_prompt_speech_emb is None:
            n = t3.cond_prompt_speech_tokens.shape[1]
            t3.cond_prompt_speech_emb = m.speech_emb(t3.cond_prompt_speech_tokens) + m.speech_pos_emb(
                torch.arange(0, n, device=self.device)
            )
        return m.cond_enc(t3)[0].contiguous()

    def _prompt_cache(self, gen: Dict[str, Any]):
        """S3Gen's flow prompt caches and x-vector of a voice's reference dict (``embed_ref``)."""
        dd = self.detok_device
        tok = gen["prompt_token"].to(dd).long()
        n = min(tok.shape[1], gen["prompt_feat"].shape[1] // 2)
        embedding = gen["embedding"].to(dd, torch.bfloat16).reshape(1, -1)
        ref = {
            "prompt_speech_token": tok[:, :n],
            "prompt_speech_token_len": n,
            "prompt_feat": gen["prompt_feat"][:, : 2 * n].to(dd, torch.bfloat16),
            "prompt_feat_len": 2 * n,
            "embedding": embedding,
        }
        with torch.inference_mode():
            cache = self.decoder.init_cache(ref)
        return cache, embedding

    @torch.inference_mode()
    def hook_encode_voice(self, path: str, slot: int) -> Dict[str, Any]:
        """Encodes reference clip ``path`` into voice slot ``slot`` (T3 conditioning and S3Gen prompt caches),
        which no running request uses."""
        if not 0 < slot < self.n_voices:
            raise ValueError(f"voice slot {slot} out of range")
        conds = self.model.prepare_conditionals(path)
        cond = self._cond_emb(conds.t3)
        cache, embedding = self._prompt_cache(conds.gen)
        self.prompts.write(slot, cache, embedding)
        self.voice_cond[slot].copy_(cond)
        # The engine streams read the slot without synchronizing with this (hook) stream.
        torch.cuda.current_stream(self.device).synchronize()
        torch.cuda.current_stream(self.detok_device).synchronize()
        return {"slot": slot}

    # ---- LM -------------------------------------------------------------------------------------

    def allocate_state(self) -> None:
        super().allocate_state()
        # Per LM slot: the request's voice and how many tokens it has generated.
        self.req_voice = torch.zeros(self.max_slots + 1, dtype=torch.int32, device=self.device)
        self.req_ntok = torch.zeros(self.max_slots + 1, dtype=torch.int32, device=self.device)
        self._allocate_flow_hift()

    def declare(self, builder: EngineSpecBuilder) -> None:
        super().declare(builder)
        self._declare_flow_hift(builder)

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model.model
        cfg = m.config
        mask = arena["mask"][:rows]
        kind = mask[:, :1]
        voice = mask[:, 1].long().clamp(0, self.n_voices - 1)
        tok = ids[:, 0]
        pos = arena["pos"][:rows].long()
        cond = self.voice_cond[voice, tok.clamp(0, self.cond_len - 1)]
        text = m.text_emb(tok.clamp(0, cfg.text_tokens_dict_size - 1)) + m.text_pos_emb(
            (pos - self.cond_len).clamp(0, cfg.max_text_tokens + 1)
        )
        speech_tok = m.speech_emb(tok.clamp(0, cfg.speech_tokens_dict_size - 1))
        start = speech_tok + m.speech_pos_emb(torch.zeros_like(pos))
        # A decode row feeds back its request's latest token: the 1-based index of that generated token.
        speech_pos = self.req_ntok[arena["row_slot"][:rows].long()].long()
        speech = speech_tok + m.speech_pos_emb(speech_pos.clamp(0, cfg.max_speech_tokens))
        h = torch.where(kind == COND, cond, text)
        h = torch.where(kind == SPEECH_START, start, h)
        return torch.where(kind == SPEECH, speech, h)

    def fuse_lm_modules(self):
        return [self.model.model.tfmr]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.tfmr(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.model.speech_head(hidden)[:, None, :]

    def decode_tokens(self, hidden: torch.Tensor, arena: Arena, out_tok: torch.Tensor, n_req: int) -> None:
        self.sample_into(self.head(hidden), arena, out_tok, n_req, self.LM_SALT, penalty=self.uses_repetition_penalty)
        slot = arena["slot"][:n_req].long()
        first = arena["first"][:n_req] != 0
        last_row = arena["last_row"][:n_req].long()
        voice = torch.where(first, arena["mask"][:, 1][last_row], self.req_voice[slot])
        self.req_voice.index_copy_(0, slot, voice)
        self.req_ntok.index_copy_(0, slot, torch.where(first, 1, self.req_ntok[slot] + 1))
        out_tok[:, 1] = voice

    def stop_tokens(self) -> List[int]:
        return [self.model.model.config.stop_speech_token]

    def audio_tokens(self) -> Optional[Tuple[int, int]]:
        return (0, self.model.speech_vocab_size)

    # ---- detok ----------------------------------------------------------------------------------

    def detok_step(self, arena: Arena, out: Arena, bd: int, lane: int = 0) -> None:
        toks = arena["dt_tok"][:bd].long()
        tok = toks[:, :, 0].clamp(0, self.model.speech_vocab_size - 1)
        voice = toks[:, :, 1].amax(1).clamp(0, self.n_voices - 1)
        self._flow_hift_step(arena, out, bd, tok, voice)

    def _postprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if self.perth is None:
            return audio
        n = audio.shape[1]
        wm = self.perth(audio)[:, :n]  # the STFT round trip can drop a partial hop at the end
        return torch.nn.functional.pad(wm, (0, n - wm.shape[1]))

    # ---- prompt ---------------------------------------------------------------------------------

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """``[cond rows][START text STOP][START_SPEECH x2]`` with mask ``[kind, voice slot]``: the default
        voice, or reference audio encoded into a voice slot."""
        voices = self._voices()
        if audio_path is not None:
            voice = voices.get(audio_path, lambda slot: self.hook_encode_voice(audio_path, slot))
        else:
            voice = voices.default()
        m = self.spec_meta
        ids = list(range(m["cond_len"]))
        kinds = [COND] * m["cond_len"]
        ids += [m["start_text"]] + self._encode(punc_norm(text).replace(" ", "[SPACE]")) + [m["stop_text"]]
        kinds += [TEXT] * (len(ids) - len(kinds))
        ids += [m["start_speech"]] * 2
        kinds += [SPEECH_START] * 2
        tokens = [x for i in ids for x in (i, 0)]
        mask = [x for k in kinds for x in (k, voice.slot)]
        return Prompt(tokens, mask, hold=voices.hold(voice))

    _voice_slots: Optional[VoiceSlots] = None

    def _voices(self) -> VoiceSlots:
        if self._voice_slots is None:
            self._voice_slots = VoiceSlots(int(self.spec_meta.get("n_voices", 1)))
        return self._voice_slots

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        cfg = self.model.model.config
        return {
            "kind": "chatterbox",
            "model_name": self.model_name,
            "tokenizer_json": hf_hub_download(self.model_name, "tokenizer.json"),
            "sample_rate": 24000,
            "detok_overlap": self.detok_overlap,
            "max_tokens": self.model.max_tokens,
            "cond_len": self.cond_len,
            "start_text": cfg.start_text_token,
            "stop_text": cfg.stop_text_token,
            "start_speech": cfg.start_speech_token,
            "row_kinds": {"cond": COND, "text": TEXT, "speech_start": SPEECH_START},
            "decode_mask": [SPEECH, 0],
            "n_voices": self.n_voices,
            "audio_len_in_row": True,
            "watermark": "perth" if self.watermark else None,
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
            },
        }
