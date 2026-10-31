"""Chatterbox (T3 Llama-520M + S3Gen, Perth watermark) graph program.

The prompt embedding mixes four row kinds, selected per row by the input mask (column 0):
  0 conditioning row ``tok`` of the (constant, default-voice) ``cond_enc`` output,
  1 text row: ``text_emb(tok) + text_pos_emb(pos - cond_len)``,
  2 speech-start row: ``speech_emb(tok) + speech_pos_emb(0)``,
  3 generated speech: ``speech_emb(tok) + speech_pos_emb(clamp(pos - 45))`` (decode rows).
The Perth watermark is applied to every detokenized chunk inside the detok graph.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from ...rt.core import Arena, RtAttention
from ...rt.program import LMProgram
from ...rt.prompt import Prompt
from ...rt.watermark import istft
from .model import ChatterboxModel

COND, TEXT, SPEECH_START, SPEECH = 0, 1, 2, 3
# Speech position offset hard-coded by the reference implementation (see ChatterboxModel.forward).
SPEECH_POS_OFFSET = 45


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


class ChatterboxProgram(LMProgram):
    uses_input_mask = True

    def load(self) -> None:
        name = self.config.get("model", "ResembleAI/chatterbox")
        if name == "chatterbox":
            name = "ResembleAI/chatterbox"
        self.model_name = name
        self.model = ChatterboxModel(name, device=str(self.device), audio_decoder_device=str(self.detok_device))
        m = self.model.model
        cfg = m.config
        self.n_codebooks = 1
        self.vocab_size = cfg.speech_tokens_dict_size
        self.num_layers = self.model.num_hidden_layers
        self.num_qo_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.detok_interval = self.model.detokenize_interval
        self.detok_samples = self.model.output_audio_length

        with torch.no_grad():
            conds = self.model.default_conds.t3.to(dtype=self.dtype)
            if conds.cond_prompt_speech_tokens is not None and conds.cond_prompt_speech_emb is None:
                n = conds.cond_prompt_speech_tokens.shape[1]
                conds.cond_prompt_speech_emb = m.speech_emb(conds.cond_prompt_speech_tokens) + m.speech_pos_emb(
                    torch.arange(0, n, device=self.device)
                )
            self.cond_emb = m.cond_enc(conds)[0].contiguous()  # (cond_len, H)
        self.cond_len = self.cond_emb.shape[0]
        self.watermark = bool(self.config.get("watermark", True))
        self.perth = PerthBatched(24000, self.detok_device) if self.watermark else None

    def embed(self, ids: torch.Tensor, arena: Arena, rows: int) -> torch.Tensor:
        m = self.model.model
        cfg = m.config
        kind = arena["mask"][:rows, :1]
        tok = ids[:, 0]
        pos = arena["pos"][:rows].long()
        cond = self.cond_emb[tok.clamp(0, self.cond_len - 1)]
        text = m.text_emb(tok.clamp(0, cfg.text_tokens_dict_size - 1)) + m.text_pos_emb(
            (pos - self.cond_len).clamp(0, cfg.max_text_tokens + 1)
        )
        speech_tok = m.speech_emb(tok.clamp(0, cfg.speech_tokens_dict_size - 1))
        start = speech_tok + m.speech_pos_emb(torch.zeros_like(pos))
        speech = speech_tok + m.speech_pos_emb((pos - SPEECH_POS_OFFSET).clamp(0, cfg.max_speech_tokens))
        h = torch.where(kind == COND, cond, text)
        h = torch.where(kind == SPEECH_START, start, h)
        return torch.where(kind == SPEECH, speech, h)

    def fuse_lm_modules(self):
        return [self.model.model.tfmr]

    def backbone(self, hidden: torch.Tensor, pos: torch.Tensor, attn: RtAttention) -> torch.Tensor:
        return self.model.model.tfmr(inputs_embeds=hidden, position_ids=pos, attn_wrapper=attn, kv_cache=self.kv_cache)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.model.speech_head(hidden)[:, None, :]

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        audio = self.model.postprocess(tokens).reshape(tokens.shape[0], -1)
        if self.perth is not None:
            audio = self.perth(audio)
        return audio

    def stop_tokens(self) -> List[int]:
        return [self.model.model.config.stop_speech_token]

    def preprocess(self, text: str, audio_path: Optional[str], model_kwargs: Dict[str, Any]) -> Prompt:
        """``[cond rows][START text STOP][START_SPEECH x2]``; the mask carries the row kind."""
        if audio_path is not None:
            raise ValueError("Audio conditioning not implemented yet.")
        cfg = self.model.model.config
        tokens = list(range(self.cond_len))
        mask = [COND] * self.cond_len
        tokens += [cfg.start_text_token] + self._encode(punc_norm(text).replace(" ", "[SPACE]")) + [cfg.stop_text_token]
        mask += [TEXT] * (len(tokens) - len(mask))
        tokens += [cfg.start_speech_token] * 2
        mask += [SPEECH_START] * 2
        return Prompt(tokens, mask)

    def audio_tokens(self) -> Optional[Tuple[int, int]]:
        return (0, self.model.speech_vocab_size)

    def meta(self) -> Dict[str, Any]:
        sc = self.model.default_sampling_config
        cfg = self.model.model.config
        return {
            "kind": "chatterbox",
            "model_name": self.model_name,
            "tokenizer_json": hf_hub_download(self.model_name, "tokenizer.json"),
            "sample_rate": 24000,
            "detok_overlap": self.model.detokenize_overlap,
            "max_tokens": self.model.max_tokens,
            "cond_len": self.cond_len,
            "start_text": cfg.start_text_token,
            "stop_text": cfg.stop_text_token,
            "start_speech": cfg.start_speech_token,
            "row_kinds": {"cond": COND, "text": TEXT, "speech_start": SPEECH_START},
            "decode_mask": [SPEECH],
            "watermark": "perth" if self.watermark else None,
            "sampling": {
                "temperature": sc.temperature,
                "top_k": sc.top_k or 0,
                "top_p": sc.top_p if sc.top_p is not None else 1.0,
                "min_p": sc.min_p or 0.0,
                "repetition_penalty": sc.repetition_penalty or 1.0,
            },
        }
