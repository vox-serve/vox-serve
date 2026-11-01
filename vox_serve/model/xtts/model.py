"""XTTS-v2 (``coqui/XTTS-v2``): modules, weights, voice conditioning and constants.

A GPT-2 (30 layers, 1024 wide, learned position embeddings added outside the stack) generates DVAE codes
(1024 + start / stop) after a prompt of 32 conditioning latents (a perceiver resampler over the reference
clip's mel spectrogram) and the BPE text tokens. The HiFi-GAN vocoder decodes the GPT's final hidden states
(the *latents*, one per generated code, 1024 samples at 22.05 kHz each) to 24 kHz audio, conditioned on a
512-d speaker embedding. The checkpoint ships 58 named voices (``speakers_xtts.pth``) as precomputed
conditioning latents and speaker embeddings.

Adapted from the XTTS-v2 integration of PR #50 and Coqui TTS (https://github.com/coqui-ai/TTS,
``TTS/tts/models/xtts.py`` and ``TTS/tts/layers/xtts/gpt.py``; MPL-2.0). The model weights are under the
Coqui Public Model License (non-commercial use only).
"""

import importlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import types
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from ...encoder.xtts import ConditioningEncoder, HifiDecoder, PerceiverResampler
from ...sampling import SamplingConfig
from ...tokenizer.xtts import VoiceBpeTokenizer
from ..base import BaseLM

REPO = "coqui/XTTS-v2"
FILES = ["config.json", "vocab.json", "model.pth", "speakers_xtts.pth"]
COND_LATENTS = 32  # perceiver resampler outputs: the conditioning rows of every prompt
GPT_SR = 22050  # sample rate of the GPT's codes (and of the conditioning mel spectrograms)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.emb(pos)


class XttsGptAttention(nn.Module):
    """GPT-2 attention over the paged KV cache (no rotary embedding: positions are learned embeddings
    added to the inputs)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attn, kv_cache: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = self.c_attn(x).view(n, 3, self.num_heads, self.head_dim).unbind(1)
        attn.set_kv_cache(kv_cache, k, v)
        return self.c_proj(attn.run(q.contiguous(), kv_cache).reshape(n, -1))


class XttsGptMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size)
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class XttsGptBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = XttsGptAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = XttsGptMLP(hidden_size)

    def forward(self, x: torch.Tensor, attn, kv_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn, kv_cache)
        return x + self.mlp(self.ln_2(x))


class XttsGpt(nn.Module):
    """The GPT-2 stack (``h`` + ``ln_f``) in Hugging Face naming."""

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.h = nn.ModuleList([XttsGptBlock(hidden_size, num_heads, eps) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor, attn, kv_cache: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.h):
            x = block(x, attn, kv_cache[i])
        return self.ln_f(x)


class XTTSModel(BaseLM, nn.Module):
    """The XTTS-v2 modules: GPT (``float16``: FlashInfer has no fp32 attention, and bf16 drifts audibly over
    30 layers) and, in fp32, the conditioning encoder / perceiver (their attention overflows in fp16), the
    speaker encoder and the HiFi-GAN vocoder."""

    def __init__(self, model_name: str = REPO, device: str = "cuda", audio_decoder_device: str = None):
        nn.Module.__init__(self)
        BaseLM.__init__(self, model_name, device, torch.float16, False, audio_decoder_device)
        from huggingface_hub import snapshot_download

        repo = model_name if "/" in model_name else REPO
        self.model_dir = snapshot_download(repo, allow_patterns=FILES)
        cfg = json.load(open(os.path.join(self.model_dir, "config.json")))
        a = SimpleNamespace(**cfg["model_args"])
        self.args = a
        self.languages = list(cfg["languages"])
        self.vocab_path = os.path.join(self.model_dir, "vocab.json")
        self.tokenizer = VoiceBpeTokenizer(vocab_file=self.vocab_path)
        tok = self.tokenizer.tokenizer
        self.start_text_token = tok.token_to_id("[START]")
        self.stop_text_token = tok.token_to_id("[STOP]")
        self.num_text_tokens = self.tokenizer.get_number_tokens()
        self.num_audio_tokens = a.gpt_num_audio_tokens
        self.start_audio_token = a.gpt_start_audio_token
        self.stop_audio_token = a.gpt_stop_audio_token
        # Coqui's GPT: positions for the text (+ start / stop) and the codes (+ start / stop / conditioning).
        self.max_text_positions = a.gpt_max_text_tokens + 2
        self.max_mel_positions = a.gpt_max_audio_tokens + 2 + 1
        self.max_gen_tokens = a.gpt_max_audio_tokens - 1 - 2  # Coqui's max_gen_mel_tokens
        self.hop = a.gpt_code_stride_len  # 22.05 kHz samples per code (latent)
        self.mel_hop = a.output_hop_length  # vocoder samples per mel frame
        self.input_sr, self.output_sr = a.input_sample_rate, a.output_sample_rate

        H = a.gpt_n_model_channels
        self.conditioning_encoder = ConditioningEncoder(80, H, num_attn_heads=a.gpt_n_heads)
        self.conditioning_perceiver = PerceiverResampler(
            dim=H, depth=2, dim_context=H, num_latents=COND_LATENTS, dim_head=64, heads=8, ff_mult=4
        )
        self.text_embedding = nn.Embedding(self.num_text_tokens, H)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, H)
        self.gpt = XttsGpt(a.gpt_layers, H, a.gpt_n_heads)
        self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_positions, H)
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_positions, H)
        self.final_norm = nn.LayerNorm(H)
        self.mel_head = nn.Linear(H, self.num_audio_tokens)
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=a.input_sample_rate,
            output_sample_rate=a.output_sample_rate,
            output_hop_length=a.output_hop_length,
            ar_mel_length_compression=a.gpt_code_stride_len,
            decoder_input_dim=a.decoder_input_dim,
            d_vector_dim=a.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=a.cond_d_vector_in_each_upsampling_layer,
        )
        self.register_buffer("mel_stats", torch.ones(80))
        self.load_state_dict(self._checkpoint(os.path.join(self.model_dir, "model.pth")), strict=True)
        # The vocoder's weight norm, folded once (Coqui recomputes it every call).
        dec = self.hifigan_decoder.waveform_decoder
        for m in list(dec.ups) + [c for rb in dec.resblocks for c in rb.modules() if isinstance(c, nn.Conv1d)]:
            if hasattr(m, "parametrizations"):
                torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
        self.eval().requires_grad_(False)

        self.to(self.device)
        for m in (self.gpt, self.text_embedding, self.mel_embedding, self.text_pos_embedding, self.mel_pos_embedding):
            m.to(self.dtype)
        for m in (self.final_norm, self.mel_head):
            m.to(self.dtype)
        self.hifigan_decoder.to(self.audio_decoder_device)

        self.voices = self._builtin_voices(os.path.join(self.model_dir, "speakers_xtts.pth"))
        # Coqui's ``inference`` / ``inference_stream`` defaults (its config.json says repetition_penalty 5.0,
        # used by the high-level ``TTS.tts`` API; 10.0 is what the model's own inference methods use).
        self.default_sampling_config = SamplingConfig(
            temperature=0.75, top_k=50, top_p=0.85, min_p=0.0, repetition_penalty=10.0, max_tokens=None
        )

    # ---- BaseLM constants ---------------------------------------------------------------------------

    @property
    def n_codebooks(self) -> int:
        return 1

    @property
    def num_attention_heads(self) -> int:
        return self.args.gpt_n_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.args.gpt_n_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.args.gpt_layers

    @property
    def hidden_size(self) -> int:
        return self.args.gpt_n_model_channels

    @property
    def vocab_size(self) -> int:
        return self.num_audio_tokens

    @property
    def detokenize_interval(self) -> int:
        raise NotImplementedError("the program sets the chunk geometry")

    @property
    def detokenize_overlap(self) -> int:
        raise NotImplementedError("the program sets the chunk geometry")

    @property
    def max_tokens(self) -> int:
        return self.max_gen_tokens

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def output_audio_length(self) -> int:
        raise NotImplementedError("the program sets the chunk geometry")

    # ---- voice conditioning ---------------------------------------------------------------------------

    def _builtin_voices(self, path: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Name -> (conditioning latents [32, 1024], speaker embedding [512]), fp32 on the CPU."""
        speakers = torch.load(path, map_location="cpu", weights_only=False)
        return {
            name: (e["gpt_cond_latent"].reshape(COND_LATENTS, -1).float(), e["speaker_embedding"].reshape(-1).float())
            for name, e in speakers.items()
        }

    def _cloning_mel(self, wav: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        mel = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            power=2,
            normalized=False,
            sample_rate=GPT_SR,
            f_min=0,
            f_max=8000,
            n_mels=80,
            norm="slaney",
        ).to(wav.device)(wav)
        return torch.log(torch.clamp(mel, min=1e-5)) / self.mel_stats[None, :, None]

    @torch.inference_mode()
    def conditioning(
        self, wav: torch.Tensor, gpt_cond_len: int = 6, chunk_len: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A reference clip ``[1, T]`` (mono, 22.05 kHz, at most ``max_ref_length`` s) -> (conditioning latents
        [32, 1024], speaker embedding [512]), as Coqui's ``get_conditioning_latents`` with its defaults: the
        latents average the perceiver outputs over ``chunk_len``-second chunks of the first ``gpt_cond_len``
        seconds; the speaker embedding takes the whole clip at 16 kHz."""
        wav = wav.to(self.device, torch.float32)
        spk_enc = self.hifigan_decoder.speaker_encoder
        wav16 = torchaudio.functional.resample(wav, GPT_SR, 16000).to(self.audio_decoder_device)
        spk = spk_enc(wav16, l2_norm=True).reshape(-1)
        cond = wav[:, : GPT_SR * gpt_cond_len]
        latents = []
        for i in range(0, cond.shape[1], GPT_SR * chunk_len):
            chunk = cond[:, i : i + GPT_SR * chunk_len]
            if chunk.shape[1] < int(GPT_SR * 0.33):
                continue
            mel = self._cloning_mel(chunk, n_fft=2048, hop=256, win=1024)
            h = self.conditioning_encoder(mel)  # [1, 1024, T]
            latents.append(self.conditioning_perceiver(h.permute(0, 2, 1)))  # [1, 32, 1024]
        if not latents:
            raise ValueError("reference audio is too short (at least 0.33 s)")
        return torch.stack(latents).mean(0)[0].float().cpu(), spk.float().cpu()

    # ---- checkpoint -----------------------------------------------------------------------------------

    def _checkpoint(self, path: str) -> Dict[str, torch.Tensor]:
        """Coqui's ``model.pth`` state dict in this module's naming: the GPT's sub-modules move up a level,
        the GPT-2 stack drops a ``gpt.``, its ``Conv1D`` weights are transposed to ``nn.Linear``. The text head,
        the DVAE and the training mel extractors are not used for inference."""
        ckpt = _load_with_tts_stubs(path)
        ckpt = ckpt.get("model", ckpt)
        state = {}
        for key, value in ckpt.items():
            key = key.removeprefix("xtts.")
            if key.split(".")[0] in ("dvae", "torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae"):
                continue
            if key.startswith("gpt.text_head."):
                continue
            if key.startswith("gpt.gpt."):
                key = key[len("gpt.") :]
                if key.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
                    value = value.t().contiguous()
            elif key.startswith("gpt."):
                key = key[len("gpt.") :]
            state[key] = value
        return state


def _load_with_tts_stubs(path: str):
    """``model.pth`` pickles Coqui ``TTS.*`` config classes next to the weights: stub modules let it unpickle
    without the TTS package (the weights are plain tensors). The stubs are removed afterwards, so they never
    shadow an installed TTS package."""
    finder = _TTSStubFinder()
    saved = {k: v for k, v in sys.modules.items() if k == "TTS" or k.startswith("TTS.")}
    for k in saved:
        del sys.modules[k]
    sys.meta_path.insert(0, finder)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k == "TTS" or k.startswith("TTS.")]:
            del sys.modules[k]
        sys.modules.update(saved)


class _Stub:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


class _TTSStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "TTS" or name.startswith("TTS."):
            spec = importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
            spec.submodule_search_locations = []
            return spec
        return None
