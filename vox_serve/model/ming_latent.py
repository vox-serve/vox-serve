from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ..utils import ensure_ming_code_available, get_logger, materialize_ming_code
from .base import BaseLatentLM


@dataclass
class _PromptBundle:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    latent: torch.Tensor


class MingUniAudioNative(BaseLatentLM):
    """
    Thin adapter around Ming-UniAudio's native implementation.

    The integration mirrors `BailingMMNativeForConditionalGeneration.sample_tokens`
    to obtain latent trajectories and reuses the bundled VAE decoder for waveform
    reconstruction.
    """

    IS_LATENT_MODEL: bool = True
    generates_latent: bool = True

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flashinfer: bool = False,
        page_size: int = 128,
        enable_torch_compile: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            enable_torch_compile=enable_torch_compile,
        )
        self.logger = get_logger(__name__)
        self.page_size = page_size
        self.use_flashinfer_requested = use_flashinfer
        self.use_flashinfer_active = False

        resolved_model_name = self._normalize_model_name(model_name)
        model_path = Path(resolved_model_name)
        if model_path.exists():
            self._model_dir = model_path.resolve()
        else:
            try:
                self._model_dir = Path(
                    snapshot_download(
                        repo_id=resolved_model_name,
                    )
                ).resolve()
            except Exception as exc:  # pragma: no cover - hub error
                raise RuntimeError(
                    f"Unable to load Ming from '{model_name}'. Pass a local path or HF repo id."
                ) from exc

        code_dir = ensure_ming_code_available(self._model_dir)
        materialize_ming_code(self._model_dir, code_dir)

        bailing_module = import_module("modeling_bailingmm")
        self._bailing_model_cls = getattr(
            bailing_module,
            "BailingMMNativeForConditionalGeneration",
        )

        self._resolved_model_name = str(self._model_dir)

        self._model = self._load_model(self._resolved_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._resolved_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.tokenizer = self._tokenizer

        enc_kwargs = getattr(self._model.config.audio_tokenizer_config, "enc_kwargs", {})
        self.latent_dim = int(enc_kwargs.get("latent_dim", 1024))
        self.fps = int(enc_kwargs.get("frame_rate", 50))
        self.sample_rate = int(getattr(self._model.config.audio_tokenizer_config, "sample_rate", 16_000))

        if self.use_flashinfer_requested:
            self.logger.warning("FlashInfer integration is not yet available; falling back to SDPA.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if model_name.lower() == "ming":
            return "inclusionAI/Ming-UniAudio-16B-A3B"
        return model_name

    def _load_model(self, model_name: str):
        try:
            model = (
                self._bailing_model_cls.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                .to(self.device)
                .eval()
            )
        except Exception as exc:  # pragma: no cover - execution-time validation
            raise RuntimeError(
                f"Unable to load Ming from '{model_name}'. Pass a local path or HF repo id."
            ) from exc

        torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False,
        )
        return model

    def _load_prompt(self, prompt_wav: str) -> _PromptBundle:
        waveform, sr = torchaudio.load(prompt_wav)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.to(self.device)
        length = torch.tensor([waveform.shape[-1]], dtype=torch.int, device=self.device)

        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
            latent, _ = self._model.audio.encode_latent(waveform, length)

        return _PromptBundle(waveform, length, latent.to(self.device))

    # ------------------------------------------------------------------
    # BaseLatentLM interface
    # ------------------------------------------------------------------
    def supports_audio_input(self) -> bool:
        return True

    @torch.inference_mode()
    def generate_latents(
        self,
        text: str,
        *,
        prompt_wav: str,
        language: str = "en",
        patch_size: int = 5,
        cfg: float = 2.0,
        max_frames: Optional[int] = None,
        progress: bool = False,
        prompt_text: str = "",
        **_,
    ) -> torch.Tensor:
        prompt = self._load_prompt(prompt_wav)

        prompt_text_full = f"Please translate the text to speech.\n{prompt_text}".strip()
        kwargs = {
            "z_prompt_vae_latent": prompt.latent,
            "prompt_text": prompt_text_full,
            "patch_size": patch_size,
        }

        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
            sampled_tokens = self._model.sample_tokens(
                text=text,
                lang=language,
                cfg=cfg,
                progress=progress,
                **kwargs,
            )

        if not sampled_tokens:
            raise RuntimeError("Ming sample_tokens returned no latent tokens.")

        latents = torch.cat(sampled_tokens, dim=1).to(torch.float32)
        if max_frames is not None and latents.shape[1] > max_frames:
            latents = latents[:, :max_frames, :]

        self.logger.info(
            "Ming latents generated: text_len=%d, frames=%d, latent_dim=%d",
            len(text),
            latents.shape[1],
            latents.shape[2],
        )
        return latents

    @torch.inference_mode()
    def decode_latents_to_audio(
        self,
        latents: torch.Tensor,
        *,
        target_sr: int = 24_000,
        **_,
    ) -> torch.Tensor:
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
            speech = self._model.audio.decode(latents.to(self.device))

        if speech.dim() == 2:
            speech = speech.unsqueeze(0)

        speech = speech.to(torch.float32)
        if target_sr != self.sample_rate:
            speech = torchaudio.functional.resample(
                speech,
                orig_freq=self.sample_rate,
                new_freq=target_sr,
            )

        max_val = speech.abs().max()
        if torch.isnan(max_val) or max_val <= 0:
            self.logger.warning("Decoded audio is near-silent (max=%s)", max_val)
        else:
            speech = speech / max_val.clamp(min=1e-6)

        self.logger.info(
            "Ming audio decoded: frames=%d, sample_rate=%d",
            speech.shape[-1],
            target_sr,
        )
        return speech
