"""Graph-capturable, batched watermark encoders for detokenizer outputs."""

import torch
import torchaudio

# Fixed message (5 bytes) that SilentCipher embeds in every watermarked chunk.
SILENTCIPHER_KEY = [11, 91, 60, 147, 209]


def istft(spec: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """``torch.istft(spec, n_fft, hop, window=window, center=True)`` for ``spec`` [B, F, T] without
    its NOLA check, whose ``.item()`` sync is not allowed during graph capture."""
    w = window.float()
    if w.shape[0] < n_fft:
        left = (n_fft - w.shape[0]) // 2
        w = torch.nn.functional.pad(w, (left, n_fft - w.shape[0] - left))
    frames = torch.fft.irfft(spec, n=n_fft, dim=1) * w[None, :, None]
    T = frames.shape[-1]
    n = n_fft + hop * (T - 1)
    fold = lambda x: torch.nn.functional.fold(x, (1, n), (1, n_fft), stride=(1, hop))[:, 0, 0]  # noqa: E731
    y = fold(frames)
    env = fold((w * w)[None, :, None].expand(1, n_fft, T).contiguous())
    lo, hi = n_fft // 2, n - n_fft // 2
    return y[:, lo:hi] / env[:, lo:hi].clamp_min(1e-11)


class SilentCipherBatched(torch.nn.Module):
    """Batched SilentCipher watermarking with the 44.1 kHz model and ``SILENTCIPHER_KEY``.

    The reference processes one chunk at a time: resample it to 44.1 kHz, encode it with
    ``Model.encode_wav`` (``vox_serve/watermarker/silentcipher.py``, message SDR 36 dB) and resample
    it back to ``sr``. Here every chunk has the same length (``n_samples`` at ``sr``), so the message
    carrier, STFT sizes and resampling are fixed at construction. Silent chunks (zero power) are only
    resampled to 44.1 kHz and back, as in the reference, which skips encoding them.
    """

    def __init__(self, sr: int, n_samples: int, device: torch.device, key=SILENTCIPHER_KEY, message_sdr: float = 36):
        super().__init__()
        from ..watermarker import silentcipher

        wm = silentcipher.get_model(model_type="44.1k", device=device)
        cfg = wm.config
        self.cfg = cfg
        self.enc_c, self.dec_c = wm.enc_c, wm.dec_c
        self.message_sdr = message_sdr
        self.avg_energy = float(wm.average_energy_VCTK)
        self.n_fft, self.hop = cfg.N_FFT, cfg.HOP_LENGTH
        self.n_samples = n_samples
        self.up = torchaudio.transforms.Resample(sr, wm.sr).to(device)
        self.down = torchaudio.transforms.Resample(wm.sr, sr).to(device)
        self.register_buffer("window", torch.hann_window(self.n_fft, device=device))

        with torch.no_grad():
            probe = self.up(torch.zeros(1, n_samples, device=device))
            self.n_wm = probe.shape[-1]
            if self.down(probe).shape[-1] < n_samples:
                raise ValueError(f"SilentCipher resampling round trip shortens {n_samples}-sample chunks")
            self.pad = self.n_fft - self.n_wm % self.n_fft
            frames = (self.n_wm + self.pad) // self.hop + 1
            bits = "".join(f"{b:08b}" for b in key)
            symbols = [int(bits[i : i + 2], 2) for i in range(0, len(bits), 2)]
            msgs, _ = wm.letters_encoding(frames, [symbols])
            msg = torch.tensor(msgs, device=device, dtype=torch.float32).unsqueeze(0)
            self.register_buffer("msg_enc", self.enc_c.transform_message(msg))

    @staticmethod
    def _layers(layers: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """The reference never calls ``eval()``, so its BatchNorms normalize with the statistics of
        the (single) chunk; instance norm with the BatchNorm affine reproduces that per chunk."""
        for layer in layers:
            x = layer.conv(x) * torch.sigmoid(layer.gate(x))
            x = torch.nn.functional.instance_norm(x, weight=layer.bn.weight, bias=layer.bn.bias, eps=layer.bn.eps)
        return x

    def _carrier_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """``CarrierDecoder.forward`` with per-chunk normalization (see ``_layers``)."""
        dec = self.dec_c
        h = self._layers(dec.main, x)
        if self.cfg.ensure_negative_message:
            h = torch.abs(h)
        band = torch.arange(h.shape[2], device=h.device)[None, None, :, None] < dec.message_band_size
        h = h * band
        if not self.cfg.no_normalization:
            h = h / torch.mean(h**2, dim=2, keepdim=True) ** 0.5 / (10 ** (self.message_sdr / 20))
        return h

    def _stft(self, x: torch.Tensor):
        x = torch.nn.functional.pad(x, (0, self.pad))
        fft = torch.stft(x, self.n_fft, self.hop, self.n_fft, window=self.window, return_complex=True)
        sq = fft.real**2 + fft.imag**2
        eps = (sq == 0).float() * 1e-24
        return torch.sqrt(sq + eps) - torch.sqrt(eps), torch.atan2(fft.imag, fft.real).float()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio``: [B, n_samples] float at ``sr``; returns the watermarked audio, same shape."""
        cfg = self.cfg
        y = self.up(audio.float())
        power = (y**2).mean(dim=1, keepdim=True)
        silent = power == 0
        power = torch.where(silent, torch.ones_like(power), power)
        x = y * torch.sqrt(self.avg_energy / power)
        carrier, phase = self._stft(x)
        carrier, phase = carrier[:, None], phase[:, None]
        B = carrier.shape[0]
        merged = torch.cat(
            (self._layers(self.enc_c.main, carrier), carrier.repeat(1, 32, 1, 1), self.msg_enc.expand(B, 32, -1, -1)),
            dim=1,
        )
        info = self._carrier_decoder(merged)
        if cfg.frame_level_normalization:
            info = info * torch.mean(carrier**2, dim=2, keepdim=True) ** 0.5
        elif cfg.utterance_level_normalization:
            info = info * torch.mean(carrier**2, dim=(2, 3), keepdim=True) ** 0.5
        if cfg.ensure_negative_message:
            recon = torch.relu(carrier - info)
        elif cfg.ensure_constrained_message:
            info = torch.where(info > carrier, carrier, info)
            info = torch.where(-info > carrier, -carrier, info)
            recon = info + carrier
        else:
            recon = torch.abs(info + carrier)
        mag, ph = recon[:, 0], phase[:, 0]
        spec = torch.complex(mag * torch.cos(ph), mag * torch.sin(ph))
        out = istft(spec, self.n_fft, self.hop, self.window)
        out = out[:, : out.shape[1] - self.pad] * torch.sqrt(power / self.avg_energy)
        out = torch.where(silent, y, out)
        return self.down(out)[:, : self.n_samples]
