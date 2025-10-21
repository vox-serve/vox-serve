import math
import secrets
import sys
from cmath import exp
from math import cos, pi, sin, sqrt
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from librosa import resample
from scipy.signal import butter, filtfilt
from torch import nn
from torchaudio.transforms import InverseSpectrogram, Spectrogram, TimeStretch

from ..utils import download_github_file


class PerthConfig(NamedTuple):
    use_wandb: bool
    batch_size: int
    sample_rate: int
    n_fft: int
    hop_size: int
    window_size: int
    use_lr_scheduler: bool
    stft_magnitude_min: float
    min_lr: float
    max_lr: float
    window_fn: str
    max_wmark_freq: float
    hidden_size: int
    # "simple" or "psychoacoustic"
    loss_type: str


default_hp = PerthConfig(
    use_wandb=True,
    batch_size=16,
    sample_rate=32000,
    n_fft=2048,
    hop_size=320,
    window_size=2048,
    use_lr_scheduler=False,
    stft_magnitude_min=1e-9,
    min_lr=1e-5,
    max_lr=1e-4,
    window_fn="hann",
    max_wmark_freq=2000,
    hidden_size=256,
    # loss_type="simple",
    loss_type="psychoacoustic",
)


def stream(message):
    sys.stdout.write(f"\r{message}")


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding_transposed(x: int, k: int, s: int, d: int):
    return max((x - 1) * (s - 1) + (k - 1) * d, 0)


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def pad_same_transposed(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    # pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding_transposed(iw, k[1], s[1], d[1])
    pad_h, pad_w = get_same_padding_transposed(ih, k[0], s[0], d[0]), get_same_padding_transposed(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def normalize(hp, magspec, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    magspec = (magspec - min_level_db) / (-min_level_db + headroom_db)
    return magspec


def denormalize_spectrogram(hp, magspec, headroom_db=15):
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    return magspec * (-min_level_db + headroom_db) + min_level_db


def magphase_to_cx(hp, magspec, phases):
    magspec = denormalize_spectrogram(hp, magspec)
    magspec = 10.0 ** ((magspec / 20).clip(max=10))
    phases = torch.exp(1.0j * phases)
    spectrum = magspec * phases
    return spectrum


def cx_to_magphase(hp, spec):
    phase = torch.angle(spec)
    mag = spec.abs()  # (nfreq, T)
    mag = 20 * torch.log10(mag.clip(hp.stft_magnitude_min))
    mag = normalize(hp, mag)
    return mag, phase


## Imported from Repo


def butter_lowpass(cutoff, sr=16000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=4000, sr=16000, order=16):
    b, a = butter_lowpass(cutoff, sr, order=order)
    return filtfilt(b, a, data)


def bwsk(k, n):
    # Returns k-th pole s_k of Butterworth transfer
    # function in S-domain. Note that omega_c
    # is not taken into account here
    arg = pi * (2 * k + n - 1) / (2 * n)
    return complex(cos(arg), sin(arg))


def bwj(k, n):
    # Returns (s - s_k) * H(s), where
    # H(s) - BW transfer function
    # s_k  - k-th pole of H(s)
    res = complex(1, 0)
    for m in range(1, n + 1):
        if m == k:
            continue
        else:
            res /= bwsk(k, n) - bwsk(m, n)
    return res


def bwh(n=16, fc=400, fs=16e3, length=25):
    # Returns h(t) - BW transfer function in t-domain.
    # length is in ms.
    omegaC = 2 * pi * fc
    dt = 1 / fs
    number_of_samples = int(fs * length / 1000)
    result = []
    for x in range(number_of_samples):
        res = complex(0, 0)
        if x >= 0:
            for k in range(1, n + 1):
                res += exp(omegaC * x * dt / sqrt(2) * bwsk(k, n)) * bwj(k, n)
        result.append((res).real)
    return result


def snr(input_signal, output_signal):
    Ps = np.sum(np.abs(input_signal**2))
    Pn = np.sum(np.abs((input_signal - output_signal) ** 2))
    return 10 * np.log10((Ps / Pn))


class AudioProcessor(nn.Module):
    "Module wrapper for audio processing, for easy device management"

    def __init__(self, hp: PerthConfig):
        super().__init__()
        self.hp = hp
        self.window_fn = {"hamm": torch.hamming_window, "hann": torch.hann_window, "kaiser": torch.kaiser_window}[
            hp.window_fn
        ]
        self.spectrogram = Spectrogram(
            n_fft=hp.n_fft,
            win_length=hp.window_size,
            power=None,
            hop_length=hp.hop_size,
            window_fn=self.window_fn,
            normalized=False,
        )
        self.inv_spectrogram = InverseSpectrogram(
            n_fft=hp.n_fft,
            win_length=hp.window_size,
            hop_length=hp.hop_size,
            window_fn=self.window_fn,
            normalized=False,
        )
        self.stretch = TimeStretch(
            n_freq=hp.n_fft // 2 + 1,
            hop_length=hp.hop_size,
        )

    def signal_to_magphase(self, signal):
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal.copy())
        signal = signal.float()
        spec = self.spectrogram(signal)
        mag, phase = cx_to_magphase(self.hp, spec)
        return mag, phase

    def magphase_to_signal(self, mag, phase):
        spec = magphase_to_cx(self.hp, mag, phase)
        signal = self.inv_spectrogram(spec)
        return signal


class CheckpointManager:
    def __init__(self, dataset_hp: PerthConfig = None):
        self.hparams_file = download_github_file(
            "resemble-ai",
            "Perth",
            "perth/perth_net/pretrained/implicit/hparams.yaml",
            branch="master",
        )
        if self.hparams_file.exists():
            self.hp = self.load_hparams()
            if dataset_hp is not None:
                assert self.hp == dataset_hp
        else:
            raise RuntimeError("No hparams file found!")

        self.id_file = download_github_file(
            "resemble-ai",
            "Perth",
            "perth/perth_net/pretrained/implicit/id.txt",
            branch="master",
        )
        if self.id_file.exists():
            self.id = self.id_file.read_text()
        else:
            self.id = secrets.token_urlsafe(16)
            self.id_file.write_text(self.id)

    def load_latest(self):
        return torch.load(
            download_github_file(
                "resemble-ai",
                "Perth",
                "perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar",
                branch="master",
            ),
            map_location="cpu",
        )

    def load_hparams(self):
        with self.hparams_file.open("r") as hp_file:
            return PerthConfig(**yaml.load(hp_file, Loader=yaml.FullLoader))


class Conv(nn.Module):
    def __init__(self, i, o, k, p="auto", s=1, act=True):
        super().__init__()
        assert k % 2 == 1
        if p == "auto":
            assert s == 1
            p = (k - 1) // 2
        self.conv = nn.Conv1d(i, o, k, padding=p, stride=s)
        self.act = act
        if act:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.act(x)
        return x


def compute_subband_freq(config: PerthConfig):
    nfreq = config.n_fft // 2 + 1
    topfreq = config.sample_rate / 2
    subband = int(round(nfreq * config.max_wmark_freq / topfreq))
    return subband


def magmask(magspec, p=0.05):
    s = magspec.sum(dim=1)  # (B, T)
    thresh = s.max(dim=1).values * p  # (B,)
    return (s > thresh[:, None]).float()  # (B, T)


class Encoder(nn.Module):
    """
    Inserts a watermark into a magnitude spectrogram.
    """

    def __init__(self, hidden, subband):
        super().__init__()
        self.subband = subband
        # residual encoder
        self.layers = nn.Sequential(
            Conv(self.subband, hidden, k=1),
            *[Conv(hidden, hidden, k=7) for _ in range(5)],
            Conv(hidden, self.subband, k=1, act=False),
        )

    def forward(self, magspec):
        magspec = magspec.clone()

        # create mask for valid watermark locations
        mask = magmask(magspec)[:, None]

        # crop required region of spectrogram
        sub_mag = magspec[:, : self.subband]

        # encode watermark as spectrogram residual
        res = self.layers(sub_mag) * mask

        # add residual
        magspec[:, : self.subband] += res

        # return wmarked signal and mask
        return magspec, mask


def _layers(subband, hidden):
    return nn.Sequential(
        Conv(subband, hidden, 1),
        *[Conv(hidden, hidden, k=7) for _ in range(5)],
        Conv(hidden, 2, k=1, act=False),
    )


def _masked_mean(x, m):
    return (x * m).sum(dim=2) / m.sum(dim=2)  # (B, C)


def _lerp(x, s):
    return F.interpolate(x, size=s, mode="linear", align_corners=True)


def _nerp(x, s):
    return F.interpolate(x, size=s, mode="nearest")


class Decoder(nn.Module):
    """
    Decoder a watermark from a magnitude spectrogram.
    """

    def __init__(self, hidden, subband):
        super().__init__()
        self.subband = subband
        # multi-scale decoder
        self.slow_layers = _layers(subband, hidden)
        self.normal_layers = _layers(subband, hidden)
        self.fast_layers = _layers(subband, hidden)

    def forward(self, magspec):
        mask = magmask(magspec.detach())[:, None]  # (B, 1, T)
        subband = magspec[:, : self.subband]
        B, _, T = subband.shape

        # slow branch
        slow_subband = _lerp(subband, int(T * 1.25))
        slow_out = self.slow_layers(slow_subband)  # (B, 2, T_slow)
        slow_attn = slow_out[:, :1]  # (B, 1, T_slow)
        slow_wmarks = slow_out[:, 1:]  # (B, 1, T_slow)
        slow_mask = _nerp(mask, slow_wmarks.size(2))  # (B, 1, T_slow)
        slow_wmarks = _masked_mean(slow_wmarks, slow_mask)  # (B, 1)
        slow_attn = _masked_mean(slow_attn, slow_mask)  # (B, 1)

        # normal branch
        normal_out = self.normal_layers(subband)  # (B, 2, T_normal)
        normal_attn = normal_out[:, :1]  # (B, 1, T_normal)
        normal_wmarks = normal_out[:, 1:]  # (B, 1, T_normal)
        normal_mask = _nerp(mask, normal_wmarks.size(2))  # (B, 1, T_normal)
        normal_wmarks = _masked_mean(normal_wmarks, normal_mask)  # (B, 1)
        normal_attn = _masked_mean(normal_attn, normal_mask)  # (B, 1)

        # fast branch
        fast_subband = _lerp(subband, int(T * 0.75))
        fast_out = self.fast_layers(fast_subband)  # (B, 2, T_fast)
        fast_attn = fast_out[:, :1]  # (B, 1, T_fast)
        fast_wmarks = fast_out[:, 1:]  # (B, 1, T_fast)
        fast_mask = _nerp(mask, fast_wmarks.size(2))  # (B, 1, T_fast)
        fast_wmarks = _masked_mean(fast_wmarks, fast_mask)  # (B, 1)
        fast_attn = _masked_mean(fast_attn, fast_mask)  # (B, 1)

        # combine branches with attention
        attn = torch.cat([slow_attn, normal_attn, fast_attn], dim=1)  # (B, 3)
        attn = F.softmax(attn, dim=1)  # (B, 3)
        wmarks = torch.cat([slow_wmarks, normal_wmarks, fast_wmarks], dim=1)  # (B, 3)
        wmarks = (wmarks * attn).sum(dim=1)  # (B,)

        # single float for each batch item indicating confidence of watermark
        return wmarks


def lerp(x, size=None, scale=None):
    return F.interpolate(
        x, size=size, scale_factor=scale, mode="linear", align_corners=True, recompute_scale_factor=False
    )


def random_stretch(x):
    assert x.ndim >= 3
    r = 0.9 + 0.2 * torch.rand(1).item()
    return lerp(x, scale=r)


def _attack(mag, phase, audio_proc):
    # gaussian magspec noise
    if torch.rand(1).item() < 1 / 8:
        peak = mag.mean() + 3 * mag.std()
        r = torch.randn_like(mag) * 0.01 * peak
        mag = mag + r

    # STFT-iSTFT cycle
    if torch.rand(1).item() < 1 / 4 and phase is not None:
        # # phase noise
        # if torch.rand(1).item() < 1/3:
        #     phase = phase + torch.randn_like(phase) * 0.01

        # iSTFT
        signal = audio_proc.magphase_to_signal(mag, phase)

        # # random stretch directly on signal as well
        # if torch.rand(1).item() < 1/3:
        #     signal = random_stretch(signal[None])[0]

        # STFT
        mag, phase = audio_proc.signal_to_magphase(signal)

    # random offset (NOTE: do this after phase-dependent attacks)
    if torch.rand(1).item() < 1 / 8:
        i = torch.randint(1, 13, (1,)).item()
        mag = torch.roll(mag, i, dims=2)

    # random magspec stretch (NOTE: should be near the end of attacks)
    if torch.rand(1).item() < 1 / 8:
        mag = random_stretch(mag)

    # random time masking
    # torchaudio.functional.mask_along_axis(mag, mask_param=, mask_value=mag.min().detach(), axis=2, p=0.05)

    return mag


class PerthNet(nn.Module):
    """
    PerthNet (PERceptual THreshold) watermarking model.
    Inserts and detects watermarks from a magnitude spectrogram.
    """

    def __init__(self, hp: PerthConfig):
        super().__init__()
        self.hp = hp
        self.subband = compute_subband_freq(hp)
        self.encoder = Encoder(hp.hidden_size, self.subband)
        self.decoder = Decoder(hp.hidden_size, self.subband)
        self.ap = AudioProcessor(hp)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, magspec, attack=False, phase=None):
        "Run watermarker and decoder (training)"

        # encode watermark
        wmarked, mask = self.encoder(magspec)

        # decode from un-watermarked mag
        dec_input = magspec
        if attack:
            dec_input = _attack(dec_input, phase, self.ap)
        no_wmark_pred = self.decoder(dec_input)

        # decode from watermarked mag
        dec_input = wmarked
        if attack:
            dec_input = _attack(dec_input, phase, self.ap)
        wmark_pred = self.decoder(dec_input)

        return wmarked, no_wmark_pred, wmark_pred, mask

    @staticmethod
    def from_cm(cm):
        perth_net = PerthNet(cm.hp)
        ckpt = cm.load_latest()
        assert ckpt is not None, "No checkpoint found"
        perth_net.load_state_dict(ckpt["model"])
        print(f"loaded PerthNet (Implicit) at step {ckpt['step']:,}")
        return perth_net

    @staticmethod
    def load():
        cm = CheckpointManager()
        return PerthNet.from_cm(cm)


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    return x.to(dtype=torch.float, device=device)


class PerthImplicitWatermarker:
    def __init__(self, device="cuda"):
        self.perth_net = PerthNet.load().to(device)
        self.sr = self.perth_net.hp.sample_rate

    def encode_wav(self, signal, sample_rate):
        # split signal into magnitude and phase
        signal = _to_tensor(signal, self.perth_net.device)
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        wm_signal = wm_signal.detach().cpu().numpy()
        return wm_signal

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        if change_rate:
            wm_signal = resample(
                wm_signal, orig_sr=sample_rate, target_sr=self.perth_net.hp.sample_rate, res_type="polyphase"
            )
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clip(0.0, 1.0)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        return wmark_pred.detach().cpu().numpy()
