"""Parity checks for the graph-capturable watermarks: the sync-free istft vs torch.istft, batched
SilentCipher (CSM) vs the Python worker's per-chunk run_watermark path, and batched Perth
(Chatterbox) vs PerthImplicitWatermarker.encode_wav. Each batched module is also graph-captured."""

import torch
import torchaudio

from vox_serve.model.chatterbox.program import PerthBatched
from vox_serve.rt.watermark import SILENTCIPHER_KEY, SilentCipherBatched, istft
from vox_serve.watermarker import silentcipher


def check_istft(dev):
    g = torch.Generator(device=dev).manual_seed(1)
    for n_fft, hop, win in [(2048, 320, 2048), (2048, 512, 2048), (1024, 256, 800)]:
        x = torch.randn(3, 44100, device=dev, generator=g)
        w = torch.hann_window(win, device=dev)
        spec = torch.stft(x, n_fft, hop, win, window=w, return_complex=True)
        ref = torch.istft(spec, n_fft, hop, win, window=w)
        err = (istft(spec, n_fft, hop, w) - ref).abs().max().item()
        print(f"istft n_fft={n_fft} hop={hop} win={win}: max|ours-torch|={err:.2e}")
        assert err < 1e-4, "istft parity failed"


def captured(module, x):
    with torch.no_grad():
        inp = x.clone()
        module(inp)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = module(inp)
        graph.replay()
        torch.cuda.synchronize()
        return out


def test_audio(dev, n):
    g = torch.Generator(device=dev).manual_seed(0)
    t = torch.arange(n, device=dev) / 24000
    return torch.stack(
        [
            0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.randn(n, device=dev, generator=g),
            0.1 * torch.randn(n, device=dev, generator=g),
            torch.zeros(n, device=dev),
        ]
    )


def check(name, module, reference, audio, tol):
    with torch.no_grad():
        out = module(audio)
    graph_out = captured(module, audio)
    n = audio.shape[1]
    for i in range(audio.shape[0]):
        ref = reference(audio[i].clone())[:n].to(out.device).float()
        m = min(ref.shape[0], out.shape[1])
        err = (out[i, :m] - ref[:m]).abs().max().item()
        gerr = (graph_out[i] - out[i]).abs().max().item()
        delta = (ref - audio[i, : ref.shape[0]]).abs().max().item()
        print(f"{name} chunk {i}: max|batched-ref|={err:.2e} max|graph-eager|={gerr:.2e} max|wm-orig|={delta:.2e}")
        assert err < tol and gerr < 1e-4, f"{name} parity failed"


def main():
    dev = torch.device("cuda")
    check_istft(dev)

    n = 19200  # CSM chunk (10 frames)
    wm = silentcipher.get_model(model_type="44.1k", device=dev)

    def sc_ref(x):
        y = torchaudio.functional.resample(x, orig_freq=24000, new_freq=wm.sr)
        enc = wm.encode_wav(y, wm.sr, SILENTCIPHER_KEY)
        if isinstance(enc, tuple):
            enc = enc[0]
        return torchaudio.functional.resample(enc, orig_freq=wm.sr, new_freq=24000)

    check("silentcipher", SilentCipherBatched(24000, n, dev), sc_ref, test_audio(dev, n), 1e-3)

    perth = PerthBatched(24000, dev)
    ap = perth.net.ap

    def perth_ref(x):
        # encode_wav's math, eager (torchaudio InverseSpectrogram -> torch.istft).
        with torch.no_grad():
            y = perth.up(x[None].float())
            mag, phase = ap.signal_to_magphase(y)
            wm_mag, _ = perth.net.encoder(mag)
            return perth.down(ap.magphase_to_signal(wm_mag, phase))[0]

    check("perth", perth, perth_ref, test_audio(dev, 24000)[:2], 1e-3)
    print("ok")


if __name__ == "__main__":
    main()
