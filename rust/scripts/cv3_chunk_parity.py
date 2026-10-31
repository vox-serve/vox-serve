"""Checks the CosyVoice3 graph program's chunked detokenizer (eager, no graphs) against whole-sequence
runs of the same modules: the flow over [voice prompt, generated tokens] with the equivalent
attention mask (prompt block-causal on its own, generated blocks anchored after it and seeing the
whole prompt), then the HiFT over the whole mel. Optionally also against the reference dump of
``cv3_parity_ref.py`` (whose prompt shares blocks with the generated frames).

    python rust/scripts/cv3_chunk_parity.py PROMPT_WAV REF_DUMP.pt [--out chunked.wav]
"""

import argparse
import json

import torch

from vox_serve.model.cosyvoice3.program import BLOCK_FRAMES, CORE, LEFT, SAMPLES_PER_FRAME, WINDOW, CosyVoice3Program


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return f"max_abs={((a - b).abs().max()):.3e} rel_l2={((a - b).norm() / b.norm()):.3e}"


def spec(x):
    return torch.stft(x.float(), 1920, 480, window=torch.hann_window(1920, device=x.device), return_complex=True).abs()


def chunk_inputs(tokens, voice, k, n_frames, dev):
    """dt_* inputs of chunk k of a request whose preseeded frames are [0, 0] + tokens (all generated)."""
    start = CORE * k
    end = min(start + WINDOW, n_frames)
    frames = [(0, 0)] * LEFT + [(int(t), voice) for t in tokens]
    win = frames[start:end] + [frames[end - 1]] * (WINDOW - (end - start))
    return {
        "dt_tok": torch.tensor(win, dtype=torch.int32, device=dev),
        "dt_reset": torch.tensor(int(k == 0), dtype=torch.int32, device=dev),
        "dt_len": torch.tensor(end - start, dtype=torch.int32, device=dev),
        "dt_last": torch.tensor(int(start + CORE >= n_frames), dtype=torch.int32, device=dev),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_wav")
    ap.add_argument("dump")
    ap.add_argument("--out")
    ap.add_argument("--strict", action="store_true", help="disable TF32 (isolates windowing errors)")
    ap.add_argument("--config", default="{}", help="extra program config keys (JSON), e.g. fuse_codec")
    a = ap.parse_args()
    if a.strict:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")
    ref = torch.load(a.dump)
    gtok = ref["gen_token"][0].long()
    G = gtok.shape[0]
    n_frames = LEFT + G
    n_chunks = -(-n_frames // CORE)
    cfg = {"model": "cosyvoice3", "max_batch_size": 2, "max_context": 512, "history_chunks": n_chunks}
    cfg.update(json.loads(a.config))
    p = CosyVoice3Program(cfg, dev)
    p.allocate_state()
    voice = 1
    ptok = torch.tensor(p.hook_encode_voice(a.prompt_wav, voice)["speech_tokens"])
    print("prompt tokens", ptok.shape[0], "generated", G, "chunks", n_chunks)

    # Chunked: two identical requests in detok slots 0 and 1 (checks the batched indexing).
    mels, pcm = [], []
    orig = p._flow_chunk

    def rec(*args):
        x = orig(*args)
        mels.append((x[0].clone(), int(args[-1][0])))
        return x

    p._flow_chunk = rec
    nb = 2
    out = {"audio": torch.zeros(nb, p.detok_samples, dtype=torch.int16, device=dev)}
    with torch.no_grad():
        for k in range(n_chunks):
            one = chunk_inputs(gtok.tolist(), voice, k, n_frames, dev)
            arena = {key: torch.stack([v, v]) for key, v in one.items()}
            arena["dt_slot"] = torch.tensor([0, 1], dtype=torch.int32, device=dev)
            p.detok_step(arena, out, nb)
            row = out["audio"]
            n = [int(row[i, -1]) for i in range(nb)]
            assert n[0] == n[1] and torch.equal(row[0, : n[0]], row[1, : n[1]]), "batched rows differ"
            pcm.append(row[0, : n[0]].clone())
    mel_c = torch.cat([m[:c] for m, c in mels])  # [2G, 80]
    audio_c = torch.cat(pcm).float() / 32767
    print("chunked frames", mel_c.shape[0], "samples", audio_c.shape[0], "expected", 2 * G * SAMPLES_PER_FRAME)

    # Whole sequence with the equivalent mask.
    f = p.flow
    m = int(p.voice_len[voice]) // 2
    ftok = ptok[-m:].to(dev)
    P2 = 2 * m
    T = P2 + 2 * G
    with torch.no_grad():
        mu_p = f.sequence_mu(ftok[None])
        emb = f.input_embedding(torch.cat([ftok[-LEFT:], gtok.to(dev)]))[None]
        mu_g = f.token_mu(torch.nn.functional.pad(emb, (0, 0, 0, 3)), LEFT)
        mu = torch.cat([mu_p, mu_g], 1)
        _, mel_p, xvec = p.frontend.encode(a.prompt_wav)
        cond = torch.zeros(1, T, 80, device=dev)
        cond[0, :P2] = mel_p[-P2:].to(dev)
        spk = f.speaker(xvec[None].to(dev))
        i = torch.arange(T, device=dev)
        blk = torch.where(i < P2, i // BLOCK_FRAMES, (i - P2) // BLOCK_FRAMES)
        same = (i[:, None] < P2) == (i[None, :] < P2)
        mask = (same & (blk[None, :] <= blk[:, None])) | ((i[:, None] >= P2) & (i[None, :] < P2))
        mel_w = f.solve(mu, spk, cond, p.flow_noise[:T][None], block_mask=mask, mods=p.mods)[0, P2:]
        print("flow chunked vs whole:", rel(mel_c, mel_w))
        noise = p.src_noise[: 2 * G * SAMPLES_PER_FRAME][None]
        audio_w = p.hift(mel_c.T[None].float(), noise)[0]
        print("hift chunked vs whole (same mel):", rel(audio_c, audio_w))
        audio_ww = p.hift(mel_w.T[None].float(), noise)[0]
        print("audio chunked vs whole flow + hift:", rel(audio_c, audio_ww))
        # Waveforms decorrelate once f0 differs slightly (the source phase integrates it): compare spectra.
        print("  |STFT| (n_fft 1920, hop 480):", rel(spec(audio_c), spec(audio_ww)))
        print("flow chunked vs reference block-streaming dump:", rel(mel_c.T, ref["mel_block"][0]))
    if a.out:
        import soundfile as sf

        sf.write(a.out, audio_c.cpu().numpy(), 24000)


if __name__ == "__main__":
    main()
