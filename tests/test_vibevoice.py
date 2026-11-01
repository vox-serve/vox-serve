"""VibeVoice-Realtime pieces that need neither a GPU nor the weights: the streamed acoustic decoder, the
DPM-Solver sampler against diffusers' scheduler, voice names and the prompt layout."""

import json

import pytest
import torch

from vox_serve.model.vibevoice.model import (
    ENGLISH_VOICES,
    dpm_solver_plan,
    match_voice,
    resolve_voices,
    sample_latent,
)
from vox_serve.tokenizer.vibevoice import VibeVoiceAcousticDecoder

TINY_DECODER = {
    "vae_dim": 4,
    "decoder_n_filters": 2,
    "encoder_depths": "1-2-1",
    "decoder_depths": None,
    "decoder_ratios": [3, 2],
    "channels": 1,
    "layernorm_eps": 1e-5,
    "mixer_layer": "depthwise_conv",
    "causal": True,
    "disable_last_norm": True,
}


@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_decoder_chunks_match_one_pass(chunk):
    torch.manual_seed(0)
    dec = VibeVoiceAcousticDecoder(TINY_DECODER).eval()
    for p in dec.parameters():  # break the identity layer scales so every block matters
        p.data.normal_(0, 0.5)
    lat = torch.randn(2, 4, 10)

    def zeros():
        return [torch.zeros(2, c, n) for c, n in dec.state_shapes()]

    with torch.no_grad():
        whole, _ = dec(lat, zeros())
        st, parts = zeros(), []
        for i in range(0, 10, chunk):
            y, st = dec(lat[:, :, i : i + chunk], st)
            parts.append(y)
    assert whole.shape == (2, 1, 10 * dec.hop)
    torch.testing.assert_close(torch.cat(parts, -1), whole, rtol=1e-5, atol=1e-6)


class _ToyHead(torch.nn.Module):
    """A deterministic v-prediction with the diffusion head's interface."""

    def __init__(self, latent=8, H=16):
        super().__init__()
        self.cond_proj = torch.nn.Linear(H, H, bias=False)
        self.x_proj = torch.nn.Linear(latent, H, bias=False)
        self.out = torch.nn.Linear(H, latent, bias=False)

    def forward(self, x, c):
        return self.out(torch.tanh(self.x_proj(x) + c))


@pytest.mark.parametrize("steps", [3, 5, 10])
def test_sampler_matches_diffusers_scheduler(steps):
    from diffusers import DPMSolverMultistepScheduler

    torch.manual_seed(0)
    cfg = {"ddpm_num_steps": 1000, "ddpm_beta_schedule": "cosine", "prediction_type": "v_prediction"}
    head = _ToyHead()
    t_emb = torch.randn(steps, 16)
    n, cfg_scale = 3, 1.5
    cond, neg, x = torch.randn(n, 16), torch.randn(n, 16), torch.randn(n, 8)
    with torch.no_grad():
        ours = sample_latent(head, dpm_solver_plan(cfg, steps), t_emb, cond, neg, x, cfg_scale)
        # The reference's sample_speech_tokens, with diffusers' scheduler.
        sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction"
        )
        sched.set_timesteps(steps)
        condition = head.cond_proj(torch.cat([cond, neg]))
        speech = torch.cat([x, x])
        for k, t in enumerate(sched.timesteps):
            half = speech[:n]
            eps = head(torch.cat([half, half]), condition + t_emb[k])
            c_eps, u_eps = eps.split(n)
            half_eps = u_eps + cfg_scale * (c_eps - u_eps)
            speech = sched.step(torch.cat([half_eps, half_eps]), t, speech).prev_sample
    torch.testing.assert_close(ours, speech[:n], rtol=1e-5, atol=1e-5)


def test_voice_names():
    names = resolve_voices(None)
    assert len(names) == 25 and names[: len(ENGLISH_VOICES)] == ENGLISH_VOICES
    assert resolve_voices("english") == ENGLISH_VOICES
    assert resolve_voices("en-Carter_man, fr-Spk1_woman") == ["en-Carter_man", "fr-Spk1_woman"]
    assert match_voice("carter", names) == "en-Carter_man"
    assert match_voice("EN-EMMA_WOMAN", names) == "en-Emma_woman"
    assert match_voice("Spk1_woman", ["fr-Spk1_woman"]) == "fr-Spk1_woman"
    assert match_voice("nobody", names) is None


def test_prompt_layout(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from vox_serve.model.vibevoice.program import DECODE, PROMPT, VibeVoiceProgram  # noqa: F401

    words = [f"w{i}" for i in range(200)]
    tok = Tokenizer(models.WordLevel({w: i + 1 for i, w in enumerate(words)} | {"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    voices = {
        "en-Carter_man": {"index": 0, "offset": 0, "len": 3},
        "en-Emma_woman": {"index": 1, "offset": 3, "len": 4},
    }
    meta = {"tokenizer_json": str(path), "n_codebooks": 64, "max_text_tokens": 150, "voices": voices}
    meta["default_voice"] = "en-Carter_man"
    program = VibeVoiceProgram.for_prompts(json.loads(json.dumps(meta)))

    text = " ".join(words[:100])
    p = program.preprocess(f"  {text}  ", None, {"voice": "emma"})
    rows = [p.tokens[i : i + 64] for i in range(0, len(p.tokens), 64)]
    masks = [p.mask[i : i + 64] for i in range(0, len(p.mask), 64)]
    assert len(rows) == 4  # the voice's prompt rows
    assert [r[0] for r in rows] == [3, 4, 5, 6]  # its rows of the packed voice prompts
    ids = [t for r in rows for t in r[1:]]
    # 63 text tokens per row (this tokenizer drops the trailing "\n"), then zero padding.
    assert ids[:100] == list(range(1, 101)) and not any(ids[100:])
    assert all(m[:3] == [PROMPT, 100, 1] for m in masks)

    p = program.preprocess("w1", None, {})  # default voice
    assert len(p.tokens) == 3 * 64 and p.mask[2] == 0
    with pytest.raises(ValueError, match="unknown VibeVoice voice"):
        program.preprocess("w1", None, {"voice": "nobody"})
    with pytest.raises(ValueError, match="exceeds max_text_tokens"):
        program.preprocess(" ".join(words), None, {})
    with pytest.raises(ValueError, match="does not clone"):
        program.preprocess("w1", "ref.wav", {})
    with pytest.raises(ValueError, match="no text"):
        program.preprocess("   ", None, {})
