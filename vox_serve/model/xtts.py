import json
import os
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from huggingface_hub import hf_hub_download

from ..flashinfer_utils import FlashInferWrapper
from ..encoder.xtts import (
    ConditioningEncoder,
    HifiDecoder,
    PerceiverResampler,
    load_fsspec,
)
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.base import DecoderCache
from ..tokenizer.xtts import VoiceBpeTokenizer
from .base import BaseLM, PreprocessOutput


@dataclass
class XttsAudioConfig:
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
        dvae_sample_rate (int): The sample rate of the DVAE
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000
    dvae_sample_rate: int = 22050


@dataclass
class XttsArgs:
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


@dataclass
class XttsConfig:
    """Defines parameters for XTTS TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (XttsArgs):
            Model architecture arguments. Defaults to `XttsArgs()`.

        audio (XttsAudioConfig):
            Audio processing configuration. Defaults to `XttsAudioConfig()`.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        gpt_cond_len (int):
            Length of the audio used as conditioning for the autoregressive  model. If audio is shorter,
            then audio length is used else the first `gpt_cond_len` secs is used. Defaults to 12 seconds.

        gpt_cond_chunk_len (int):
            Audio chunk size in seconds. Audio is split into chunks and latents are extracted for each chunk. Then
            the latents are averaged. Chunking improves the stability. It must be <= gpt_cond_len.
            If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to `4` seconds.

        max_ref_len (int):
            Maximum number of seconds of audio to be used as conditioning for the decoder. Defaults to `10`.

        sound_norm_refs (bool):
            Whether to normalize the conditioning audio. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> config = XttsConfig()
    """

    model: str = "xtts"
    _supports_cloning: bool = True
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    languages: list[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False

    # new config
    checkpoint_path: str = "~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
    repo_id: str = "coqui/XTTS-v2"
    layer_norm_epsilon=1e-5

    def load_json(self, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            current = getattr(self, key, None)
            if isinstance(value, dict) and hasattr(current, "__dict__"):
                for k, v in value.items():
                    setattr(current, k, v)
            else:
                setattr(self, key, value)
        return self
    
    def to_dict(self):
        return self.__dict__.copy()

    def __getitem__(self, key):
        return getattr(self, key)
        
    def __contains__(self, key):
        return hasattr(self, key)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            import random
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class XttsGptMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class XttsGptAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if query_states.dim() == 4:
            query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
            key_states = key_states.reshape(-1, self.num_heads, self.head_dim)
            value_states = value_states.reshape(-1, self.num_heads, self.head_dim)

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.view(*input_shape, self.hidden_size).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output

class XttsGptBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, layer_norm_epsilon: int, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = XttsGptAttention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = XttsGptMLP(hidden_size, hidden_size * 4)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # FlashInfer Layer expects kv_cache specifically scoped for this layer_idx
        layer_kv_cache = kv_cache[self.layer_idx]
        
        attn_output = self.attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=layer_kv_cache,
        )
        
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class XttsGptModel(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, layer_norm_epsilon: int):
        super().__init__()
        self.h = nn.ModuleList([
            XttsGptBlock(hidden_size, num_heads, layer_norm_epsilon, layer_idx=i) #(transformers) hidden_size == embed_dim
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

    # inputs_embeds have to have positional embeddings.
    # Causal mask is applied inside Block
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = inputs_embeds
        
        for block in self.h:
            hidden_states = block(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache,
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class SpeakerManager:
    def __init__(self, speaker_file_path=None):
        self.speakers = torch.load(speaker_file_path, weights_only=False)

    @property
    def name_to_id(self):
        return self.speakers

    @property
    def num_speakers(self):
        return len(self.name_to_id)

    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())


class LanguageManager:
    def __init__(self, config):
        self.langs = config["languages"]

    @property
    def name_to_id(self):
        return self.langs

    @property
    def num_languages(self):
        return len(self.name_to_id)

    @property
    def language_names(self):
        return list(self.name_to_id)




XTTS_OVERLAP_WAV_LEN = 1024  # crossfade window between HiFiGAN chunks


@dataclass
class XTTSDecoderCache(DecoderCache):
    """Per-request state for streaming HiFiGAN decoding.

    Each chunk runs HiFiGAN on every accumulated latent, slices
    ``[prev_wav_len - overlap : cur_wav_len - overlap]``, and crossfades with the
    previous chunk's ``wav_overlap``.
    """

    speaker_embedding: torch.Tensor = None       # (B, 512, 1)
    accumulated_latents: torch.Tensor = None     # (B, MAX_MEL, hidden), fp32
    latent_count: torch.Tensor = None            # (B, 1), int32
    wav_overlap: torch.Tensor = None             # (B, 1, OVERLAP)
    has_prev_chunk: torch.Tensor = None          # (B, 1), int32
    prev_wav_len: torch.Tensor = None            # (B, 1), int32


class XTTSModel(BaseLM, nn.Module):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        # fp16 is the default: bf16 accumulates audible drift across 30 layers; fp32
        # is unsupported by FlashInfer attention.
        dtype: torch.dtype = torch.float16,
        enable_torch_compile: bool = False,
        audio_decoder_device: str | None = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        BaseLM.__init__(self, model_name, device, dtype, enable_torch_compile, audio_decoder_device)
        self.config = XttsConfig()
        self.checkpoint_path = self.config.checkpoint_path
        self.repo_id = self.config.repo_id

        config_path = self._get_cached_or_download_file(self.repo_id, "config.json")
        vocab_path = self._get_cached_or_download_file(self.repo_id, "vocab.json")
        model_path = self._get_cached_or_download_file(self.repo_id, "model.pth")
        speakers_path = self._get_cached_or_download_file(self.repo_id, "speakers_xtts.pth")

        if os.path.exists(config_path):
            self.config.load_json(config_path)

        self.args = getattr(self.config, "model_args", self.config)
        if speakers_path and os.path.exists(speakers_path):
            self.speaker_manager = SpeakerManager(speakers_path)
        self.language_manager = LanguageManager(self.config)

        self.mel_stats_path = None
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint
        self.gpt_batch_size = self.args.gpt_batch_size
        if vocab_path and Path(vocab_path).is_file():
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")
        
        if self.args.gpt_number_text_tokens:
            # initialize GPT
            self.start_prompt_token = self.args.gpt_start_audio_token
            self.stop_prompt_token = self.args.gpt_stop_audio_token
            self.max_conditioning_inputs = 1
            self.max_gen_mel_tokens = self.args.gpt_max_audio_tokens - self.max_conditioning_inputs - 2
            self.max_mel_tokens = -1 if self.args.gpt_max_audio_tokens == -1 else self.args.gpt_max_audio_tokens + 2 + self.max_conditioning_inputs
            self.max_text_tokens = -1 if self.args.gpt_max_text_tokens == -1 else self.args.gpt_max_text_tokens + 2
            self.conditioning_encoder = ConditioningEncoder(80, self.args.gpt_n_model_channels, num_attn_heads=self.args.gpt_n_heads)
            self.conditioning_dropout = nn.Dropout1d(0.1)
            self.average_conditioning_embeddings = False
            self.use_perceiver_resampler = self.args.gpt_use_perceiver_resampler
            self.perceiver_cond_length_compression = 256
            self.text_embedding = nn.Embedding(self.args.gpt_number_text_tokens, self.args.gpt_n_model_channels)
            self.mel_embedding = nn.Embedding(self.args.gpt_num_audio_tokens, self.args.gpt_n_model_channels)

            self.gpt = XttsGptModel(
                num_layers=self.args.gpt_layers,
                hidden_size=self.args.gpt_n_model_channels,
                num_heads=self.args.gpt_n_heads,
                layer_norm_epsilon=self.config.layer_norm_epsilon,
            )

            self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_tokens, self.args.gpt_n_model_channels)
            self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens, self.args.gpt_n_model_channels)
            self.mel_layer_pos_embedding = None
            self.text_layer_pos_embedding = None
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0
            self.final_norm = nn.LayerNorm(self.args.gpt_n_model_channels)
            self.text_head = nn.Linear(self.args.gpt_n_model_channels, self.args.gpt_number_text_tokens)
            self.mel_head = nn.Linear(self.args.gpt_n_model_channels, self.args.gpt_num_audio_tokens)
            self.conditioning_perceiver = PerceiverResampler(dim=self.args.gpt_n_model_channels, depth=2, dim_context=self.args.gpt_n_model_channels, num_latents=32, dim_head=64, heads=8, ff_mult=4, use_flash_attn=False)

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )

        self.register_buffer("mel_stats", torch.ones(80))

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)
        checkpoint = self.prepare_checkpoint(checkpoint)
        self.load_state_dict(checkpoint, strict=True)

        self.hifigan_decoder.eval()
        self.gpt.eval()
        self.eval()

        # repetition_window=1 means "accumulate all past tokens in a single slot",
        # matching HF generate()'s RepetitionPenaltyLogitsProcessor. Small windows
        # let the model re-emit early tokens and produce garbled audio.
        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=0.85,
            min_p=None,
            temperature=0.75,
            repetition_penalty=10.0,
            repetition_window=1,
            cfg_scale=None,
        )

        # output_audio_length: HiFiGAN expands latents by interp×4 then ×24000/22050
        # then waveform_decoder ×256.
        self._detokenize_interval = 20
        self._detokenize_overlap = 1
        _stage1 = self._detokenize_interval * (
            self.args.gpt_code_stride_len // self.args.output_hop_length
        )
        _stage2 = int(_stage1 * self.args.output_sample_rate / self.args.input_sample_rate)
        self._output_audio_length = _stage2 * self.args.output_hop_length

        self._max_accumulated_latents = int(self.args.gpt_max_audio_tokens)
        self._wav_len_stride = int(self.args.gpt_code_stride_len // self.args.output_hop_length)
        self._wav_len_osr = int(self.args.output_sample_rate)
        self._wav_len_isr = int(self.args.input_sample_rate)
        self._wav_len_hop = int(self.args.output_hop_length)

        # forward() stashes hidden states + row count here for sampling() to read.
        # Use in-place GPU ops so the values propagate across CUDA-graph replays;
        # Python attribute assignments only run at capture time.
        _bridge_rows = 1024
        self.register_buffer(
            "_hidden_states_buffer",
            torch.zeros(_bridge_rows, self.hidden_size, dtype=self.dtype),
            persistent=False,
        )
        self.register_buffer(
            "_last_n_rows_buffer",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )

        self.default_speaker_ref_dict: dict | None = None
        self.default_speaker_ref_dict_audio_decoder: dict | None = None
        self._mel_stop_token_id = int(self.args.gpt_stop_audio_token)

        self.to(self.device)
        for module in (
            self.gpt,
            self.text_embedding,
            self.mel_embedding,
            self.text_pos_embedding,
            self.mel_pos_embedding,
            self.final_norm,
            self.text_head,
            self.mel_head,
            self.conditioning_dropout,
        ):
            module.to(self.dtype)
        # Conditioning encoder/perceiver and HiFiGAN are pinned to fp32: MHA over
        # the cloning-mel sequence overflows in fp16, and HiFiGAN matches the
        # reference's vocoder dtype.
        self.conditioning_encoder.to(torch.float32)
        self.conditioning_perceiver.to(torch.float32)
        self.hifigan_decoder.to(self.audio_decoder_device, dtype=torch.float32)

        self._init_default_speaker_ref(speakers_path)

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
        return self.args.gpt_num_audio_tokens

    @property
    def detokenize_interval(self) -> int:
        return self._detokenize_interval

    @property
    def detokenize_overlap(self) -> int:
        return self._detokenize_overlap

    @property
    def max_tokens(self) -> int:
        return self.args.gpt_max_audio_tokens

    @property
    def n_channels(self) -> int:
        return 1

    @property
    def output_audio_length(self) -> int:
        return self._output_audio_length

    @property
    def needs_input_features(self) -> bool:
        return True

    @property
    def needs_input_masks(self) -> bool:
        return True

    @property
    def supports_audio_input(self) -> bool:
        # Required so the scheduler forwards ``audio_path`` to preprocess
        # (used here for reference-audio voice cloning).
        return True

    @property
    def supports_postprocess_cuda_graph(self) -> bool:
        # HiFiGAN runs on a variable-length latent slice each chunk, so postprocess
        # cannot be captured. Prefill/decode still use CUDA graphs.
        return False

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return int(token_ids[0]) == self._mel_stop_token_id

    @staticmethod
    def _load_audio(audiopath, sampling_rate=22050):
        audio, lsr = torchaudio.load(audiopath)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        audio = audio.clamp(-1, 1)
        return audio

    def _wav_to_mel_cloning(
        self,
        wav: torch.Tensor,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int = 80,
        sample_rate: int = 22050,
        f_min: int = 0,
        f_max: int = 8000,
    ) -> torch.Tensor:
        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2,
            normalized=False,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).to(wav.device)
        mel = mel_stft(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel / self.mel_stats.to(wav.device).unsqueeze(0).unsqueeze(-1)
        return mel

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio: torch.Tensor, sr: int, length: int = 30, chunk_length: int = 6) -> torch.Tensor:
        MIN_AUDIO_SECONDS = 0.33
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        # conditioning_encoder + conditioning_perceiver live in fp32 (see __init__)
        # because their multi-head-attention overflows in fp16. Feed fp32 mel in
        # and cast the result to the model's dtype at the end so the caller can
        # concat the cond_latent into the (potentially fp16) GPT prefix.
        cond_dtype = torch.float32
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]
                if audio_chunk.size(-1) < int(22050 * MIN_AUDIO_SECONDS):
                    continue
                mel_chunk = self._wav_to_mel_cloning(
                    audio_chunk, n_fft=2048, hop_length=256, win_length=1024
                )
                mel_chunk = mel_chunk.to(device=self.device, dtype=cond_dtype)
                conds = self.conditioning_encoder(mel_chunk)  # (1, 1024, T) fp32
                conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)  # (1, 1024, 32) fp32
                style_embs.append(conds)
            if not style_embs:
                raise RuntimeError(
                    f"Reference audio too short (minimum {MIN_AUDIO_SECONDS:.2f}s)."
                )
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = self._wav_to_mel_cloning(audio, n_fft=4096, hop_length=1024, win_length=4096)
            mel = mel.to(device=self.device, dtype=cond_dtype)
            cond_latent = self.conditioning_encoder(mel)
        return cond_latent.transpose(1, 2).to(self.dtype)  # (1, 32, 1024) in model dtype

    @torch.inference_mode()
    def get_speaker_embedding(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        speaker_encoder_device = next(self.hifigan_decoder.speaker_encoder.parameters()).device
        return (
            self.hifigan_decoder.speaker_encoder.forward(
                audio_16k.to(speaker_encoder_device), l2_norm=True
            )
            .unsqueeze(-1)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length: int = 30,
        gpt_cond_len: int = 6,
        gpt_cond_chunk_len: int = 6,
        load_sr: int = 22050,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_paths = audio_path if isinstance(audio_path, list) else [audio_path]
        speaker_embeddings = []
        audios = []
        for fp in audio_paths:
            audio = self._load_audio(fp, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            speaker_embeddings.append(self.get_speaker_embedding(audio, load_sr))
            audios.append(audio)
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )
        speaker_embedding = torch.stack(speaker_embeddings).mean(dim=0)
        return gpt_cond_latents, speaker_embedding

    def _init_default_speaker_ref(self, speakers_path: str | None) -> None:
        """Cache a default conditioning so requests without ``audio_path`` still work.

        Pick by name via ``VOX_XTTS_SPEAKER`` env var; falls back to the first entry.
        """
        if not (speakers_path and os.path.exists(speakers_path)):
            return

        try:
            speakers = torch.load(speakers_path, map_location="cpu", weights_only=False)
        except Exception:
            return

        if not (isinstance(speakers, dict) and speakers):
            return

        preferred = os.environ.get("VOX_XTTS_SPEAKER")
        name = preferred if (preferred and preferred in speakers) else next(iter(speakers))
        entry = speakers[name]
        if not (isinstance(entry, dict) and "gpt_cond_latent" in entry and "speaker_embedding" in entry):
            return

        gpt_cond = entry["gpt_cond_latent"]
        spk = entry["speaker_embedding"]
        if not torch.is_tensor(gpt_cond):
            gpt_cond = torch.as_tensor(gpt_cond)
        if not torch.is_tensor(spk):
            spk = torch.as_tensor(spk)
        if gpt_cond.ndim == 2:
            gpt_cond = gpt_cond.unsqueeze(0)
        if spk.ndim == 1:
            spk = spk.view(1, -1, 1)
        elif spk.ndim == 2:
            spk = spk.unsqueeze(-1)

        self.default_speaker_ref_dict = {
            "gpt_cond_latent": gpt_cond.to(self.device, dtype=torch.float32),
            "speaker_embedding": spk.to(self.device, dtype=torch.float32),
        }
        self.default_speaker_ref_dict_audio_decoder = {
            "gpt_cond_latent": gpt_cond.to(self.audio_decoder_device, dtype=torch.float32),
            "speaker_embedding": spk.to(self.audio_decoder_device, dtype=torch.float32),
        }

    def audio_decoder_initial_cache(self, batch_size: int) -> XTTSDecoderCache:
        """Allocate a fresh per-request decoder cache, broadcasting the default speaker
        when no reference audio was supplied.
        """
        device = self.audio_decoder_device
        # Default speaker embedding (B, 512, 1); fall back to zeros until Phase 3 runs.
        if self.default_speaker_ref_dict_audio_decoder is not None:
            spk = self.default_speaker_ref_dict_audio_decoder["speaker_embedding"]
            speaker_embedding = spk.expand(batch_size, -1, -1).clone().contiguous()
        else:
            speaker_embedding = torch.zeros(
                batch_size, self.args.d_vector_dim, 1,
                dtype=torch.float32, device=device,
            )

        accumulated_latents = torch.zeros(
            batch_size, self._max_accumulated_latents, self.hidden_size,
            dtype=torch.float32, device=device,
        )
        latent_count = torch.zeros(
            batch_size, 1, dtype=torch.int32, device=device,
        )
        wav_overlap = torch.zeros(
            batch_size, self.n_channels, XTTS_OVERLAP_WAV_LEN,
            dtype=torch.float32, device=device,
        )
        has_prev_chunk = torch.zeros(
            batch_size, 1, dtype=torch.int32, device=device,
        )
        prev_wav_len = torch.zeros(
            batch_size, 1, dtype=torch.int32, device=device,
        )
        return XTTSDecoderCache(
            speaker_embedding=speaker_embedding,
            accumulated_latents=accumulated_latents,
            latent_count=latent_count,
            wav_overlap=wav_overlap,
            has_prev_chunk=has_prev_chunk,
            prev_wav_len=prev_wav_len,
        )

    # ------------------------------------------------------------------
    # Inference: preprocess / forward / sampling / postprocess
    # ------------------------------------------------------------------

    def _build_prefix_embedding(
        self,
        text_token_ids: torch.Tensor,
        gpt_cond_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Match xtts encoder.GPT.compute_embeddings:

            emb = [cond_latents | text_embedding(text) + text_pos_embedding | start_mel + mel_pos[0]]
        """
        device = self.device

        # 1. text embedding with learned text-position embedding (cond-relative positions 0..T-1).
        text_emb = self.text_embedding(text_token_ids) + self.text_pos_embedding(text_token_ids)

        # 2. start_audio_token embedding + mel position 0.
        start_token = torch.tensor(
            [[self.args.gpt_start_audio_token]], dtype=torch.long, device=device
        )
        start_mel_emb = self.mel_embedding(start_token) + self.mel_pos_embedding(start_token)

        # 3. cond_latents has shape (1, T_cond, hidden_size); no positional embedding.
        cond_emb = gpt_cond_latent.to(device=device, dtype=text_emb.dtype)
        if cond_emb.ndim == 2:
            cond_emb = cond_emb.unsqueeze(0)

        # Final prefix: cond | text(+pos) | start_mel(+pos[0]).
        prefix = torch.cat([cond_emb, text_emb, start_mel_emb], dim=1)
        return prefix  # (1, prefix_len, hidden_size)

    @torch.no_grad()
    def preprocess(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
        language: str = "en",
        **kwargs: Any,
    ) -> PreprocessOutput:
        if prompt is None:
            raise ValueError("XTTS preprocess requires a `prompt` text argument.")

        device = self.device

        # 1. Tokenize text and bracket with [START]/[STOP] (matches XTTS GPT.compute_embeddings).
        bpe_ids = self.tokenizer.encode(prompt, language)
        text_ids = (
            [int(self.args.gpt_start_text_token)]
            + list(bpe_ids)
            + [int(self.args.gpt_stop_text_token)]
        )
        text_token_ids = torch.tensor([text_ids], dtype=torch.long, device=device)

        # 2. Obtain conditioning latents — per-request audio override, otherwise default speaker.
        if audio_path is not None:
            gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(audio_path)
        else:
            if self.default_speaker_ref_dict is None:
                raise RuntimeError(
                    "No `audio_path` provided and no default speaker is cached. "
                    "Pass audio_path, or ensure speakers_xtts.pth is downloadable."
                )
            gpt_cond_latent = self.default_speaker_ref_dict["gpt_cond_latent"]
            speaker_embedding = self.default_speaker_ref_dict["speaker_embedding"]

        # 3. Build the prefix in embedding space.
        prefix_emb = self._build_prefix_embedding(text_token_ids, gpt_cond_latent)
        prefix_len = prefix_emb.shape[1]

        # 4. Pack vox-serve's PreprocessOutput shape.
        #    input_tokens shape: (prefix_len, n_codebooks=1) -- values are placeholders for prefix.
        #    input_masks True everywhere -> forward will use input_features directly.
        input_tokens = torch.full(
            (prefix_len, self.n_codebooks),
            fill_value=int(self.args.gpt_start_audio_token),
            dtype=torch.int32,
            device=device,
        )
        # input_features: (prefix_len, hidden_size)
        input_features = prefix_emb.squeeze(0).to(device=device)
        # input_masks: (prefix_len, n_codebooks) bool, all True for prefix.
        input_masks = torch.ones(
            prefix_len, self.n_codebooks, dtype=torch.bool, device=device,
        )

        # 5. Per-request decoder cache (override default speaker_embedding into it).
        decoder_cache = self.audio_decoder_initial_cache(batch_size=1)
        if speaker_embedding is not None:
            spk = speaker_embedding.to(
                device=decoder_cache.speaker_embedding.device,
                dtype=decoder_cache.speaker_embedding.dtype,
            )
            if spk.ndim == 2:
                spk = spk.unsqueeze(-1)
            decoder_cache.speaker_embedding.copy_(spk)

        # Pre-seed the repetition-penalty cache with token 1 and start_audio_token
        # so the first-step penalty matches HF's RepetitionPenaltyLogitsProcessor
        # (it sees the placeholder-filled input_ids = [1, ..., 1, start_audio_token]).
        repetition_cache = None
        cfg = self.default_sampling_config
        if (
            cfg.repetition_penalty is not None
            and cfg.repetition_penalty != 1.0
            and cfg.repetition_window is not None
        ):
            repetition_cache = torch.zeros(
                cfg.repetition_window if cfg.repetition_window > 0 else 1,
                self.n_codebooks,
                self.vocab_size,
                dtype=torch.bool,
                device=device,
            )
            if 1 < self.vocab_size:
                repetition_cache[:, :, 1] = True
            start_id = int(self.args.gpt_start_audio_token)
            if 0 <= start_id < self.vocab_size:
                repetition_cache[:, :, start_id] = True

        return PreprocessOutput(
            input_tokens=input_tokens,
            repetition_cache=repetition_cache,
            input_masks=input_masks,
            input_features=input_features,
            decoder_cache=decoder_cache,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_features: torch.Tensor | None = None,
        input_masks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # ``input_features`` carries the pre-computed embedding to feed (prefix at
        # prefill; mel_emb + mel_pos at decode). Use it directly when supplied;
        # the torch.where fallback introduces fp16 drift that can flip argmax.
        if input_features is not None:
            inputs_embeds = input_features
        else:
            mel_token_ids = input_ids[:, 0].clamp(0, self.vocab_size - 1)
            inputs_embeds = self.mel_embedding(mel_token_ids)

        hidden_states = self.gpt(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        # Second LayerNorm before mel_head; this is the "latent" HiFiGAN consumes.
        hidden_states = self.final_norm(hidden_states)

        # Stash hidden_states for sampling() via a pre-allocated buffer. Python
        # attribute assignment runs only at CUDA-graph capture, so use in-place
        # GPU kernels (.copy_, .fill_) that get re-executed on every replay.
        n_rows = hidden_states.shape[0]
        self._hidden_states_buffer[:n_rows].copy_(hidden_states)
        self._last_n_rows_buffer.fill_(n_rows)

        logits = self.mel_head(hidden_states)
        return logits[:, None, :]

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        qo_indptr: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Coroutine]:
        cfg = sampling_params or self.default_sampling_config

        if repetition_cache is not None:
            logits = Sampler.apply_repetition_penalty(
                logits, repetition_cache, cfg.repetition_penalty
            )

        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=cfg)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        if repetition_cache is not None:
            Sampler.update_repetition_penalty_cache(
                repetition_cache, output_ids, cfg.repetition_window
            )

        # Recover the per-request final hidden state stashed by forward().
        # ``qo_indptr`` (cumulative token counts) lets us pick the last row of
        # each request uniformly across pure-prefill / pure-decode / mixed batches.
        n_rows = int(self._last_n_rows_buffer.item())
        h_full = self._hidden_states_buffer[:n_rows]
        if qo_indptr is not None:
            last_idx_t = (qo_indptr[1:] - 1).to(h_full.device, dtype=torch.long)
            last_idx_t = last_idx_t.clamp(max=n_rows - 1)
            last_h = h_full.index_select(0, last_idx_t).contiguous()
        elif n_rows == logits.shape[0]:
            last_h = h_full.contiguous()
        else:
            last_idx = []
            cum = 0
            for req in requests:
                seq_len = req.input_length if req.input_length else 1
                cum += seq_len
                last_idx.append(min(cum - 1, n_rows - 1))
            last_idx_t = torch.tensor(last_idx, dtype=torch.long, device=h_full.device)
            last_h = h_full.index_select(0, last_idx_t).contiguous()
        device = output_ids.device
        stop_id = self._mel_stop_token_id

        # Pre-compute the next decode step's input embedding for each request:
        # next_input = mel_embedding(sampled_id) + mel_pos_embedding(next_step), where
        # next_step = number of mel tokens generated so far AFTER this append (1-indexed from start_mel pos 0).
        for i, req in enumerate(requests):
            sampled = output_ids[i : i + 1, :]  # (1, n_codebooks=1)
            req.input_tokens = sampled

            next_step = len(req.lm_output_tokens) + 1  # +1 because async append hasn't happened yet
            sampled_id = sampled[:, 0].clamp(0, self.vocab_size - 1)
            mel_emb = self.mel_embedding(sampled_id)  # (1, hidden_size)
            mel_pos = self.mel_pos_embedding.get_fixed_embedding(next_step, device).view(1, -1)
            next_input = (mel_emb + mel_pos).detach()
            req.input_features = next_input
            req.input_masks = torch.ones(1, self.n_codebooks, dtype=torch.bool, device=device)

        async def update_req_states():
            stop_mask = output_ids[:, 0] == stop_id
            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])

                # Push a placeholder "audio token" so the worker's chunk-trigger
                # fires; postprocess reads the actual latents from decoder_cache.
                req.lm_output_audio_tokens.append(
                    torch.zeros(1, self.n_codebooks, dtype=torch.int32, device=device)
                )

                cache = getattr(req, "decoder_cache", None)
                if cache is not None and cache.accumulated_latents is not None:
                    pos = int(cache.latent_count[0, 0].item())
                    max_mel = cache.accumulated_latents.shape[1]
                    if pos < max_mel:
                        cache.accumulated_latents[0, pos].copy_(
                            last_h[i].to(
                                cache.accumulated_latents.device,
                                dtype=cache.accumulated_latents.dtype,
                            ),
                            non_blocking=True,
                        )
                        cache.latent_count[0, 0] = pos + 1

                if bool(stop_mask[i].item()):
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

            if repetition_cache is not None:
                for i, req in enumerate(requests):
                    req.repetition_cache = repetition_cache[i]

        return output_ids, update_req_states()

    @torch.inference_mode()
    def postprocess(
        self,
        _token_ids: torch.Tensor,
        decoder_cache: XTTSDecoderCache | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode the accumulated GPT latents into one chunk of PCM audio.

        For each row: run HiFiGAN on every latent accumulated so far, slice the
        new chunk from the right tail, and crossfade with the previous chunk's
        ``wav_overlap``. Variable-length input precludes CUDA-graph capture.
        """
        target_device = self.audio_decoder_device
        if decoder_cache is None or decoder_cache.accumulated_latents is None:
            raise RuntimeError(
                "XTTS postprocess requires decoder_cache with accumulated_latents.",
            )

        overlap = XTTS_OVERLAP_WAV_LEN

        if (
            not hasattr(self, "_pp_ramp_in")
            or self._pp_ramp_in.device != torch.device(target_device)
        ):
            self._pp_ramp_in = torch.linspace(
                0.0, 1.0, overlap, dtype=torch.float32, device=target_device,
            )
            self._pp_ramp_out = torch.linspace(
                1.0, 0.0, overlap, dtype=torch.float32, device=target_device,
            )
        ramp_in = self._pp_ramp_in
        ramp_out = self._pp_ramp_out

        all_acc = decoder_cache.accumulated_latents.to(
            device=target_device, dtype=torch.float32,
        )
        spk_all = decoder_cache.speaker_embedding.to(
            device=target_device, dtype=torch.float32,
        )

        B = all_acc.shape[0]
        per_row_wavs: list[torch.Tensor] = []
        for i in range(B):
            n_latents = int(decoder_cache.latent_count[i, 0].item())
            if n_latents <= 0:
                per_row_wavs.append(torch.zeros(0, device=target_device))
                continue

            latents_i = all_acc[i : i + 1, :n_latents, :]
            spk_i = spk_all[i : i + 1]

            wav_gen = self.hifigan_decoder(latents_i, g=spk_i)
            if wav_gen.dim() == 3 and wav_gen.shape[0] == 1 and wav_gen.shape[1] == 1:
                wav_gen = wav_gen.squeeze(0).squeeze(0)
            elif wav_gen.dim() == 3:
                wav_gen = wav_gen[0, 0]
            elif wav_gen.dim() == 2:
                wav_gen = wav_gen[0] if wav_gen.shape[0] == 1 else wav_gen.reshape(-1)

            cur_wav_len = wav_gen.shape[0]
            prev_wav_len = int(decoder_cache.prev_wav_len[i, 0].item())
            has_prev = bool(decoder_cache.has_prev_chunk[i, 0].item())

            if has_prev:
                start = max(prev_wav_len - overlap, 0)
                end = max(cur_wav_len - overlap, start)
                wav_chunk = wav_gen[start:end].clone()
                if wav_chunk.shape[0] >= overlap:
                    wo = decoder_cache.wav_overlap[i, 0].to(target_device, dtype=torch.float32)
                    wav_chunk[:overlap] = wo * ramp_out + wav_chunk[:overlap] * ramp_in
            else:
                end = max(cur_wav_len - overlap, 0)
                wav_chunk = wav_gen[:end].clone()

            if cur_wav_len >= overlap:
                new_wo = wav_gen[cur_wav_len - overlap : cur_wav_len]
                decoder_cache.wav_overlap[i, 0].copy_(
                    new_wo.to(decoder_cache.wav_overlap.dtype)
                )

            decoder_cache.prev_wav_len[i, 0] = int(cur_wav_len)
            decoder_cache.has_prev_chunk[i, 0].fill_(1)

            per_row_wavs.append(wav_chunk)

        # batch=1 → T equals the natural chunk length; batch>1 → zero-pad to max(T).
        T_max = max((w.shape[0] for w in per_row_wavs), default=0)
        out = torch.zeros(
            B, self.n_channels, T_max,
            dtype=torch.float32, device=target_device,
        )
        for i, w in enumerate(per_row_wavs):
            if w.shape[0] > 0:
                out[i, 0, : w.shape[0]] = w.to(out.dtype)

        return out

    def prepare_checkpoint(self, checkpoint):
        """Remap original XTTS checkpoint keys onto the XTTSModel architecture."""
        new_checkpoint = {}

        # Sub-component prefixes that move from gpt.X to X (directly on XTTSModel)
        gpt_subcomponents = (
            "gpt.conditioning_encoder.",
            "gpt.text_embedding.",
            "gpt.mel_embedding.",
            "gpt.mel_pos_embedding.",
            "gpt.text_pos_embedding.",
            "gpt.final_norm.",
            "gpt.text_head.",
            "gpt.mel_head.",
            "gpt.conditioning_perceiver.",
            "gpt.conditioning_dropout.",
        )

        for key, value in checkpoint.items():
            new_key = key

            # Step 1: Strip 'gpt.' prefix from sub-components
            for prefix in gpt_subcomponents:
                if key.startswith(prefix):
                    new_key = key[len("gpt."):]  # remove 'gpt.' prefix
                    break

            # Step 2: Collapse 'gpt.gpt.' → 'gpt.' for transformer layers
            if new_key.startswith("gpt.gpt."):
                new_key = new_key.replace("gpt.gpt.", "gpt.", 1)

            # Step 3: Split fused c_attn Conv1D → q_proj, k_proj, v_proj Linear
            if ".attn.c_attn.weight" in new_key:
                # Conv1D weight shape: (hidden, 3*hidden) → transpose → (3*hidden, hidden) → split
                base = new_key.replace(".attn.c_attn.weight", "")
                w = value.T  # (3*hidden, hidden)
                q_w, k_w, v_w = w.chunk(3, dim=0)
                new_checkpoint[f"{base}.attn.q_proj.weight"] = q_w.contiguous()
                new_checkpoint[f"{base}.attn.k_proj.weight"] = k_w.contiguous()
                new_checkpoint[f"{base}.attn.v_proj.weight"] = v_w.contiguous()
                continue
            elif ".attn.c_attn.bias" in new_key:
                # Bias shape: (3*hidden,) → split
                base = new_key.replace(".attn.c_attn.bias", "")
                q_b, k_b, v_b = value.chunk(3, dim=0)
                new_checkpoint[f"{base}.attn.q_proj.bias"] = q_b
                new_checkpoint[f"{base}.attn.k_proj.bias"] = k_b
                new_checkpoint[f"{base}.attn.v_proj.bias"] = v_b
                continue

            # Step 4: Rename c_proj → out_proj and transpose weight
            if ".attn.c_proj.weight" in new_key:
                new_key = new_key.replace(".attn.c_proj.", ".attn.out_proj.")
                value = value.T.contiguous()  # Conv1D (in, out) → Linear (out, in)
            elif ".attn.c_proj.bias" in new_key:
                new_key = new_key.replace(".attn.c_proj.", ".attn.out_proj.")

            # Step 5: Transpose MLP Conv1D weights
            elif ".mlp.c_fc.weight" in new_key or ".mlp.c_proj.weight" in new_key:
                value = value.T.contiguous()  # Conv1D (in, out) → Linear (out, in)

            new_checkpoint[new_key] = value

        return new_checkpoint
    
    def get_compatible_checkpoint_state_dict(self, model_path: Path) -> dict:
        checkpoint = self._load_checkpoint(model_path)
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def _load_checkpoint(self, model_path: Path) -> dict:
        # XTTS-v2's model.pth references ``TTS.*`` classes from the original
        # coqui-TTS package; register stub modules so unpickling works without it.
        self._register_legacy_tts_modules()
        try:
            checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))
        except (pickle.UnpicklingError, Exception):
            checkpoint = torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=False
            )
        return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    @staticmethod
    def _register_legacy_tts_modules() -> None:
        """Install a MetaPathFinder that fabricates stub ``TTS.*`` modules on demand."""
        if getattr(XTTSModel, "_legacy_tts_finder_installed", False):
            return

        import importlib
        import importlib.abc
        import importlib.machinery

        class _StubPlaceholder:
            def __init__(self, *args, **kwargs):
                self._stub_args = args
                self.__dict__.update(kwargs)

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

        class _StubModule(types.ModuleType):
            def __getattr__(self, name):
                # Dunder lookups must raise so torch's frame introspection works.
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                cls = type(name, (_StubPlaceholder,), {"__module__": self.__name__})
                setattr(self, name, cls)
                return cls

        class _LegacyTTSLoader(importlib.abc.Loader):
            def create_module(self, spec):
                m = _StubModule(spec.name)
                m.__path__ = []
                return m

            def exec_module(self, module):
                pass

        class _LegacyTTSFinder(importlib.abc.MetaPathFinder):
            _loader = _LegacyTTSLoader()

            def find_spec(self, name, path=None, target=None):
                if name == "TTS" or name.startswith("TTS."):
                    spec = importlib.machinery.ModuleSpec(name, self._loader, is_package=True)
                    spec.submodule_search_locations = []
                    return spec
                return None

        sys.meta_path.insert(0, _LegacyTTSFinder())

        # Pre-bind the dataclasses we actually want to use during unpickling.
        # Forcing the import goes through our finder and yields a _StubModule.
        cfg_mod = importlib.import_module("TTS.tts.configs.xtts_config")
        for cls in (XttsConfig, XttsArgs, XttsAudioConfig):
            setattr(cfg_mod, cls.__name__, cls)

        XTTSModel._legacy_tts_finder_installed = True
    
    def _get_cached_or_download_file(self, repo_id: str, filename: str) -> str:
        """Return a local path for ``filename``, downloading from HF Hub if absent."""
        local_dir = os.path.expanduser(self.checkpoint_path)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            return local_path
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    