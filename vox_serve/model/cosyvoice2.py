import json
from dataclasses import dataclass, field
from functools import lru_cache, partial
import os
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from ..flashinfer_utils import FlashInferWrapper, apply_rope_pos_ids, rms_norm
from ..requests import Request
from ..sampling import Sampler, SamplingConfig
from ..tokenizer.s3 import S3TokenizerV2
from ..tokenizer.cosyvoice2 import CosyVoice2Decoder
from ..utils import get_logger, load_hf_safetensor_state_dict
from .base import BaseLM, PreprocessOutput
from librosa.filters import mel as librosa_mel_fn


@dataclass
class CosyVoice2Config:
    llm_input_size = 896
    llm_output_size = 896
    speech_token_size = 6561
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    num_hidden_layers: int = 24
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-06


class CosyVoice2Tokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path, subfolder="CosyVoice-BlankEN")
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


class CosyVoice2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        CosyVoice2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(
            hidden_states=hidden_states,
            weight=self.weight,
            eps=self.variance_epsilon,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CosyVoice2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return output


class CosyVoice2Attention(nn.Module):
    def __init__(self, config: CosyVoice2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.rope_theta = config.rope_theta

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        key_states = self.k_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape)  # .transpose(0, 1)

        query_states, key_states = apply_rope_pos_ids(
            query_states=query_states,
            key_states=key_states,
            position_ids=position_ids,
            # rotary_dim=self.head_dim // 2,
            # interleave=True,
            rope_theta=self.rope_theta,
        )

        attn_wrapper.set_kv_cache(kv_cache, key_states, value_states)
        attn_output = attn_wrapper.run(query_states, kv_cache)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class CosyVoice2DecoderLayer(nn.Module):
    def __init__(self, config: CosyVoice2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CosyVoice2Attention(config=config, layer_idx=layer_idx)
        self.mlp = CosyVoice2MLP(config)

        self.input_layernorm = CosyVoice2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CosyVoice2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CosyVoice2BackboneModel(nn.Module):
    def __init__(self, config: CosyVoice2Config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            CosyVoice2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CosyVoice2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_wrapper=attn_wrapper,
                kv_cache=kv_cache[i],
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class CosyVoice2LLM2(nn.Module):
    # for compatibility with checkpoint naming
    def __init__(self, config: CosyVoice2Config):
        super().__init__()
        self.model = CosyVoice2BackboneModel(config)
        # unused
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )


class CosyVoice2LLM(nn.Module):
    def __init__(self, config: CosyVoice2Config):
        super().__init__()
        self.model = CosyVoice2LLM2(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )


class CosyVoice2ForCausalLM(nn.Module):
    def __init__(self, config: CosyVoice2Config):
        super().__init__()
        self.llm = CosyVoice2LLM(config)
        self.llm_embedding = nn.Embedding(2, config.llm_input_size)
        self.llm_decoder = nn.Linear(config.llm_output_size, config.speech_token_size + 3)
        self.speech_embedding = nn.Embedding(config.speech_token_size + 3, config.llm_input_size)
        self.vocab_size = config.vocab_size

    def embed_tokens(self, input_ids):
        return self.llm_embedding(input_ids)

    def embed_tokens_speech(self, input_ids):
        return self.speech_embedding(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
    ):
        hidden_states = self.llm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        logits = self.llm_decoder(hidden_states)

        return logits


class CosyVoice2Model(BaseLM):
    def __init__(self, model_name, dtype=torch.bfloat16, device="cuda:0", enable_torch_compile=False):
        if model_name == "cosyvoice2":
            model_name = "FunAudioLLM/CosyVoice2-0.5B"
        super().__init__(model_name, device, dtype, enable_torch_compile)
        self.logger = get_logger(__name__)
        # fixed config for now
        self.config = CosyVoice2Config()

        # Use provided tokenizer path or default to model_name
        self.text_tokenizer = self._load_tokenizer(model_name)

        self.sos = 0
        self.task_id = 1
        self.eos_token = self.config.speech_token_size
        self.fill_token = self.config.speech_token_size + 2

        self.model = CosyVoice2ForCausalLM(self.config)
        self.model.load_state_dict(
            torch.load(
                hf_hub_download(repo_id=model_name, filename="llm.pt", revision=None),
                map_location="cpu", 
                weights_only=True
            ),
            strict=True,
        )
        self.model.to(dtype).to(device)

        audio_tokenizer_path = hf_hub_download(
            repo_id=model_name,
            filename="speech_tokenizer_v2.onnx",
            revision=None,
        )
        self.audio_tokenizer = S3TokenizerV2(audio_tokenizer_path).to(device)

        spk_model_path = hf_hub_download(
            repo_id=model_name,
            filename="campplus.onnx",
            revision=None,
        )
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_model = onnxruntime.InferenceSession(
            spk_model_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

        self.audio_decoder = CosyVoice2Decoder(
            model_path=model_name,
            device=device,
        )
        self.audio_decoder.to(device)
        # self._audio_decoder_initial_cache = self.audio_decoder.init_cache()

        self._num_attention_heads = self.config.num_attention_heads
        self._num_key_value_heads = self.config.num_key_value_heads
        self._num_hidden_layers = self.config.num_hidden_layers
        self._hidden_size = self.config.hidden_size

        self.stop_token_ids = [self.config.speech_token_size + i for i in range(3)]
        
        # Initialize mel basis and hann window caches
        self.mel_basis = {}
        self.hann_window = {}

        self.default_sampling_config = SamplingConfig(
            top_k=None,
            top_p=0.9,
            min_p=None,
            temperature=0.7,
            repetition_penalty=1.05,
            repetition_window=-1,
            cfg_scale=None,
        )

        self._init_default_cond()

    @property
    def n_codebooks(self):
        """Number of codebooks in the model."""
        return 1

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        return self._num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        return self._num_key_value_heads

    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        return self._num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Hidden size of the model."""
        return self._hidden_size

    @property
    def supports_audio_input(self) -> bool:
        """Indicates if the model accepts audio input."""
        return True

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return True

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return True

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 28

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 3

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return 1  # Mono audio

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return 24000

    @property
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        if self.default_sampling_config.max_tokens is not None:
            return self.default_sampling_config.max_tokens
        return 4096

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self.config.speech_token_size + 3

    def is_stop_id(self, token_ids: List[int]) -> bool:
        return token_ids[0] in self.stop_token_ids

    def _load_tokenizer(self, tokenizer_path):
        self.allowed_special = "all"
        return CosyVoice2Tokenizer(tokenizer_path)
    
    def _audio_feat_extractor(self, y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False):
        """
        Mel-spectrogram feature extraction function.
        """
        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        # Use instance variables for caching
        if f"{str(fmax)}_{str(y.device)}" not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
        )
        y = y.squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=self.hann_window[str(y.device)],
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis[str(fmax) + "_" + str(y.device)], spec)
        spec = self.spectral_normalize_torch(spec)

        return spec
    
    def spectral_normalize_torch(self, magnitudes):
        """Dynamic range compression for spectral normalization."""
        return torch.log(torch.clamp(magnitudes, min=1e-5))

    def _spell_out_number(self, text: str, inflect_parser):
        new_text = []
        st = None
        for i, c in enumerate(text):
            if not c.isdigit():
                if st is not None:
                    num_str = inflect_parser.number_to_words(text[st: i])
                    new_text.append(num_str)
                    st = None
                new_text.append(c)
            else:
                if st is None:
                    st = i
        if st is not None and st < len(text):
            num_str = inflect_parser.number_to_words(text[st:])
            new_text.append(num_str)
        return ''.join(new_text)

    def _split_paragraph(self, text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
        def calc_utt_length(_text: str):
            if lang == "zh":
                return len(_text)
            else:
                return len(tokenize(_text))

        def should_merge(_text: str):
            if lang == "zh":
                return len(_text) < merge_len
            else:
                return len(tokenize(_text)) < merge_len

        if lang == "zh":
            pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
        else:
            pounc = ['.', '?', '!', ';', ':']
        if comma_split:
            pounc.extend(['，', ','])

        if text[-1] not in pounc:
            if lang == "zh":
                text += "。"
            else:
                text += "."

        st = 0
        utts = []
        for i, c in enumerate(text):
            if c in pounc:
                if len(text[st: i]) > 0:
                    utts.append(text[st: i] + c)
                if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                    tmp = utts.pop(-1)
                    utts.append(tmp + text[i + 1])
                    st = i + 2
                else:
                    st = i + 1

        final_utts = []
        cur_utt = ""
        for utt in utts:
            if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
                final_utts.append(cur_utt)
                cur_utt = ""
            cur_utt = cur_utt + utt
        if len(cur_utt) > 0:
            if should_merge(cur_utt) and len(final_utts) != 0:
                final_utts[-1] = final_utts[-1] + cur_utt
            else:
                final_utts.append(cur_utt)

        return final_utts
    
    def _is_only_punctuation(self, text):
        import regex
        # Regular expression: Match strings that consist only of punctuation marks or are empty.
        punctuation_pattern = r'^[\p{P}\p{S}]*$'
        return bool(regex.fullmatch(punctuation_pattern, text))

    def _text_normalize(self, text, split=False, text_frontend=True):
        import inflect
        self.inflect_parser = inflect.engine()
        # NOTE skip text_frontend when ssml symbol in text
        if '<|' in text and '|>' in text:
            text_frontend = False
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        # TODO: simplified normalization for now
        text = self._spell_out_number(text, self.inflect_parser)
        texts = list(self._split_paragraph(text, partial(self.text_tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                        token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not self._is_only_punctuation(i)]
        return texts if split is True else text
    
    def _extract_text_token(self, text):
        text_token = self.text_tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        return text_token
    
    def _extract_speech_feat(self, prompt_wav):
        speech, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != 24000:
            assert sample_rate >= 16000, 'wav sample rate {} must be greater than {}'.format(sample_rate, 16000)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(speech)
        speech_feat = self._audio_feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len
    
    def _log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        n_mels: int = 80,
        padding: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        (from openai/whisper)
        Compute the log-Mel spectrogram
        """
        N_FFT = 400
        HOP_LENGTH = 160
        if device is not None:
            audio = audio.to(device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = torch.from_numpy(
            librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
        ).to(device)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    def _extract_speech_token(self, prompt_wav):
        speech, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            assert sample_rate >= 16000, 'wav sample rate {} must be greater than {}'.format(sample_rate, 16000)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = self._log_mel_spectrogram(speech, n_mels=128).to(self.device)
        speech_token, speech_token_len = self.audio_tokenizer.quantize(
            feat,
            torch.tensor([feat.shape[2]], dtype=torch.int32, device=self.device)
        )
        return speech_token, speech_token_len
    
    def _extract_spk_embedding(self, prompt_wav):
        import torchaudio.compliance.kaldi as kaldi
        speech, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            assert sample_rate >= 16000, 'wav sample rate {} must be greater than {}'.format(sample_rate, 16000)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
        
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_model.run(
            None,
            {self.campplus_model.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding
    
    def _init_default_cond(self):
        ref_audio_path = '/home/keisuke/vox-serve/audiobench.mp3'
        ref_audio_text = 'How long does it take our eyes to fully adapt to the darkness?'
        
        ref_text_ids = self._extract_text_token(ref_audio_text)
        speech_feat, speech_feat_len = self._extract_speech_feat(ref_audio_path)
        speech_token, speech_token_len = self._extract_speech_token(ref_audio_path)
        embedding = self._extract_spk_embedding(ref_audio_path)

        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len = speech_token[:, :token_len], token_len

        # print(f"{ref_text_ids.shape=} {ref_text_ids=}") # [1, 14] ref_text_ids=tensor([[ 4340,  1293,  1558,   432,  1896,  1039,  6414,   311,  7225, 10515, 311,   279, 26298,    30]]
        # print(f"{speech_feat.shape=} {speech_feat=}") # [1, 258, 80]
        # print(f"{speech_feat_len=}") # 258
        # print(f"{speech_token.shape=} {speech_token=}") # [1, 129]
        # print(f"{speech_token_len=}") # 129
        # print(f"{embedding.shape=} {embedding=}") # [1, 192]

        self.default_speaker_ref_dict = {
            'ref_text_ids': ref_text_ids,
            'prompt_feat': speech_feat,
            'prompt_feat_len': speech_feat_len,
            'prompt_speech_token': speech_token,
            'prompt_speech_token_len': speech_token_len,
            'embedding': embedding,
        }

    def preprocess(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
    ) -> PreprocessOutput:
        assert audio_path is None, "audio_path is not supported yet for this model"

        prompt = self._text_normalize(prompt, split=False, text_frontend=True)
        prompt_ids = self._extract_text_token(prompt)

        input_ids = torch.concat([
            torch.tensor([[self.sos]], dtype=torch.int32, device=self.device),
            self.default_speaker_ref_dict['ref_text_ids'],
            prompt_ids,
            torch.tensor([[self.task_id]], dtype=torch.int32, device=self.device),
            self.default_speaker_ref_dict['prompt_speech_token'],
        ], dim=1).view(-1, 1) # add codebook dimension

        input_features = torch.zeros(
            input_ids.shape[0],
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        input_masks = torch.zeros(
            input_ids.shape[0],
            self.n_codebooks,
            device=self.device,
            dtype=torch.bool,
        )
        input_masks[-self.default_speaker_ref_dict['prompt_speech_token_len']:, :] = 1
        print(f"{input_ids.shape=}, {input_features.shape=}, {input_masks.shape=}")

        # Create repetition cache if repetition penalty is enabled
        repetition_cache = None
        config = self.default_sampling_config
        if (config.repetition_penalty is not None and
            config.repetition_window is not None and
            config.repetition_penalty != 1.0):
            repetition_cache = torch.zeros(
                config.repetition_window if config.repetition_window > 0 else 1,
                self.n_codebooks,
                self.vocab_size,
                dtype=torch.bool,
                device=self.device,
            )

        # Initial decoder cache for this request (default cache), sized for batch 1
        # decoder_cache = self.audio_decoder_initial_cache(batch_size=1)

        return PreprocessOutput(
            input_tokens=input_ids,
            repetition_cache=repetition_cache,
            input_masks=input_masks,
            input_features=input_features,
            # decoder_cache=decoder_cache,
        )

    # def audio_decoder_initial_cache(self, batch_size: int):
    #     base_cache = self._audio_decoder_initial_cache

    #     # Default: clone per-tensor to create an independent cache instance
    #     return CosyVoice2DecoderCache(
    #         spk_emb=base_cache.spk_emb.repeat(batch_size, 1),
    #         conformer_cnn_cache=base_cache.conformer_cnn_cache.repeat(batch_size, 1, 1),
    #         conformer_att_cache=base_cache.conformer_att_cache.repeat(batch_size, 1, 1, 1, 1),
    #         # estimator_cnn_cache=base_cache.estimator_cnn_cache.repeat(batch_size, 1, 1, 1, 1, 1),
    #         # estimator_att_cache=base_cache.estimator_att_cache.repeat(batch_size, 1, 1, 1, 1, 1, 1),
    #         hift_mel_cache=base_cache.hift_mel_cache.repeat(batch_size, 1, 1),
    #         hift_source_cache=base_cache.hift_source_cache.repeat(batch_size, 1, 1),
    #         hift_speech_cache=base_cache.hift_speech_cache.repeat(batch_size, 1),
    #     )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_masks: torch.Tensor,
        input_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # remove codebook dimension
        inputs_embeds = self.model.embed_tokens(input_ids[:, 0].clamp(0, self.model.llm_embedding.weight.shape[0] - 1))
        speech_embeds = self.model.embed_tokens_speech(input_ids[:, 0].clamp(0, self.model.speech_embedding.weight.shape[0] - 1))

        inputs_embeds = torch.where(input_masks, input_features, speech_embeds)

        logits = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )

        return logits[:, None, :]  # add codebook dimension

    def sampling(
        self,
        logits: torch.Tensor,
        requests: List[Request],
        sampling_params: SamplingConfig | None = None,
        repetition_cache: torch.Tensor | None = None,
        cfg_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if sampling_params is None:
            sampling_params = self.default_sampling_config

        if repetition_cache is not None:
            logits = Sampler.apply_repetition_penalty(
                logits, repetition_cache, sampling_params.repetition_penalty
            )

        output_ids = Sampler.run_sampling(logits.view(-1, self.vocab_size), config=sampling_params)
        output_ids = output_ids.view(logits.shape[0], logits.shape[1])

        if repetition_cache is not None:
            Sampler.update_repetition_penalty_cache(
                repetition_cache,
                output_ids,
                sampling_params.repetition_window,
            )

        for i, req in enumerate(requests):
            req.input_tokens = output_ids[i : i + 1]

            req.input_features = req.input_features[:1].zero_()
            req.input_masks = req.input_masks[:1].zero_()

        async def update_req_states():
            stop_mask = (output_ids[:, 0] == self.stop_token_ids[0]) | \
                        (output_ids[:, 0] == self.stop_token_ids[1]) | \
                        (output_ids[:, 0] == self.stop_token_ids[2])

            for i, req in enumerate(requests):
                req.lm_output_tokens.append(output_ids[i : i + 1])
                if not stop_mask[i]:
                    req.lm_output_audio_tokens.append(output_ids[i : i + 1])
                if stop_mask[i]:
                    req.done_lm_generation = True
                    req.finish_reason = "stop_id_encountered"
                if req.next_position_id > self.max_tokens:
                    req.done_lm_generation = True
                    req.finish_reason = "max_tokens_reached"

            if repetition_cache is not None:
                # Update repetition cache in requests
                for i, req in enumerate(requests):
                    req.repetition_cache = repetition_cache[i]

        task = update_req_states()

        return output_ids, task

    def postprocess(self, token_ids: torch.Tensor):
        audio_tensor = self.audio_decoder.decode(
            token_ids[:, :, 0],
            speech_token_lens=self.detokenize_interval,
            ref_dict=self.default_speaker_ref_dict,
        )
        # decoder_cache.spk_emb[:] = new_decoder_cache.spk_emb
        # decoder_cache.conformer_cnn_cache[:] = new_decoder_cache.conformer_cnn_cache
        # decoder_cache.conformer_att_cache[:] = new_decoder_cache.conformer_att_cache
        # # decoder_cache.estimator_cnn_cache[:] = new_decoder_cache.estimator_cnn_cache
        # # decoder_cache.estimator_att_cache[:] = new_decoder_cache.estimator_att_cache
        # decoder_cache.hift_mel_cache[:] = new_decoder_cache.hift_mel_cache
        # decoder_cache.hift_source_cache[:] = new_decoder_cache.hift_source_cache
        # decoder_cache.hift_speech_cache[:] = new_decoder_cache.hift_speech_cache
        return audio_tensor[:, None, :]  # add channel dimension

if __name__ == "__main__":
    model = CosyVoice2Model("FunAudioLLM/CosyVoice2-0.5B")
