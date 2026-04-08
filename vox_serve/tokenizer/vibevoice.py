import json
import math
import os
import re
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)

@dataclass
class VibeVoicePreprocessorConfig:
    processor_class: str = "VibeVoiceProcessor"
    speech_tok_compress_ratio: int = 3200
    db_normalize: bool = True
    language_model_pretrained_name: str = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        language_model_pretrained_name = kwargs.pop(
            "language_model_pretrained_name", 
            "Qwen/Qwen2.5-1.5B"  # VibeVoice default
        )
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
            language_model_pretrained_name, **kwargs
        )
        return cls(
            tokenizer=tokenizer,
            audio_processor=None,
            speech_tok_compress_ratio=3200,  # unused at inference
            db_normalize=True,               # unused at inference
        )

class VibeVoiceTextTokenizerFast(Qwen2TokenizerFast):
    """
    Construct a "fast" VibeVoice tokenizer (backed by HuggingFace's *tokenizers* library).
    Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
                "<|image_pad|>"
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")

        # self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')
        self._eos_id = self.eos_token_id # qwen2 / qwen3
        self._pad_id = self.convert_tokens_to_ids('<|image_pad|>')
        
        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return self._pad_id

class VibeVoiceStreamingProcessor:
    r"""
    Constructs a VibeVoice Streaming processor which wraps a VibeVoice tokenizer and audio processor into a single processor.

    Args:
        tokenizer (`VibeVoiceTextTokenizer` or `VibeVoiceTextTokenizerFast`):
            The tokenizer for text processing.
        audio_processor (`VibeVoiceTokenizerProcessor`):
            The audio processor for speech processing.
        speech_tok_compress_ratio (`int`, *optional*, defaults to 3200):
            The compression ratio for speech tokenization.
        db_normalize (`bool`, *optional*, defaults to True):
            Whether to apply decibel normalization to audio inputs.
    """

    def __init__(self, tokenizer=None, audio_processor=None, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = None  # Will initialize AudioNormalizer when needed

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibeVoiceStreamingProcessor from a pretrained VibeVoice Streaming processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, *model id* of a pretrained model
                - a path to a *directory* containing processor config

        Returns:
            [`VibeVoiceStreamingProcessor`]: The processor object instantiated from pretrained model.
        """
        # Load configuration using VibeVoicePreprocessorConfig
        config = VibeVoicePreprocessorConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Extract main processor parameters from config
        speech_tok_compress_ratio = config.speech_tok_compress_ratio
        db_normalize = config.db_normalize
        
        # Load tokenizer from config, allowing override via kwargs
        language_model_pretrained_name = kwargs.pop("language_model_pretrained_name", config.language_model_pretrained_name)
        logger.info(f"Loading tokenizer from {language_model_pretrained_name}")
        
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
            language_model_pretrained_name,
            **kwargs
        )
        
        # Audio processor - will be None if not provided
        audio_processor = None
        
        # Create and return processor
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )

    # TODO: cached_prompt is strictly required here.
    # Reuse the _process_single implementation in the vibevoice/processor/vibevoice_processor.py
    # for scenarios where cached_prompt is not provided.
    def process_input_with_cached_prompt(
        self,
        text: Optional[str] = None,
        cached_prompt: Optional[Dict[str, Any]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to process one text script based on cached prompt. The function currently only supports single examples.

        Args:
            text (`str`):
                The input text to process.
            cached_prompt (`Dict[str, Any]`, *optional*):
                The cached prompt to use for processing. It contains kv cache of voice prompt.
            padding (`bool`, `str` or `PaddingStrategy`, defaults to `True`):
                Whether to pad sequences to same length
            truncation (`bool`, `str` or `TruncationStrategy`, defaults to `False`):
                Whether to truncate sequences
            max_length (`int`, *optional*):
                Maximum length of returned sequences
            return_tensors (`str` or `TensorType`, *optional*):
                If set, will return tensors of a particular framework
            return_attention_mask (`bool`, defaults to `True`):
                Whether to return attention mask

        Returns:
            `BatchEncoding`: A BatchEncoding with following fields:
                - **input_ids** -- List of token id sequences or tensor
                - **attention_mask** -- List of attention masks or tensor
                - **tts_lm_input_ids** -- List of token id sequences or tensor used for TTS LM
                - **tts_lm_attention_mask** -- List of attention masks or tensor used for TTS LM
                - **tts_text_ids** -- List of token id sequences or tensor for TTS text input
                - **speech_tensors** -- Padded speech inputs (if voice_samples provided)
                - **speech_masks** -- Speech masks (if voice_samples provided)
                - **speech_input_mask** -- Boolean masks indicating speech token positions
        """
        # Only support single example
        texts = [text]
        cached_prompts = [cached_prompt]
        is_batched = False
        
        # Process each input
        all_encodings = []
        for text_input, cached_prompt_input in zip(texts, cached_prompts):
            script_tokens = self.tokenizer.encode(text_input.strip() + "\n", add_special_tokens=False)
            input_id_length = cached_prompt_input['lm']['last_hidden_state'].size(1)
            tts_lm_input_id_length = cached_prompt_input['tts_lm']['last_hidden_state'].size(1)

            # psudo input ids and masks
            input_ids = [self.tokenizer.pad_id] * input_id_length
            tts_lm_input_ids = [self.tokenizer.pad_id] * tts_lm_input_id_length
            speech_input_mask = [False] * tts_lm_input_id_length

            encoding = {
                        "input_ids": input_ids,
                        "tts_lm_input_ids": tts_lm_input_ids,
                        "tts_text_ids": script_tokens,
                        "speech_inputs": None,
                        "speech_input_mask": speech_input_mask,
                    }
            all_encodings.append(encoding)
            
        # Combine batch
        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )
        
        return batch_encoding
    
    def _batch_encode(
        self,
        encodings: List[Dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
    ) -> BatchEncoding:
        """Combine multiple encodings into a batch with padding."""
        # Extract input_ids and create attention_mask
        input_ids_list = [enc["input_ids"] for enc in encodings]
        tts_lm_input_ids_list = [enc["tts_lm_input_ids"] for enc in encodings]
        tts_text_ids_list = [enc["tts_text_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]
        
        attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None
        tts_lm_attention_masks = [[1] * len(ids) for ids in tts_lm_input_ids_list] if return_attention_mask else None
            
        # Process speech inputs
        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True
                
        # Prepare batch encoding
        batch_encoding = BatchEncoding()
        
        # Handle tensor conversion
        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            batch_encoding["tts_lm_input_ids"] = torch.tensor(tts_lm_input_ids_list, dtype=torch.long)
            batch_encoding["tts_text_ids"] = torch.tensor(tts_text_ids_list, dtype=torch.long)

            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
                batch_encoding["tts_lm_attention_mask"] = torch.tensor(tts_lm_attention_masks, dtype=torch.long)
            
            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            batch_encoding["tts_lm_input_ids"] = tts_lm_input_ids_list
            batch_encoding["tts_text_ids"] = tts_text_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
                batch_encoding["tts_lm_attention_mask"] = tts_lm_attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list
            
        # Process speech tensors if present
        # NOTE: No speech processing is done in this version. If desired, add prepare_speech_inputs method here.
        batch_encoding["speech_tensors"] = None
        batch_encoding["speech_masks"] = None
            
        return batch_encoding

    @property
    def model_input_names(self):
        """
        Return list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names if self.audio_processor else []
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names + ["speech_inputs", "speech_input_mask"]))
    

__all__ = [
    "VibeVoicePreprocessorConfig",
    "VibeVoiceTextTokenizerFast",
    "VibeVoiceStreamingProcessor",
]