"""
VibeVoice Model implementation for VoxServe

This module provides a implementation of the VibeVoice model that extends
BaseLMWithContinuousSpeech for text-to-speech generation with continuous
audio representations using cached voice prompts.
"""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Optional, Tuple, List

from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import SamplingConfig
from .base import BaseLMWithContinuousSpeech, PreprocessOutput
from ..tokenizer.base import DecoderCache
from ..utils import get_logger


class VibeVoiceModel(BaseLMWithContinuousSpeech):
    """
    VibeVoice model implementation for VoxServe.

    This model extends BaseLMWithContinuousSpeech to support text-to-speech generation
    with continuous audio representations. It uses cached voice prompts and a diffusion
    head to generate speech latents that are decoded to audio by an acoustic decoder.

    Model Architecture Flow:
        Text Input + Cached Prompt
                ↓
        VibeVoiceStreamingProcessor.process_input_with_cached_prompt()
                ↓
        BatchEncoding (input_ids, tts_lm_input_ids, tts_text_ids, speech_input_mask)
                ↓
        LM Forward (with cached KV states)
                ↓
        Hidden States → Diffusion Head → Continuous Latents
                ↓
        VibeVoiceDecoder.decode_chunk() → Audio Waveform
    """

    def __init__(
        self,
        model_name: str,
        voice_cache_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_torch_compile: bool = False,
        audio_decoder_device: Optional[str] = None,
    ):
        """
        Initialize VibeVoice model.

        Args:
            model_name: HuggingFace model identifier or local path
            voice_cache_path: Path to pre-generated voice cache file (.pt)
            device: Device to load the model on
            dtype: Data type for model tensors
            enable_torch_compile: Whether to enable torch.compile optimization
            audio_decoder_device: Device for audio decoder (defaults to main device)
        """
        # Initialize parent class
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)

        # Store configuration
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Load voice cache
        self.voice_cache = self._load_voice_cache(voice_cache_path)

        # Initialize components
        self._init_tokenizer()
        self._init_streaming_processor()
        self._init_model()
        self._init_audio_decoder()

        # Initialize model properties
        self._init_model_properties()

    def _load_voice_cache(self, voice_cache_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Load voice cache from .pt file.

        Args:
            voice_cache_path: Path to voice cache file. If None, loads default cache.

        Returns:
            Dictionary containing cached prompt hidden states with keys:
                - 'lm': Contains 'last_hidden_state' tensor
                - 'tts_lm': Contains 'last_hidden_state' tensor
        """
        if voice_cache_path is None:
            # Load default voice cache from model repo
            try:
                voice_cache_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="default_voice_cache.pt",
                    weights_only=False
                )
                self.logger.info(f"Loaded default voice cache from {voice_cache_path}")
            except Exception as e:
                self.logger.warning(f"Could not load default voice cache: {e}")
                return None

        try:
            voice_cache = torch.load(voice_cache_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Loaded voice cache from {voice_cache_path}")
            self.logger.info(f"  LM cache shape: {voice_cache['lm']['last_hidden_state'].shape}")
            self.logger.info(f"  TTS LM cache shape: {voice_cache['tts_lm']['last_hidden_state'].shape}")
            return voice_cache
        except Exception as e:
            self.logger.error(f"Failed to load voice cache from {voice_cache_path}: {e}")
            raise

    def _init_tokenizer(self):
        """Initialize the text tokenizer."""
        from transformers import AutoTokenizer

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            use_fast=True
        )

    def _init_streaming_processor(self):
        """Initialize the streaming processor."""
        from ..tokenizer.vibevoice import VibeVoiceStreamingProcessor

        self.streaming_processor = VibeVoiceStreamingProcessor.from_pretrained(
            self.model_name
        )

    def _init_model(self):
        """
        Initialize the main VibeVoice model components.

        This includes:
        - LM backbone (Qwen2.5)
        - TTS LM adapter
        - Diffusion head
        """
        # Placeholder for model initialization
        # The actual implementation would load pre-trained weights
        self.lm_backbone = nn.Identity().to(self.device).to(self.dtype)
        self.tts_lm_adapter = nn.Identity().to(self.device).to(self.dtype)
        self.diffusion_head = nn.Identity().to(self.device).to(self.dtype)

    def _init_audio_decoder(self):
        """Initialize the acoustic decoder."""
        from ..tokenizer.vibevoice_acoustic import VibeVoiceAcousticTokenizer

        # Initialize acoustic tokenizer (which includes the decoder)
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizer(
            config=None,  # Will be loaded from model config
            device=self.audio_decoder_device
        )
        self.audio_decoder = self.acoustic_tokenizer

    def _init_model_properties(self):
        """Initialize model properties based on configuration."""
        # Set basic properties from model config
        self._num_attention_heads = 12  # Default for Qwen2.5-1.5B
        self._num_key_value_heads = 2
        self._num_hidden_layers = 28
        self._hidden_size = 1536
        self._vocab_size = 151936
        self._latent_dim = 64  # VibeVoice latent dimension

        # Audio properties
        self._n_codebooks = 1  # Using continuous latents
        self._n_channels = 1  # Mono audio
        self._output_audio_length = 24000  # 1 second at 24kHz

        # Token IDs
        self._stop_token_id = self.text_tokenizer.eos_token_id

        # Default sampling config
        self.default_sampling_config = SamplingConfig(
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05
        )

    # Properties
    @property
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        return self._n_codebooks

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
    def latent_dim(self) -> int:
        """Dimension of the continuous speech latents."""
        return self._latent_dim

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        return self._vocab_size

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the model generates in a single request."""
        return 2048

    @property
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        return self._n_channels

    @property
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        return self._output_audio_length

    @property
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        return 1  # Continuous generation, detokenize each step

    @property
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        return 0

    @property
    def needs_input_features(self) -> bool:
        """Indicates if the model requires input_features."""
        return True

    @property
    def needs_input_masks(self) -> bool:
        """Indicates if the model requires input_masks."""
        return True

    def is_stop_id(self, token_ids: List[int]) -> bool:
        """Check if the given token ID is a stop token."""
        return token_ids[0] == self._stop_token_id

    def preprocess(
        self,
        prompt: str = None,
        audio_path: str = None,
        voice_cache_path: Optional[str] = None,
        **kwargs,
    ) -> PreprocessOutput:
        """
        Preprocess input text and voice cache for TTS generation.

        Args:
            prompt: Input text for TTS generation
            audio_path: Optional path to audio file (not used with cached voice)
            voice_cache_path: Optional path to voice cache file (overrides default)
            **kwargs: Additional parameters

        Returns:
            PreprocessOutput with preprocessed inputs including:
                - input_tokens: Combined pseudo IDs + text tokens
                - input_features: Cached hidden states
                - input_masks: Speech/text masks
                - decoder_cache: Initial VibeVoiceDecoderCache
        """
        from ..tokenizer.vibevoice import VibeVoiceStreamingProcessor

        # Load voice cache if path provided
        if voice_cache_path is not None:
            self.voice_cache = self._load_voice_cache(voice_cache_path)

        # Verify voice cache is available
        # TODO: Handling the case where the voice cache is strictly required.
        # Reuse the _process_single implementation in the vibevoice/processor/vibevoice_processor.py
        # Handle this in teh tokenizer
        if self.voice_cache is None:
            raise ValueError(
                "Voice cache is required for VibeVoice preprocessing. "
                "Provide voice_cache_path or ensure default cache is available."
            )

        # Use streaming processor with cached prompt
        batch_encoding = self.streaming_processor.process_input_with_cached_prompt(
            text=prompt,
            cached_prompt=self.voice_cache,
            return_tensors="pt"
        )

        # Convert to PreprocessOutput format
        # Extract input_ids from batch encoding
        input_ids = batch_encoding["input_ids"]  # [1, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Convert to list of lists for PreprocessOutput
        input_tokens_list = input_ids.tolist()

        # Create input masks from speech_input_mask
        input_masks = batch_encoding.get("speech_input_mask", None)
        if input_masks is not None:
            # Convert boolean mask to float tensor for compatibility
            input_masks = input_masks.float().to(self.device)

        # Create input features from cached hidden states
        input_features = self.voice_cache["lm"]["last_hidden_state"].to(self.device)

        # Create repetition cache if needed
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

        # Initialize decoder cache
        decoder_cache = self.audio_decoder_initial_cache(batch_size=1)

        return PreprocessOutput(
            input_tokens=input_tokens_list,
            input_masks=input_masks,
            input_features=input_features,
            repetition_cache=repetition_cache,
            decoder_cache=decoder_cache,
        )

    def diffusion_forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through diffusion head to generate speech latents.

        Args:
            hidden_states: Hidden states from backbone model. Shape: (batch_size, seq_len, hidden_size)
            **kwargs: Additional model-specific parameters

        Returns:
            Continuous speech latents. Shape: (batch_size, seq_len, latent_dim)
        """
        # Placeholder implementation
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        # Generate placeholder latents
        latents = torch.randn(
            batch_size, seq_len, self._latent_dim,
            device=self.device, dtype=self.dtype
        )

        return latents

    def postprocess(
        self,
        latents: torch.Tensor,
        decoder_cache: Optional[DecoderCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert continuous latents to audio waveform using acoustic decoder.

        Args:
            latents: Continuous speech latents from diffusion head. Shape: (batch_size, seq_len, latent_dim)
            decoder_cache: Optional decoder cache for acoustic decoder
            **kwargs: Additional model-specific parameters

        Returns:
            Audio tensor. Shape: (batch_size, n_channels, audio_length)
        """
        # Placeholder implementation
        batch_size = latents.size(0)

        # Create dummy audio
        audio = torch.randn(
            batch_size, self.n_channels, self._output_audio_length,
            device=self.audio_decoder_device, dtype=self.dtype
        )

        return audio

    def audio_decoder_initial_cache(self, batch_size: int) -> Optional[DecoderCache]:
        """
        Initialize audio decoder cache for streaming generation.

        Args:
            batch_size: Desired batch size for the cache instance

        Returns:
            VibeVoiceDecoderCache initialized for the given batch size
        """
        from ..tokenizer.vibevoice_acoustic import init_vibevoice_decoder_cache

        return init_vibevoice_decoder_cache(
            acoustic_tokenizer=self.acoustic_tokenizer,
            batch_size=batch_size,
            device=self.audio_decoder_device,
            dtype=self.dtype
        )
