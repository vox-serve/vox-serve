"""
VibeVoice Model implementation for VoxServe

This module provides a implementation of the VibeVoice model that extends
BaseLMWithContinuousSpeech for text-to-speech generation with continuous
audio representations using cached voice prompts.
"""

from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from ..flashinfer_utils import FlashInferWrapper
from ..requests import Request
from ..sampling import SamplingConfig
from .base import BaseLMWithContinuousSpeech, PreprocessOutput
from ..tokenizer.base import DecoderCache
from ..utils import get_logger


class BinaryClassifier(nn.Module):
    """Binary classifier for EOS prediction."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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
            repo_root = Path(__file__).resolve().parents[3]
            voice_cache_path = repo_root / "vox-serve/demo/voices/de-Spk0_man.pt"
        else:
            voice_cache_path = Path(voice_cache_path)

        voice_cache_path = voice_cache_path.expanduser().resolve()
        if voice_cache_path.suffix.lower() != ".pt":
            raise ValueError(
                f"Voice cache must be a .pt file, got: {voice_cache_path}"
            )
        if not voice_cache_path.exists():
            raise FileNotFoundError(f"Voice cache not found: {voice_cache_path}")

        try:
            voice_cache = torch.load(str(voice_cache_path), map_location=self.device, weights_only=False)
        except Exception as e:
            self.logger.error(f"Failed to load voice cache from {voice_cache_path}: {e}")
            raise

        if not isinstance(voice_cache, dict):
            raise ValueError(f"Voice cache must be a dictionary, got {type(voice_cache).__name__}")

        for key in ("lm", "tts_lm"):
            if key not in voice_cache:
                raise ValueError(f"Voice cache missing required key: '{key}'")
            if not isinstance(voice_cache[key], dict):
                raise ValueError(f"Voice cache key '{key}' must be a dictionary")
            if "last_hidden_state" not in voice_cache[key]:
                raise ValueError(f"Voice cache key '{key}' missing 'last_hidden_state'")
            if not isinstance(voice_cache[key]["last_hidden_state"], torch.Tensor):
                raise ValueError(f"Voice cache key '{key}.last_hidden_state' must be a torch.Tensor")

        self.logger.info(f"Loaded voice cache from {voice_cache_path}")
        self.logger.info(f"  LM cache shape: {voice_cache['lm']['last_hidden_state'].shape}")
        self.logger.info(f"  TTS LM cache shape: {voice_cache['tts_lm']['last_hidden_state'].shape}")
        return voice_cache

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
        - Language backbone (lower Qwen2 layers for text encoding)
        - TTS backbone (upper Qwen2 layers for speech generation)
        """
        from .qwen2 import load_vibevoice_qwen2_split_from_hf

        # Load the split Qwen2 models from HuggingFace
        self.language_backbone, self.tts_backbone = load_vibevoice_qwen2_split_from_hf(
            repo_id=self.model_name,
            tts_backbone_num_hidden_layers=20,
            device=self.device,
            dtype=self.dtype,
        )

        # The language backbone's final norm is not used during inference
        # (the hidden states are passed directly to the TTS backbone)
        self.language_backbone.norm = nn.Identity()

        # Store split metadata for reference
        self.split_metadata = self.language_backbone.split_metadata
        
        # Add type embedding for text/speech distinction in TTS LM
        self.tts_input_types = nn.Embedding(
            num_embeddings=2, 
            embedding_dim=self._hidden_size
        )
        
        # Add EOS classifier for TTS generation
        self.tts_eos_classifier = BinaryClassifier(self._hidden_size)
        
        # Load TTS component weights from HuggingFace checkpoint
        self._load_tts_component_weights()

    def _load_tts_component_weights(self):
        """
        Load TTS-specific component weights from HuggingFace checkpoint.
        
        Loads weights for:
            - tts_input_types: Embedding for text/speech type distinction
            - tts_eos_classifier: Binary classifier for EOS prediction
        
        Expected keys in the state dict:
            - model.tts_input_types.weight
            - tts_eos_classifier.fc1.weight
            - tts_eos_classifier.fc1.bias
            - tts_eos_classifier.fc2.weight
            - tts_eos_classifier.fc2.bias
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        try:
            # Download the model safetensors index
            index_path = hf_hub_download(
                repo_id=self.model_name,
                filename="model.safetensors.index.json",
                revision=None
            )
            
            with open(index_path, "r", encoding="utf-8") as f:
                import json
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            
            # Keys for TTS components
            tts_keys = [
                "model.tts_input_types.weight",
                "tts_eos_classifier.fc1.weight",
                "tts_eos_classifier.fc1.bias",
                "tts_eos_classifier.fc2.weight",
                "tts_eos_classifier.fc2.bias",
            ]
            
            # Find shards containing TTS component weights
            shards_needed = set()
            for key in tts_keys:
                if key in weight_map:
                    shards_needed.add(weight_map[key])
            
            # Load weights from each shard
            tts_state_dict = {}
            for shard in shards_needed:
                shard_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=shard,
                    revision=None
                )
                shard_state = load_file(shard_path, device=str(self.device))
                for key in tts_keys:
                    if key in shard_state:
                        tts_state_dict[key] = shard_state[key]
            
            # Load tts_input_types weights
            if "model.tts_input_types.weight" in tts_state_dict:
                self.tts_input_types.weight.data.copy_(tts_state_dict["model.tts_input_types.weight"])
                self.logger.info("Loaded tts_input_types weights from HuggingFace checkpoint")
            else:
                self.logger.warning(
                    "tts_input_types.weight not found in checkpoint, using random initialization"
                )
            
            # Load tts_eos_classifier weights
            eos_classifier_state = {}
            key_mapping = {
                "tts_eos_classifier.fc1.weight": "fc1.weight",
                "tts_eos_classifier.fc1.bias": "fc1.bias",
                "tts_eos_classifier.fc2.weight": "fc2.weight",
                "tts_eos_classifier.fc2.bias": "fc2.bias",
            }
            
            for hf_key, local_key in key_mapping.items():
                if hf_key in tts_state_dict:
                    eos_classifier_state[local_key] = tts_state_dict[hf_key]
            
            if eos_classifier_state:
                self.tts_eos_classifier.load_state_dict(eos_classifier_state, strict=False)
                self.logger.info("Loaded tts_eos_classifier weights from HuggingFace checkpoint")
            else:
                self.logger.warning(
                    "tts_eos_classifier weights not found in checkpoint, using random initialization"
                )
                
        except Exception as e:
            self.logger.warning(
                f"Failed to load TTS component weights from HuggingFace checkpoint: {e}. "
                "Using random initialization."
            )

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
    def tts_num_hidden_layers(self) -> int:
        """Number of hidden layers in the TTS backbone."""
        return 20

    @property
    def tts_num_key_value_heads(self) -> int:
        """Number of key-value heads in the TTS backbone."""
        return 2

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
                - input_tokens: Packed pseudo IDs for LM and TTS-LM cached prefixes
                - input_features: Packed LM/TTS-LM cached hidden states
                - input_masks: Segment masks (LM=False, TTS-LM=True)
                - decoder_cache: Initial VibeVoiceDecoderCache
        """
        # Load voice cache if path provided
        if voice_cache_path is not None:
            self.voice_cache = self._load_voice_cache(voice_cache_path)

        # Verify voice cache is available
        if self.voice_cache is None:
            raise ValueError(
                "Voice cache is required for VibeVoice preprocessing. "
                "Provide voice_cache_path or ensure default cache is available."
            )
        if prompt is None:
            raise ValueError("prompt is required for VibeVoice preprocessing.")

        # Use streaming processor with cached prompt
        batch_encoding = self.streaming_processor.process_input_with_cached_prompt(
            text=prompt,
            cached_prompt=self.voice_cache,
            return_tensors="pt"
        )

        # Pseudo token IDs for cached prompt prefixes:
        # - These IDs use pad_id as placeholders; they carry no semantic meaning
        # - Required for: (1) shape tracking for KV cache allocation,
        #                 (2) position_id computation,
        #                 (3) embedding layer input when input_features not used directly
        # - The actual prompt semantics are in input_features (pre-computed hidden states)
        lm_ids = batch_encoding["input_ids"].view(-1, 1).to(device=self.device, dtype=torch.long)
        tts_ids = batch_encoding["tts_lm_input_ids"].view(-1, 1).to(device=self.device, dtype=torch.long)
        input_tokens = torch.cat([lm_ids, tts_ids], dim=0)

        lm_hidden = self.voice_cache["lm"]["last_hidden_state"]
        tts_hidden = self.voice_cache["tts_lm"]["last_hidden_state"]

        # Remove batch dimension (batch_size=1) to enable sequence wise concatenation
        # Voice cache stores shape (1, seq_len, hidden_dim) from single voice precomputation
        lm_hidden = lm_hidden.squeeze(0)
        tts_hidden = tts_hidden.squeeze(0)

        input_features = torch.cat([lm_hidden, tts_hidden], dim=0).to(device=self.device, dtype=self.dtype)

        lm_mask = torch.zeros((lm_ids.shape[0], 1), dtype=torch.bool, device=self.device)
        tts_mask = torch.ones((tts_ids.shape[0], 1), dtype=torch.bool, device=self.device)
        input_masks = torch.cat([lm_mask, tts_mask], dim=0)

        # Initialize decoder cache
        decoder_cache = self.audio_decoder_initial_cache(batch_size=1)

        return PreprocessOutput(
            input_tokens=input_tokens,
            input_masks=input_masks,
            input_features=input_features,
            repetition_cache=None,
            decoder_cache=decoder_cache,
        )

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.Tensor]:
        """
        Single pass of the base text LM (language backbone).
        
        Converts input_ids to embeddings and forwards through language_backbone.
        When input_features is provided (for cached voice prompts), uses those
        pre-computed hidden states directly instead of computing embeddings.

        Args:
            input_ids: Token IDs. Shape: (batch_size, seq_len)
            position_ids: Position IDs for tokens. Shape: (batch_size, seq_len)
            attn_wrapper: FlashInfer attention wrapper.
            kv_cache: KV cache tensor for language backbone.
            input_features: Pre-computed hidden states for cached prompts. Shape: (batch_size, seq_len, hidden_size)
            input_masks: Segment masks distinguishing LM vs TTS-LM segments. TASK2: Reserved for future segment-aware attention masking.
        """
        # input_masks handling - currently passed through for compatibility
        
        # Step 1: Get embeddings - use pre-computed features if available (for cached prompts)
        if input_features is not None:
            inputs_embeds = input_features
        else:
            inputs_embeds = self.language_backbone.get_input_embeddings()(input_ids)
        
        # Step 2: Forward through backbone layers
        hidden_states = self.language_backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        # Note: kv_cache is updated in-place by FlashInfer
        return hidden_states, kv_cache

    def forward_tts_lm(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_wrapper: FlashInferWrapper,
        kv_cache: torch.Tensor,
        lm_last_hidden_state: torch.FloatTensor,
        tts_text_masks: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.Tensor]:
        """
        Single pass of the TTS LM.

        - Overwrites tail embeddings with `lm_last_hidden_state`.
        - Adds type embedding via `tts_text_masks` (1=text, 0=speech).
        - Predicts EOS from last hidden state (binary classifier).
        - No loss / no full acoustic decoding here.

        Args:
            input_ids: (batch_size, seq_len) token ids for TTS backbone.
            position_ids: (batch_size, seq_len) position ids for tokens.
            attn_wrapper: FlashInfer attention wrapper.
            kv_cache: KV cache tensor for TTS backbone.
            lm_last_hidden_state: (batch_size, 1, hidden_size) hidden states from LM backbone.
            tts_text_masks: (batch_size, seq_len) mask marking positions as text(1)/speech(0).
            input_features: Pre-computed embeddings for cached TTS-LM prompts. Shape: (batch_size, seq_len, hidden_size)
            input_masks: Segment masks for TTS-LM segment identification. TASK2: Reserved for segment-aware attention.

        Returns:
            Tuple containing:
                - tts_eos_logits: (batch_size, 1) - EOS classification logits
                - last_hidden_state: (batch_size, 1, hidden_size) - final hidden state
                - kv_cache: Updated KV cache
        """
        # input_masks handling - currently passed through for compatibility
        
        # Step 1: Get embeddings - use pre-computed features if available (for cached prompts)
        if input_features is not None:
            inputs_embeds = input_features
        else:
            # input_ids shape: (batch_size, seq_len)
            # embedding layer expects (batch_size, seq_len), returns (batch_size, seq_len, hidden_size)
            inputs_embeds = self.tts_backbone.get_input_embeddings()(input_ids)
        
        # Step 2: Replace tail embeddings with LM hidden states
        # lm_last_hidden_state shape: (batch_size, 1, hidden_size)
        if lm_last_hidden_state is not None:
            start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
            inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
        
        # Step 3: Add type embedding (text=1, speech=0)
        type_embeds = self.tts_input_types(tts_text_masks.long())
        inputs_embeds = inputs_embeds + type_embeds
        
        # Step 4: Forward through TTS backbone
        hidden_states = self.tts_backbone(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attn_wrapper=attn_wrapper,
            kv_cache=kv_cache,
        )
        
        # Step 5: Get EOS logits from last hidden state
        eos_logits = self.tts_eos_classifier(hidden_states[:, -1, :])
        
        return eos_logits, hidden_states[:, -1:, :], kv_cache

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
