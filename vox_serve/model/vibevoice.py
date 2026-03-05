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
from .base import BaseLMWithContinuousSpeech, PreprocessOutput, VibeVoiceForwardOutput
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
        - Diffusion head (for speech latent generation)
        - Acoustic connector (bridge from latent to TTS LM hidden size)
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
        
        # Load diffusion head for speech latent generation
        self._init_diffusion_head()
        
        # Initialize acoustic connector (bridge from latent to TTS LM hidden size)
        self._init_acoustic_connector()

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

    def _init_diffusion_head(self):
        """
        Initialize the diffusion head for speech latent generation.
        
        Loads the VibeVoiceDiffusionHead from the HuggingFace checkpoint.
        This component is used in the NON-GRAPH region for diffusion sampling.
        """
        from .vibevoice_diffusion_head import VibeVoiceDiffusionHead
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        try:
            # Try to load diffusion head from HuggingFace
            diffusion_head_path = hf_hub_download(
                repo_id=self.model_name,
                filename="diffusion_head/config.json",
                revision=None
            )
            
            # Load the diffusion head
            self.diffusion_head = VibeVoiceDiffusionHead.from_pretrained(
                self.model_name,
                subfolder="diffusion_head",
            ).to(self.device).to(self.dtype)
            
            self.logger.info("Loaded VibeVoiceDiffusionHead from HuggingFace checkpoint")
            
        except Exception as e:
            self.logger.warning(
                f"Failed to load diffusion head from HuggingFace checkpoint: {e}. "
                "Using default initialization."
            )
            # Fall back to default config
            from .vibevoice_config import VibeVoiceDiffusionHeadConfig
            config = VibeVoiceDiffusionHeadConfig(
                hidden_size=self._hidden_size,
                latent_size=self._latent_dim,
            )
            self.diffusion_head = VibeVoiceDiffusionHead(config).to(self.device).to(self.dtype)
        
        # Initialize diffusion scheduler (DPM-Solver for faster sampling)
        self._init_diffusion_scheduler()
    
    def _init_diffusion_scheduler(self):
        """
        Initialize the DPM-Solver scheduler for diffusion sampling.
        
        Uses DPMSolverMultistepScheduler for efficient sampling with fewer steps.
        """
        try:
            from diffusers import DPMSolverMultistepScheduler
            
            self.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_schedule="cosine",
                prediction_type="v_prediction",
            )
            self.logger.info("Initialized DPMSolverMultistepScheduler for diffusion sampling")
        except ImportError:
            self.logger.warning(
                "diffusers not available, falling back to simple DDPM scheduler"
            )
            self.scheduler = None
    
    def _init_acoustic_connector(self):
        """
        Initialize the acoustic connector that bridges latents to TTS LM hidden size.
        
        The connector projects speech latents (latent_dim=64) to the hidden size
        of the TTS LM (hidden_size=1536) so they can be injected as input embeddings.
        
        Structure matches HuggingFace model:
            - fc1: latent_dim -> hidden_size
            - fc2: hidden_size -> hidden_size
            - norm: LayerNorm(hidden_size)
        """
        class AcousticConnector(nn.Module):
            def __init__(self, latent_dim, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(latent_dim, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size)
            
            def forward(self, x):
                x = self.fc1(x)
                x = torch.nn.functional.gelu(x)
                x = self.fc2(x)
                x = self.norm(x)
                return x
        
        self.acoustic_connector = AcousticConnector(
            self._latent_dim,
            self._hidden_size
        ).to(self.device).to(self.dtype)
        
        # Try to load acoustic connector weights from HuggingFace
        self._load_acoustic_connector_weights()
    
    def _load_acoustic_connector_weights(self):
        """
        Load acoustic connector weights from HuggingFace checkpoint.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        connector_keys = [
            "model.acoustic_connector.fc1.weight",
            "model.acoustic_connector.fc1.bias",
            "model.acoustic_connector.fc2.weight",
            "model.acoustic_connector.fc2.bias",
            "model.acoustic_connector.norm.weight",
        ]
        
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
            
            # Collect all shards needed for acoustic connector
            shards_needed = set()
            for key in connector_keys:
                if key in weight_map:
                    shards_needed.add(weight_map[key])
            
            if not shards_needed:
                self.logger.warning(
                    "No acoustic_connector weights found in checkpoint weight map, "
                    "using random initialization"
                )
                return
            
            # Load all needed shards
            loaded_weights = {}
            for shard in shards_needed:
                shard_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=shard,
                    revision=None
                )
                shard_state = load_file(shard_path, device=str(self.device))
                for key in connector_keys:
                    if key in shard_state:
                        loaded_weights[key] = shard_state[key]
            
            # Apply weights to acoustic connector
            if "model.acoustic_connector.fc1.weight" in loaded_weights:
                self.acoustic_connector.fc1.weight.data.copy_(
                    loaded_weights["model.acoustic_connector.fc1.weight"]
                )
            if "model.acoustic_connector.fc1.bias" in loaded_weights:
                self.acoustic_connector.fc1.bias.data.copy_(
                    loaded_weights["model.acoustic_connector.fc1.bias"]
                )
            if "model.acoustic_connector.fc2.weight" in loaded_weights:
                self.acoustic_connector.fc2.weight.data.copy_(
                    loaded_weights["model.acoustic_connector.fc2.weight"]
                )
            if "model.acoustic_connector.fc2.bias" in loaded_weights:
                self.acoustic_connector.fc2.bias.data.copy_(
                    loaded_weights["model.acoustic_connector.fc2.bias"]
                )
            if "model.acoustic_connector.norm.weight" in loaded_weights:
                self.acoustic_connector.norm.weight.data.copy_(
                    loaded_weights["model.acoustic_connector.norm.weight"]
                )
            
            self.logger.info("Loaded acoustic_connector weights from HuggingFace checkpoint")
                
        except Exception as e:
            self.logger.warning(
                f"Failed to load acoustic connector weights from HuggingFace checkpoint: {e}. "
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

    def forward(
        self,
        # === LM Backbone Inputs ===
        lm_input_ids: torch.Tensor,
        lm_position_ids: torch.Tensor,
        lm_attn_wrapper: FlashInferWrapper,
        lm_kv_cache: torch.Tensor,
        # === TTS LM Inputs ===
        tts_input_ids: torch.Tensor,
        tts_position_ids: torch.Tensor,
        tts_attn_wrapper: FlashInferWrapper,
        tts_kv_cache: torch.Tensor,
        # === Bridge Inputs ===
        lm_last_hidden_state: torch.Tensor,
        tts_text_masks: torch.Tensor,
        # === Optional Cached Features ===
        lm_input_features: Optional[torch.Tensor] = None,
        tts_input_features: Optional[torch.Tensor] = None,
        # === Control Flags ===
        phase: str = "text",
    ) -> VibeVoiceForwardOutput:
        """
        Unified forward for both text and speech phases.

        This method chains the LM backbone and TTS LM forward passes in a single
        coherent call, supporting CUDA graph compatibility through fixed-shape
        tensor inputs.

        TEXT phase: forward_lm → forward_tts_lm (chain both)
        SPEECH phase: forward_tts_lm only (with acoustic_embed injection)

        Args:
            lm_input_ids: Input token IDs for LM backbone.
                Shape: [batch_size, text_window_size] or [batch_size, 1] for cached
            lm_position_ids: Position IDs for LM tokens.
                Shape: [batch_size, text_window_size]
            lm_attn_wrapper: Pre-planned FlashInfer wrapper for LM backbone.
            lm_kv_cache: Paged KV cache for LM backbone.
            tts_input_ids: Input token IDs for TTS LM.
                Shape: [batch_size, 1] (single step)
            tts_position_ids: Position IDs for TTS tokens.
                Shape: [batch_size, 1]
            tts_attn_wrapper: Pre-planned FlashInfer wrapper for TTS LM.
            tts_kv_cache: Paged KV cache for TTS LM.
            lm_last_hidden_state: Hidden states to inject into TTS LM.
                Shape: [batch_size, 1, hidden_size] for speech phase, or
                       [batch_size, text_window_size, hidden_size] for text phase
            tts_text_masks: Mask for text/speech distinction.
                Shape: [batch_size, 1] - 1=text, 0=speech
            lm_input_features: Pre-computed hidden states for cached LM prompts.
                Shape: [batch_size, seq_len, hidden_size] or None
            tts_input_features: Pre-computed hidden states for cached TTS prompts.
                Shape: [batch_size, seq_len, hidden_size] or None
            phase: Generation phase - "text" or "speech".

        Returns:
            VibeVoiceForwardOutput with:
                - lm_hidden_state: [batch_size, seq_len, hidden_size] or None for speech phase
                - tts_hidden_state: [batch_size, 1, hidden_size]
                - eos_logits: [batch_size, 1]

        Note:
            KV caches are updated in-place and not returned in the output.
        """
        if phase == "text":
            # TEXT phase: Chain LM backbone → TTS LM
            lm_hidden_state, lm_kv_cache = self.forward_lm(
                input_ids=lm_input_ids,
                position_ids=lm_position_ids,
                attn_wrapper=lm_attn_wrapper,
                kv_cache=lm_kv_cache,
                input_features=lm_input_features,
            )

            # Use LM hidden state as input to TTS LM
            tts_lm_input = lm_hidden_state
        else:
            # SPEECH phase: Skip LM, use acoustic_embed directly
            lm_hidden_state = None
            tts_lm_input = lm_last_hidden_state

        # Forward through TTS LM
        eos_logits, tts_hidden_state, tts_kv_cache = self.forward_tts_lm(
            input_ids=tts_input_ids,
            position_ids=tts_position_ids,
            attn_wrapper=tts_attn_wrapper,
            kv_cache=tts_kv_cache,
            lm_last_hidden_state=tts_lm_input,
            tts_text_masks=tts_text_masks,
            input_features=tts_input_features,
        )

        return VibeVoiceForwardOutput(
            lm_hidden_state=lm_hidden_state,
            tts_hidden_state=tts_hidden_state,
            eos_logits=eos_logits,
        )

    # Window constants
    TTS_TEXT_WINDOW_SIZE = 5
    TTS_SPEECH_WINDOW_SIZE = 6

    def generate_step(
        self,
        request: Request,
        lm_attn_wrapper,
        lm_kv_cache: torch.Tensor,
        tts_attn_wrapper,
        tts_kv_cache: torch.Tensor,
        dummy_token_id: int = 0,
        cfg_scale: float = 3.0,
        num_diffusion_steps: int = 10,
    ) -> Tuple[str, Optional[VibeVoiceForwardOutput]]:
        """
        Execute a single step of the windowed generation loop.

        This method implements the state machine from the plan:
        1. TEXT WINDOW LOOP: Process text tokens in windows of TTS_TEXT_WINDOW_SIZE
        2. SPEECH WINDOW LOOP: Generate TTS_SPEECH_WINDOW_SIZE speech steps per text window

        Args:
            request: The Request object containing generation state
            lm_attn_wrapper: FlashInfer wrapper for LM backbone
            lm_kv_cache: KV cache for LM backbone
            tts_attn_wrapper: FlashInfer wrapper for TTS LM
            tts_kv_cache: KV cache for TTS LM
            dummy_token_id: Token ID to use as placeholder for speech steps
            cfg_scale: Classifier-free guidance scale for diffusion sampling
            num_diffusion_steps: Number of diffusion steps for sampling

        Returns:
            Tuple of (phase, output) where:
                - phase: "text" | "speech" | "finished"
                - output: VibeVoiceForwardOutput or None if finished
        """
        # Check if generation is complete
        if self._is_generation_finished(request):
            return "finished", None

        # Determine current phase
        text_window_idx = request.text_window_index
        speech_step = request.speech_step_in_window
        total_text_tokens = len(request.text_tokens)

        # Calculate text window boundaries
        text_start = text_window_idx * self.TTS_TEXT_WINDOW_SIZE
        text_end = min(text_start + self.TTS_TEXT_WINDOW_SIZE, total_text_tokens)

        # TEXT WINDOW LOOP: Process text tokens if we haven't finished all windows
        if text_window_idx * self.TTS_TEXT_WINDOW_SIZE < total_text_tokens:
            # Check if we should process a text window or continue speech
            if speech_step == 0:
                # Start of a new text window - process text
                return self._process_text_window(
                    request=request,
                    text_tokens=request.text_tokens[text_start:text_end],
                    lm_attn_wrapper=lm_attn_wrapper,
                    lm_kv_cache=lm_kv_cache,
                    tts_attn_wrapper=tts_attn_wrapper,
                    tts_kv_cache=tts_kv_cache,
                )
            else:
                # Still in speech window from previous text window
                return self._process_speech_step(
                    request=request,
                    dummy_token_id=dummy_token_id,
                    tts_attn_wrapper=tts_attn_wrapper,
                    tts_kv_cache=tts_kv_cache,
                    cfg_scale=cfg_scale,
                    num_diffusion_steps=num_diffusion_steps,
                )
        else:
            # All text windows processed - continue speech generation until EOS
            return self._process_speech_step(
                request=request,
                dummy_token_id=dummy_token_id,
                tts_attn_wrapper=tts_attn_wrapper,
                tts_kv_cache=tts_kv_cache,
                cfg_scale=cfg_scale,
                num_diffusion_steps=num_diffusion_steps,
            )

    def _process_text_window(
        self,
        request: Request,
        text_tokens: List[int],
        lm_attn_wrapper,
        lm_kv_cache: torch.Tensor,
        tts_attn_wrapper,
        tts_kv_cache: torch.Tensor,
    ) -> Tuple[str, VibeVoiceForwardOutput]:
        """
        Process a text window: forward LM → forward TTS LM.

        Updates request state:
            - text_window_index: incremented after processing
            - speech_step_in_window: reset to 0
            - lm_last_hidden_state, tts_hidden_state, eos_logits: updated
        """
        batch_size = 1
        window_size = len(text_tokens)

        # Prepare LM inputs
        lm_input_ids = torch.tensor([text_tokens], dtype=torch.long, device=self.device)
        lm_position_ids = torch.arange(
            request.next_position_id,
            request.next_position_id + window_size,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        # Prepare TTS inputs (same tokens for TTS LM)
        tts_input_ids = lm_input_ids.clone()
        tts_position_ids = lm_position_ids.clone()

        # Text mask: 1 for text positions
        tts_text_masks = torch.ones(batch_size, window_size, dtype=torch.long, device=self.device)

        # Run unified forward in text phase
        output = self.forward(
            lm_input_ids=lm_input_ids,
            lm_position_ids=lm_position_ids,
            lm_attn_wrapper=lm_attn_wrapper,
            lm_kv_cache=lm_kv_cache,
            tts_input_ids=tts_input_ids,
            tts_position_ids=tts_position_ids,
            tts_attn_wrapper=tts_attn_wrapper,
            tts_kv_cache=tts_kv_cache,
            lm_last_hidden_state=None,  # Will be computed in text phase
            tts_text_masks=tts_text_masks,
            phase="text",
        )

        # Update request state
        request.lm_last_hidden_state = output.lm_hidden_state[:, -1:, :]  # Keep last token hidden state
        request.tts_hidden_state = output.tts_hidden_state
        request.eos_logits = output.eos_logits

        # Update position tracking
        request.next_position_id += window_size

        # Check EOS (rare in text phase, but possible)
        if self._check_eos(request, output.eos_logits):
            request.finished_tags = torch.tensor([True], device=self.device)
            return "finished", output

        # Move to speech phase
        request.text_window_index += 1
        request.speech_step_in_window = 0

        return "text", output

    def _process_speech_step(
        self,
        request: Request,
        dummy_token_id: int,
        tts_attn_wrapper,
        tts_kv_cache: torch.Tensor,
        cfg_scale: float = 3.0,
        num_diffusion_steps: int = 10,
    ) -> Tuple[str, VibeVoiceForwardOutput]:
        """
        Process a single speech step: diffusion sample → acoustic decode → TTS LM forward.

        Updates request state:
            - speech_step_in_window: incremented
            - tts_hidden_state, eos_logits: updated
            - output_audio: audio chunks are queued
        
        Args:
            request: The Request object containing generation state
            dummy_token_id: Token ID to use as placeholder for speech steps
            tts_attn_wrapper: FlashInfer wrapper for TTS LM
            tts_kv_cache: KV cache for TTS LM
            cfg_scale: Classifier-free guidance scale for diffusion
            num_diffusion_steps: Number of diffusion sampling steps
        """
        batch_size = 1

        # Prepare TTS inputs for single speech step
        tts_input_ids = torch.tensor([[dummy_token_id]], dtype=torch.long, device=self.device)
        tts_position_ids = torch.tensor([[request.next_position_id]], dtype=torch.long, device=self.device)

        # Speech mask: 0 for speech positions
        tts_text_masks = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        # === NON-GRAPH REGION: Diffusion sampling ===
        # Sample speech latent from diffusion head
        speech_latent = self.sample_speech_latent(
            tts_hidden_state=request.tts_hidden_state,
            cfg_scale=cfg_scale,
            num_inference_steps=num_diffusion_steps,
        )

        # === NON-GRAPH REGION: Acoustic decode ===
        # Decode latent to audio (if decoder cache available)
        if request.decoder_cache is not None:
            audio_chunk, request.decoder_cache = self.decode_speech_latent(
                speech_latent=speech_latent,
                decoder_cache=request.decoder_cache,
            )
            # Queue audio chunk for output
            request.output_audio.put(audio_chunk)

        # Bridge latent → acoustic embed for TTS LM injection
        acoustic_embed = self.compute_acoustic_embed(speech_latent)

        # === GRAPH-CAPTURED REGION: TTS LM forward ===
        # Run unified forward in speech phase
        output = self.forward(
            lm_input_ids=torch.empty(0, device=self.device),  # Not used in speech phase
            lm_position_ids=torch.empty(0, device=self.device),
            lm_attn_wrapper=None,
            lm_kv_cache=torch.empty(0, device=self.device),
            tts_input_ids=tts_input_ids,
            tts_position_ids=tts_position_ids,
            tts_attn_wrapper=tts_attn_wrapper,
            tts_kv_cache=tts_kv_cache,
            lm_last_hidden_state=acoustic_embed,
            tts_text_masks=tts_text_masks,
            phase="speech",
        )

        # Update request state
        request.tts_hidden_state = output.tts_hidden_state
        request.eos_logits = output.eos_logits
        request.next_position_id += 1

        # Check EOS
        if self._check_eos(request, output.eos_logits):
            request.finished_tags = torch.tensor([True], device=self.device)
            return "finished", output

        # Update speech step counter
        request.speech_step_in_window += 1

        # Check if speech window is complete
        if request.speech_step_in_window >= self.TTS_SPEECH_WINDOW_SIZE:
            # Reset for next window
            request.speech_step_in_window = 0

        # Check max tokens limit
        total_speech_tokens = request.speech_step_in_window + request.text_window_index * self.TTS_SPEECH_WINDOW_SIZE
        if total_speech_tokens >= request.max_speech_tokens:
            request.finished_tags = torch.tensor([True], device=self.device)
            return "finished", output

        return "speech", output

    def _check_eos(self, request: Request, eos_logits: torch.Tensor) -> bool:
        """
        Check if EOS has been triggered based on eos_logits.

        Args:
            request: Request containing eos_threshold
            eos_logits: [batch, 1] logits from binary classifier

        Returns:
            True if sigmoid(eos_logits) > eos_threshold
        """
        eos_prob = torch.sigmoid(eos_logits)
        return (eos_prob > request.eos_threshold).any().item()

    def _is_generation_finished(self, request: Request) -> bool:
        """
        Check if generation should terminate.

        Termination conditions:
            1. EOS detected (finished_tags == True)
            2. Max speech tokens reached
            3. All text processed and speech window complete

        Returns:
            True if generation should stop
        """
        if request.finished_tags is not None and request.finished_tags.any().item():
            return True

        total_speech_steps = (
            request.text_window_index * self.TTS_SPEECH_WINDOW_SIZE + request.speech_step_in_window
        )
        if total_speech_steps >= request.max_speech_tokens:
            return True

        return False

    def init_generation_state(self, request: Request) -> None:
        """
        Initialize VibeVoice-specific generation state on a Request.

        This should be called after preprocess() to set up:
            - text_tokens from tokenized prompt
            - total_text_windows
            - finished_tags tensor

        Args:
            request: Request to initialize
        """
        # Tokenize the prompt if not already done
        if not request.text_tokens and request.prompt:
            request.text_tokens = self.text_tokenizer.encode(request.prompt, add_special_tokens=False)

        # Calculate total text windows
        total_tokens = len(request.text_tokens)
        request.total_text_windows = (total_tokens + self.TTS_TEXT_WINDOW_SIZE - 1) // self.TTS_TEXT_WINDOW_SIZE

        # Initialize finished tags
        request.finished_tags = torch.tensor([False], device=self.device)

        # Reset window counters
        request.text_window_index = 0
        request.speech_step_in_window = 0

    def sample_speech_latent(
        self,
        tts_hidden_state: torch.Tensor,
        cfg_scale: float = 3.0,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        NON-GRAPH: Diffusion sampling to generate speech latent.
        
        This is the boundary between graph-captured forward and diffusion.
        Uses DPM-Solver for efficient sampling with classifier-free guidance.
        
        Args:
            tts_hidden_state: Condition from TTS LM. Shape: [batch_size, 1, hidden_size]
            cfg_scale: Classifier-free guidance scale. Higher values = stronger conditioning.
            num_inference_steps: Number of diffusion steps (fewer = faster, more = quality).
            generator: Optional torch.Generator for reproducible sampling.
        
        Returns:
            speech_latent: [batch_size, latent_dim]
        """
        batch_size = tts_hidden_state.shape[0]
        
        # Squeeze sequence dimension for diffusion head
        condition = tts_hidden_state.squeeze(1)  # [batch_size, hidden_size]
        
        # Initialize random noise
        latents = torch.randn(
            batch_size, self._latent_dim,
            device=self.device, dtype=self.dtype,
            generator=generator,
        )
        
        # Use DPM-Solver if available, otherwise fallback to simple DDPM
        if self.scheduler is not None:
            # Set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps.to(self.device)
            
            # DPM-Solver sampling loop
            for t in timesteps:
                # Expand latents for CFG if needed
                if cfg_scale > 1.0:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep = t.expand(latent_model_input.shape[0])
                    condition_input = torch.cat([condition, torch.zeros_like(condition)], dim=0)
                else:
                    latent_model_input = latents
                    timestep = t.expand(batch_size)
                    condition_input = condition
                
                # Predict noise
                noise_pred = self.diffusion_head(
                    noisy_images=latent_model_input,
                    timesteps=timestep,
                    condition=condition_input,
                )
                
                # Apply CFG
                if cfg_scale > 1.0:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Compute previous sample
                latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        else:
            # Fallback: Simple DDPM-style sampling
            for i in range(num_inference_steps):
                t = torch.tensor(
                    [1000 - i * (1000 // num_inference_steps)] * batch_size,
                    device=self.device, dtype=torch.long
                )
                
                # Predict noise
                noise_pred = self.diffusion_head(
                    noisy_images=latents,
                    timesteps=t,
                    condition=condition,
                )
                
                # Simple denoising step
                alpha = 1.0 - (i / num_inference_steps)
                latents = latents - (1 - alpha) * noise_pred
        
        return latents

    def decode_speech_latent(
        self,
        speech_latent: torch.Tensor,
        decoder_cache: DecoderCache,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DecoderCache]:
        """
        NON-GRAPH: Acoustic decoder to convert latent → audio.
        
        Uses the VibeVoiceDecoder for streaming audio generation with cache.
        
        Args:
            speech_latent: Speech latent from diffusion sampling. Shape: [batch_size, latent_dim]
            decoder_cache: VibeVoiceDecoderCache for streaming decode state.
            sample_indices: [batch] indices for cache lookup. Defaults to [0] for batch_size=1.
        
        Returns:
            audio_chunk: [batch_size, n_channels, audio_samples]
            decoder_cache: Updated cache
        """
        from ..tokenizer.vibevoice_acoustic import VibeVoiceDecoderCache, VibeVoiceDecoder
        
        batch_size = speech_latent.shape[0]
        
        # Default sample indices for batch_size=1
        if sample_indices is None:
            sample_indices = torch.arange(batch_size, device=self.device, dtype=torch.long)
        
        # Ensure latent has correct shape [B, 1, latent_dim] for decoder
        if speech_latent.ndim == 2:
            speech_latent = speech_latent.unsqueeze(1)  # [B, 1, latent_dim]
        
        # Decode using the acoustic tokenizer's decoder
        audio_chunk, decoder_cache = self.audio_decoder.decode_chunk(
            latents=speech_latent,
            decoder_cache=decoder_cache,
        )
        
        return audio_chunk, decoder_cache

    def compute_acoustic_embed(
        self,
        speech_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute acoustic embedding from speech latent for TTS LM injection.
        
        This bridges the diffusion output (latent_dim=64) to the TTS LM input
        space (hidden_size=1536).
        
        Args:
            speech_latent: Speech latent from diffusion sampling. Shape: [batch_size, latent_dim]
        
        Returns:
            acoustic_embed: [batch_size, 1, hidden_size] - ready for TTS LM injection
        """
        # Project latent to hidden size
        acoustic_embed = self.acoustic_connector(speech_latent)  # [batch_size, hidden_size]
        
        # Add sequence dimension for TTS LM
        acoustic_embed = acoustic_embed.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        return acoustic_embed

    def diffusion_forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through diffusion head to generate speech latents.

        This is a convenience method that wraps sample_speech_latent for backward
        compatibility. Prefer using sample_speech_latent directly for new code.

        Args:
            hidden_states: Hidden states from backbone model. Shape: (batch_size, seq_len, hidden_size)
            **kwargs: Additional parameters passed to sample_speech_latent

        Returns:
            Continuous speech latents. Shape: (batch_size, seq_len, latent_dim)
        """
        # Extract kwargs with defaults
        cfg_scale = kwargs.get("cfg_scale", 3.0)
        num_inference_steps = kwargs.get("num_inference_steps", 10)
        generator = kwargs.get("generator", None)
        
        # Sample latents using diffusion
        latents = self.sample_speech_latent(
            tts_hidden_state=hidden_states,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        
        # Add sequence dimension if needed for compatibility
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)
        
        return latents

    def postprocess(
        self,
        latents: torch.Tensor,
        decoder_cache: Optional[DecoderCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert continuous latents to audio waveform using acoustic decoder.

        This method provides non-streaming audio decode. For streaming decode,
        use decode_speech_latent() directly which maintains decoder cache state.

        Args:
            latents: Continuous speech latents from diffusion head. Shape: (batch_size, seq_len, latent_dim)
            decoder_cache: Optional decoder cache for acoustic decoder
            **kwargs: Additional model-specific parameters

        Returns:
            Audio tensor. Shape: (batch_size, n_channels, audio_length)
        """
        batch_size = latents.size(0)
        
        # If no cache provided, initialize a new one
        if decoder_cache is None:
            decoder_cache = self.audio_decoder_initial_cache(batch_size)
        
        # Flatten sequence dimension for chunk-by-chunk decode
        if latents.ndim == 3:
            seq_len = latents.size(1)
            latents_flat = latents.view(batch_size * seq_len, -1)
            # Expand cache for batch_size * seq_len
            expanded_cache = self.audio_decoder_initial_cache(batch_size * seq_len)
        else:
            latents_flat = latents
            expanded_cache = decoder_cache
        
        # Decode latents to audio
        audio, _ = self.decode_speech_latent(
            speech_latent=latents_flat,
            decoder_cache=expanded_cache,
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
