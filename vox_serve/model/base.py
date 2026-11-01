from abc import ABC, abstractmethod

import torch


class BaseLM(ABC):
    """
    Base class of the models' PyTorch side: modules, weights and the constants their programs
    (``vox_serve/model/<name>/program.py``) read. Steps, sampling and prompts live in the programs.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_torch_compile: bool = False,
        audio_decoder_device: str = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_torch_compile = enable_torch_compile
        # Audio decoder device defaults to main device if not specified
        self.audio_decoder_device = audio_decoder_device or device

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        """Number of codebooks in the model."""
        pass

    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        pass

    @property
    @abstractmethod
    def num_key_value_heads(self) -> int:
        """Number of key-value heads in the model."""
        pass

    @property
    @abstractmethod
    def num_hidden_layers(self) -> int:
        """Number of hidden layers in the model."""
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden size of the model.

        This is the attention working width (``num_attention_heads * head_dim``),
        used to size the FlashInfer attention wrappers. For most models it equals
        the residual-stream width; when they differ (e.g. an explicit ``head_dim``
        that makes ``n_heads * head_dim != residual_width``), override
        ``embedding_hidden_size`` to report the residual width separately.
        """
        pass

    @property
    def head_dim(self) -> int:
        """Head dimension of the model."""
        return self.hidden_size // self.num_attention_heads

    @property
    @abstractmethod
    def detokenize_interval(self) -> int:
        """Interval at which to detokenize outputs."""
        pass

    @property
    @abstractmethod
    def detokenize_overlap(self) -> int:
        """Overlap size for detokenization."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        Maximum number of tokens the model generates in a single request.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size of the model."""
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of audio channels in the output."""
        pass

    @property
    @abstractmethod
    def output_audio_length(self) -> int:
        """Output audio length (in samples) at each postprocess call."""
        pass


class BaseLMWithDepth(BaseLM):
    """
    Base class for language models with depth transformer used in vox-serve.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_torch_compile: bool = False,
        audio_decoder_device: str = None,
    ):
        super().__init__(model_name, device, dtype, enable_torch_compile, audio_decoder_device)

    @property
    @abstractmethod
    def depth_n_codebooks(self) -> int:
        """Number of codebooks in the depth transformer."""
        pass

    @property
    @abstractmethod
    def depth_vocab_size(self) -> int:
        """Vocabulary size of the depth transformer output."""
        pass
