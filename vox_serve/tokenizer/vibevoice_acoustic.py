from dataclasses import dataclass
import math
from functools import partial
from typing import Optional, Tuple, List, Dict
import typing as tp
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import copy
import os
from transformers.configuration_utils import PretrainedConfig 
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import logging
from .base import DecoderCache

logger = logging.get_logger(__name__)
# Try to import APEX FusedRMSNorm
try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    APEX_AVAILABLE = True
    # logger.info("APEX FusedRMSNorm is available and will be used for optimization")
    if int(os.getenv("OPTIMIZE_FOR_SPEED", "0")) == 0:
        APEX_AVAILABLE = False
        # logger.warning("APEX FusedRMSNorm is disabled by environment variable OPTIMIZE_FOR_SPEED=0")
except ImportError:
    APEX_AVAILABLE = False
    # logger.warning("APEX FusedRMSNorm not available, using native implementation")

@dataclass
class VibeVoiceDecoderCache(DecoderCache):
    """
    Tensor-backed streaming cache for VibeVoice acoustic tokenizer decode.

    states[name] = cached input tail for that streaming layer.
    Shape is fixed: [B, C_in, T_ctx]
    """
    states: Dict[str, torch.Tensor]

class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        # common 
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        # decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None, # if None, same as encoder
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        
        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        
        # decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths

# Helper functions for tensor-backed cache
def _collect_decoder_stream_layers(decoder: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Deterministic naming scheme for all streaming conv layers that touch the cache.
    Keys must be stable across runs for DecoderCache.copy_from/cat to work.
    """
    layers: List[Tuple[str, nn.Module]] = []

    # 1) upsample_layers (ModuleList of Sequentials)
    for i, seq in enumerate(decoder.upsample_layers):
        for j, layer in enumerate(seq):
            if isinstance(layer, (SConv1d, SConvTranspose1d)):
                layers.append((f"decoder.upsample_layers.{i}.{j}", layer))

    # 2) stages: Block1D mixer conv (SConv1d)
    for si, stage in enumerate(decoder.stages):
        for bi, block in enumerate(stage):
            if hasattr(block, "mixer") and hasattr(block.mixer, "conv") and isinstance(block.mixer.conv, SConv1d):
                layers.append((f"decoder.stages.{si}.{bi}.mixer", block.mixer.conv))

    # 3) head
    if isinstance(decoder.head, SConv1d):
        layers.append(("decoder.head", decoder.head))

    return layers

@torch.no_grad()
def init_vibevoice_decoder_cache(acoustic_tokenizer: "VibeVoiceAcousticTokenizerModel",
                                batch_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> VibeVoiceDecoderCache:
    layers = _collect_decoder_stream_layers(acoustic_tokenizer.decoder)

    states: Dict[str, torch.Tensor] = {}
    for name, layer in layers:
        c_in = int(layer.in_channels)
        t_ctx = int(getattr(layer, "context_size", 0))
        assert t_ctx >= 0, f"Negative context_size={t_ctx} for {name}"
        # Fixed shape for CUDA graphs:
        states[name] = torch.zeros((batch_size, c_in, t_ctx), device=device, dtype=dtype)

    return VibeVoiceDecoderCache(states=states)

def build_layer_id_to_name(acoustic_tokenizer: "VibeVoiceAcousticTokenizerModel") -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, layer in _collect_decoder_stream_layers(acoustic_tokenizer.decoder):
        mapping[layer.layer_id] = name
    return mapping

# A tensor backed implementation of VibeVoiceTokenizerStreamingCache
class VibeVoiceTensorStreamingCache:
    """
    Graph-friendly cache adapter that matches the get/set API used by SConv layers.

    Backed by VibeVoiceDecoderCache.states (dict of tensors).
    Never returns None. No dynamic shapes. Strong asserts.
    """

    def __init__(self, layer_id_to_name: dict[str, str], decoder_cache: VibeVoiceDecoderCache):
        self._layer_id_to_name = layer_id_to_name
        self._cache = decoder_cache

    def get(self, layer_id: str, sample_indices: torch.Tensor) -> torch.Tensor:
        assert layer_id in self._layer_id_to_name, f"Unknown layer_id: {layer_id}"
        name = self._layer_id_to_name[layer_id]
        assert name in self._cache.states, f"Unknown cache key: {name}"

        buf = self._cache.states[name]  # [B, C, T]
        assert sample_indices.ndim == 1
        # returns [len(sample_indices), C, T]
        return buf.index_select(0, sample_indices)

    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor) -> None:
        assert layer_id in self._layer_id_to_name, f"Unknown layer_id: {layer_id}"
        name = self._layer_id_to_name[layer_id]
        dst = self._cache.states[name]  # [B, C, T]
        assert states.shape[1:] == dst.shape[1:], (states.shape, dst.shape)
        dst.index_copy_(0, sample_indices, states)

    def set_to_zero(self, sample_indices: torch.Tensor) -> None:
        for k, v in self._cache.states.items():
            v.index_fill_(0, sample_indices, 0)

    def clear(self, layer_id: str | None = None, sample_indices: torch.Tensor | None = None) -> None:
        # “clear” in tensor world means “zero-out”
        if layer_id is None and sample_indices is None:
            for v in self._cache.states.values():
                v.zero_()
            return

        if layer_id is not None:
            # allow layer_id to be either conv's runtime layer_id or stable name
            name = self._layer_id_to_name.get(layer_id, layer_id)
            assert name in self._cache.states, f"Unknown layer key: {name}"
            if sample_indices is None:
                self._cache.states[name].zero_()
            else:
                self._cache.states[name].index_fill_(0, sample_indices, 0)
            return

        # layer_id is None, sample_indices provided
        assert sample_indices is not None
        self.set_to_zero(sample_indices)

# Normalization modules
class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x) 
        x = x.transpose(1, 2)  # b t ... -> b ... t
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        if (not APEX_AVAILABLE) or (not self.elementwise_affine):
            # Fallback to native implementation
            output = self._norm(x.float()).type_as(x)
            if self.weight is not None:
                output = output * self.weight
        else:
            output = fused_rms_norm_affine(x, self.weight, self.weight.shape, self.eps)
        output = output.transpose(1, 2)  # b t ... -> b ... t
        return output

# Convolutional layers and utilities
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Pad 1D input with handling for small inputs in reflect mode"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x

class SConv1d(nn.Module):
    """Conv1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        
        # Store configuration
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For causal convolution, we need to maintain kernel_size - 1 samples as context
        # need to check use which context_size is more suitable
        # self.context_size = (kernel_size - 1) * dilation
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        
        # For non-streaming mode, calculate padding
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        
        # Create a unique layer ID for cache management
        self._layer_id = None
                  
    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id
        
    def forward(self, x: torch.Tensor, 
                cache: Optional[VibeVoiceTensorStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False,
                is_final_chunk: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        
        Args:
            x: Input tensor [batch_size, channels, time]
            cache: VibeVoiceTensorStreamingCache object for maintaining states
            sample_indices: Indices identifying each sample for cache management
            use_cache: Whether to use cached states for streaming
            debug: Whether to print debug information
            is_final_chunk: Whether this is the final chunk (adds extra padding for alignment)
            
        Returns:
            Output tensor
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert self.causal, "Streaming mode is only supported for causal convolutions"
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug, is_final_chunk)
    
    def _forward_streaming(self, x: torch.Tensor, 
                          cache: VibeVoiceTensorStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False,
                          is_final_chunk: bool = False) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape
        
        # Cache operations (not compiled)
        cached_states = cache.get(self.layer_id, sample_indices)
        assert cached_states is not None, "cached_states is None"
        
        # Concatenate cached states with input
        input_with_context = torch.cat([cached_states, x], dim=2)
        # For final chunk, add extra padding to ensure ceil behavior (same as non-streaming)
        if is_final_chunk:
            extra_padding = get_extra_padding_for_conv1d(
                input_with_context, self.kernel_size, self.stride, self.padding_total
            )
            if extra_padding > 0:
                input_with_context = pad1d(input_with_context, (0, extra_padding), mode=self.pad_mode)

        output = self.conv(input_with_context)
        # Update cache for next chunk
        if self.context_size > 0:
            new_cache = input_with_context[:, :, -self.context_size:]
            cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        padding_total = self.padding_total
        
        # Compute extra padding for stride alignment
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}, padding_total={padding_total}, extra_padding={extra_padding}")
        
        if self.causal:
            # Left padding for causal
            if self.pad_mode == 'constant':
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode, value=0)
            else:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Symmetric padding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After padding: {x.shape}")
            
        output = self.conv(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Output shape: {output.shape}")
        
        return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, causal: bool = False,
                norm: str = 'none', trim_right_ratio: float = 1.,
                norm_kwargs: tp.Dict[str, tp.Any] = {}, bias: bool = True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                        causal=causal, norm=norm, norm_kwargs=norm_kwargs, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For transposed convolution, padding calculation is different
        self.padding_total = kernel_size - stride
        
        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - 1
        
        # Create a unique layer ID for cache management
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id
    
    def forward(self, x: torch.Tensor,
                cache: Optional[VibeVoiceTensorStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug)
    
    def _forward_streaming(self, x: torch.Tensor,
                          cache: VibeVoiceTensorStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False) -> torch.Tensor:
        B, C, T = x.shape
        
        cached_input = cache.get(self.layer_id, sample_indices)   # [B,C,k-1]
        assert cached_input.shape[2] == self.context_size

        full_input = torch.cat([cached_input, x], dim=2)
        full_output = self.convtr(full_input)
        
        # Calculate padding to remove
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        # Remove padding
        if padding_left + padding_right > 0:
            full_output = unpad1d(full_output, (padding_left, padding_right))

        expected_new_output = T * self.stride
        assert full_output.shape[2] >= expected_new_output, f"full_output.shape[2]={full_output.shape[2]}; expected_new_output={expected_new_output}"
        output = full_output[:, :, -expected_new_output:]
        assert self.context_size > 0, "ConvTranspose1d streaming expects kernel_size>1"
        new_cache = full_input[:, :, -self.context_size:]
        cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}")
        
        # Apply transposed convolution
        y = self.convtr(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After transposed conv: {y.shape}")
        
        # Calculate and remove padding
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        if padding_left + padding_right > 0:
            y = unpad1d(y, (padding_left, padding_right))
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Final output shape: {y.shape}")
            
        return y
    
# FFN 
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias) 
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Convlayer(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            pad_mode='zeros', 
            norm='weight_norm', 
            causal=True, 
        ):
        super().__init__()
        self.conv = SConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, 
                           groups=groups, bias=bias, pad_mode=pad_mode, norm=norm, causal=causal)

    def forward(self, x):
        return self.conv(x)

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv',  
                layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        
        if kwargs.get('layernorm', 'LN') == 'LN':
            self.norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))               
        elif kwargs.get('layernorm', 'RMSNorm') == 'RMSNorm':
            self.norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))

        if mixer_layer == 'conv':
            self.mixer = Convlayer(dim, dim, groups=kwargs.get('groups', 1),
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        elif mixer_layer == 'depthwise_conv':
            self.mixer = Convlayer(dim, dim, groups=dim,
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")
        
        self.ffn = FFN(
            dim, 
            kwargs.get('ffn_expansion', 4) * dim, 
            bias=kwargs.get('bias', False),
        )
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        # mixer
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        return x

class TokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        
        # IMPORTANT CHANGE: Don't reverse depths again since they're already reversed in VibeVoiceAcousticTokenizerModel
        self.depths = config.depths  # Changed from list(reversed(config.depths))
        
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and upsampling layers
        stem = nn.Sequential(
                SConv1d(self.dimension, self.n_filters * 2 ** (len(self.depths) - 1), kernel_size, norm=norm, 
                        norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(in_ch, out_ch,
                                kernel_size=self.ratios[i] * 2, stride=self.ratios[i],
                                norm=norm, norm_kwargs=norm_params, bias=bias,
                                causal=self.causal, trim_right_ratio=trim_right_ratio),
            )
            self.upsample_layers.append(upsample_layer)

        # configure transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        
        # Create stages in the same order as the original model
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.channels, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        for i in range(len(self.depths)):
            # Apply upsampling
            for layer in self.upsample_layers[i]:
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                else:
                    x = layer(x)
            
            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x
                    
                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)
    
    # No forward pass needed

class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer decoder for acoustic tokens"""
    
    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    _no_split_modules = ["TokenizerDecoder"]

    def __init__(self, config):
        super().__init__(config)
        
        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
            
        # Parse decoder depths if provided
        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split('-')]
        else:
            # Default: use reversed encoder depths if decoder_depths is None
            decoder_depths = list(reversed(encoder_depths))
        
        # Create decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = config.decoder_ratios
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        decoder_config.mixer_layer = config.mixer_layer
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize decoder
        self.decoder = TokenizerDecoder(decoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def decode(self, latents, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Convert latent representations back to audio"""
        if latents.shape[1] == self.config.vae_dim:
            pass
        else:
            latents = latents.permute(0, 2, 1)

        audio = self.decoder(latents, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return audio

    # No forward pass needed

class VibeVoiceDecoder(nn.Module):
    """
    VibeVoice acoustic decoder: acoustic latents -> waveform (streaming).

    Wraps `VibeVoiceAcousticTokenizerModel.decode(..., cache=..., use_cache=True)` and
    manages the streaming cache across chunks.
    Batch size = 1 for now.
    """

    def __init__(self, acoustic_tokenizer: "VibeVoiceAcousticTokenizerModel", device: str | torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.acoustic_tokenizer = acoustic_tokenizer.to(self.device)
        self._layer_id_to_name = build_layer_id_to_name(self.acoustic_tokenizer)

    def init_cache(self, batch_size: int) -> VibeVoiceDecoderCache:
        p = next(self.acoustic_tokenizer.parameters())
        return init_vibevoice_decoder_cache(
            acoustic_tokenizer=self.acoustic_tokenizer,
            batch_size=batch_size,
            device=p.device,
            dtype=p.dtype,
        )

    @torch.inference_mode()
    def decode_chunk(self, latents: torch.Tensor, decoder_cache: VibeVoiceDecoderCache) -> tuple[torch.Tensor, VibeVoiceDecoderCache]:
        # latents expected [B, 1, D] or [B, D]
        if latents.ndim == 2:
            latents = latents.unsqueeze(1)

        B = latents.shape[0]
        assert B == 1, "Batch size must be 1 for streaming"
        
        sample_indices = torch.arange(B, device=self.device, dtype=torch.long)

        cache_view = VibeVoiceTensorStreamingCache(self._layer_id_to_name, decoder_cache)

        audio = self.acoustic_tokenizer.decode(
            latents.to(self.device),
            cache=cache_view,
            sample_indices=sample_indices,
            use_cache=True,
            debug=False,
        )

        return audio, decoder_cache
    

__all__ = [
    "VibeVoiceDecoderCache",
    "VibeVoiceDecoder",
]