import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ..utils import get_logger

logger = get_logger(__name__)


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}
_mimi_config = {
    "sample_rate": 24000,
    "channels": 1,
    "frame_rate": 12.5,
    "seanet": _seanet_kwargs,
    "quantizer": _quantizer_kwargs,
    "transformer": _transformer_kwargs,
}


class _CodebookForwardResult(NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class _VQForwardResult(NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def zero_scalar(device) -> torch.Tensor:
    """Returns a 0. value on the given device without introducing a synchronization point."""
    return torch.zeros([1], device=device)[0]


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.

    Buffers:
        cluster_usage (torch.Tensor): EMA of the cluster usage per batch, e.g. this will
            be dependent on the batch size etc.
        embedding_sum (torch.Tensor): EMA of the sum of the assigned points to each cluster.
            In particular, this can be normalized by `cluster_usage` to obtain the
            actual cluster centroids.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        check_unused_every: int = 5,
    ):
        super().__init__()
        self.decay = decay

        self.dim = dim
        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.replaced_usage_ratio = replaced_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False

        self._initialized: torch.Tensor
        self.cluster_usage: torch.Tensor
        self.embedding_sum: torch.Tensor
        self._embedding: torch.Tensor
        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        embedding = torch.zeros(codebook_size, dim)
        self.register_buffer("embedding_sum", embedding)
        self.register_buffer("_embedding", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Mapping old names to new names
        mappings = {
            "inited": "_initialized",
            "cluster_size": "cluster_usage",
            "embed_avg": "embedding_sum",
            "embed_sum": "embedding_sum",
        }
        for old_name, new_name in mappings.items():
            old_name = prefix + old_name
            if old_name in state_dict:
                value = state_dict.pop(old_name)
                if new_name is not None:
                    state_dict[prefix + new_name] = value
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = (
                self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            )
            self.register_buffer("_embedding", embedding, persistent=False)
            return embedding
        return self._embedding

    @property
    def initialized(self) -> bool:
        """Cached version of self._initialized,
        This assumes that once the module is initialized, it will never go back to the uninitialized state."""
        if not self._cached_initialized:
            self._cached_initialized = bool(self._initialized.item())
        return self._cached_initialized

    # def _init_embedding(self, data: torch.Tensor) -> None:
    #     # Initialize the codebook, e.g. using kmeans.
    #     if self.initialized:
    #         return

    #     rank = 0
    #     if _is_distributed():
    #         rank = distributed.get_rank()
    #         # First gathering shapes in case not all GPUs have the same effective batch size.
    #         # then gathering the actual content.
    #         if rank == 0:
    #             other_shapes: List[torch.Size] = [None] * distributed.get_world_size()  # type: ignore
    #             distributed.gather_object(data.shape, other_shapes)
    #             other_data: List[torch.Tensor] = [
    #                 torch.empty(shape, device=data.device, dtype=data.dtype) for shape in other_shapes]
    #             distributed.gather(data, other_data)
    #             data = torch.cat(other_data, dim=0)
    #         else:
    #             distributed.gather_object(data.shape)
    #             distributed.gather(data)
    #     if rank == 0:
    #         embedding, cluster_usage = _run_kmeans(data, self.codebook_size)
    #         self.embedding_sum.data.copy_(embedding * cluster_usage[:, None])
    #         self.cluster_usage.data.copy_(cluster_usage)
    #         self._initialized.data.fill_(1)
    #     # Make sure all buffers across workers are in sync after initialization
    #     self._broadcast_buffers()

    # def _broadcast_buffers(self) -> None:
    #     if _is_distributed():
    #         for buffer in self.buffers():
    #             distributed.broadcast(buffer, 0)

    # def _replace_expired_codes(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
    #     # Replaces expired centroids, as indicated by `mask` (a true value indicate the code needs to be replaced).
    #     # The new codes are sampled from the batch `samples`.
    #     new_vectors = _sample_vectors(samples, self.codebook_size)
    #     replace_cluster_usage = (
    #         self.replaced_usage_ratio * self.cluster_usage.sum() / self.codebook_size
    #     )
    #     self.embedding_sum[:] = torch.where(
    #         mask[:, None], replace_cluster_usage * new_vectors, self.embedding_sum
    #     )
    #     self.cluster_usage[:] = torch.where(
    #         mask, replace_cluster_usage, self.cluster_usage
    #     )

    # def _check_expired_codes(self, batch_samples: torch.Tensor) -> torch.Tensor:
    #     # Checks whether some centroids are under utilized, and replace them if necessary.
    #     if not self.initialized:
    #         return zero_scalar(batch_samples.device)

    #     self._next_unused_check -= 1
    #     if self._next_unused_check > 0:
    #         return zero_scalar(batch_samples.device)
    #     # we don't check every iteration to avoid having too many sync points.
    #     self._next_unused_check = self.check_unused_every
    #     threshold_cluster_usage = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size
    #     expired_codes = self.cluster_usage < threshold_cluster_usage

    #     assert batch_samples.dim() == 2
    #     self._replace_expired_codes(batch_samples, mask=expired_codes)
    #     self._broadcast_buffers()

    #     return expired_codes.float().mean()

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # Flattens all the dimensions but the last one, e.g. return a vector of shape `[N, D]`.
        x = rearrange(x, "... d -> (...) d")
        return x

    def _reshape_codes(self, codes: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return codes.view(*shape[:-1])

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        # Projects each vector in `x` over the nearest centroid and return its index.
        # `x` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        assert x.dim() == 2
        dists = torch.cdist(x[None], self.embedding[None], p=2)[0]
        codes = dists.argmin(dim=-1)
        return codes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
        The codes are defined as the indexes of the centroids nearest to each vector in `x`.
        """
        assert x.dtype.is_floating_point, f"Input should be floats, got {x.dtype}"
        shape = x.shape
        x = self._reshape_input(x)
        codes = self._quantize(x)
        codes = self._reshape_codes(codes, shape)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
        corresponding to the centroids associated to each code index.
        """
        assert (
            not codes.dtype.is_floating_point
        ), f"Codes should be integers, got {codes.dtype}"
        quantized = F.embedding(codes, self.embedding)
        return quantized

    def forward(
        self, x: torch.Tensor, initialize: bool = True
    ) -> _CodebookForwardResult:
        shape = x.shape
        x = self._reshape_input(x)

        flat_codes = self._quantize(x)
        codes = self._reshape_codes(flat_codes, shape)
        quantized = self.decode(codes)
        metrics: Dict[str, torch.Tensor] = {}

        return _CodebookForwardResult(quantized, codes, metrics)


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        check_unused_every (int): Check for unused centroids every `check_unused_every` iterations.
            This is to avoid too many synchronization points.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_usage_ratio=threshold_usage_ratio,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def embedding(self):
        return self._codebook.embedding

    @property
    def initialized(self):
        return self._codebook.initialized

    def _rearrange_input(self, x):
        x = rearrange(x, "b d n -> b n d")
        return x

    def _rearrange_output(self, quantized):
        quantized = rearrange(quantized, "b n d -> b d n")
        return quantized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` into discrete integer codes."""
        x = self._rearrange_input(x)
        x = self.project_in(x)
        codes = self._codebook.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts integer codes into quantized vectors."""
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)
        return quantized

    def forward(self, x: torch.Tensor, initialize: bool = True) -> _VQForwardResult:
        x = self._rearrange_input(x)
        quantized, codes, metrics = self._codebook(x, initialize=initialize)

        # if self.training:
        #     quantized = x + (quantized - x).detach()
        #     loss = F.mse_loss(x, quantized.detach())
        # else:
        #     loss = zero_scalar(x.device)
        loss = zero_scalar(x.device)

        quantized = self.project_out(quantized)
        quantized = self._rearrange_output(quantized)

        return _VQForwardResult(quantized, codes, loss, metrics)


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, codebook_offset: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.codebook_offset = codebook_offset

    def forward(
        self, x: torch.Tensor, n_q: Optional[int] = None
    ) -> _VQForwardResult:
        """
        Args:
            x (torch.Tensor): input tensor to quantize, of shape `[B, C, T]`.
            n_q (int or None): if provided, number of codebook levels to use in RVQ.
        """

        quantized_out = zero_scalar(x.device)
        residual = x

        all_losses = []
        all_codes = []
        all_metrics: Dict[str, torch.Tensor] = {}

        n_q = n_q or len(self.layers)
        previous_layer_is_initialized = True

        for i, layer in enumerate(self.layers[:n_q]):  # type: ignore
            # if self.training:
            #     this_layer_is_initialized = layer.initialized
            # We only allow the kmeans initialization if the previous layer is already initialized from the previous
            # iterations, this is to avoid learning the subsequent kmeans on the same batch, which would eventually
            # lead to its exhaustion and running kmeans on 0 values.
            quantized, codes, loss, metrics = layer(
                residual, initialize=previous_layer_is_initialized
            )
            # if self.training:
            #     previous_layer_is_initialized = this_layer_is_initialized  # type: ignore

            quantized = quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_codes.append(codes)
            all_losses.append(loss)

            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value / n_q
                else:
                    all_metrics[key] = value / n_q
                all_metrics[key + f"_{i + self.codebook_offset}"] = value

        # if self.training:
        #     # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
        #     quantized_out = x + (quantized_out - x).detach()
        #     to_average = []
        #     for layer in self.layers:
        #         assert isinstance(layer, VectorQuantization)
        #         to_average += [layer._codebook.cluster_usage, layer._codebook.embedding_sum]
        #         _average_tensors(to_average)

        out_losses, out_codes = map(torch.stack, (all_losses, all_codes))
        return _VQForwardResult(quantized_out, out_codes, out_losses, all_metrics)

    def encode(self, x: torch.Tensor, n_q: Optional[int] = None) -> torch.Tensor:
        """Encodes `x` into discrete integer codes. If `n_q` is provided, only uses the first `n_q` codebook levels."""
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:  # type: ignore
            assert isinstance(layer, VectorQuantization)
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts the integer codes into quantized vectors."""
        quantized = zero_scalar(codes.device)
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)
        return quantized


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):
    """Base class for quantizers."""

    def __init__(self):
        super().__init__()
        self._ema_frozen = False

    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def cardinality(self) -> int:
        """Cardinality of each codebook."""
        raise NotImplementedError()

    @property
    def total_codebooks(self) -> int:
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self) -> int:
        """Number of active codebooks."""
        raise NotImplementedError()

    @property
    def semantic_quantizer(self) -> 'BaseQuantizer':
        """This returns the quantizer that models the first level of the hierarchy (typically semantic).

        In this case, it's the quantizer itself.
        """
        return self

    @property
    def acoustic_quantizer(self) -> 'BaseQuantizer':
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic).

        In this case, it's the quantizer itself.
        """
        return self

    def set_num_codebooks(self, n: int) -> None:
        """Set the number of active codebooks."""
        raise NotImplementedError()

    @property
    def ema_frozen(self) -> bool:
        """Whether to apply ema to the codebooks."""
        return self._ema_frozen

    def ema_frozen_(self, ema_frozen: bool) -> None:
        """Set whether ema should be applied to the codebooks."""
        self._ema_frozen = ema_frozen


class ResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        input_dimension (None or int): dimension of the input, defaults to `dimension` if not provided.
        output_dimension (None or int): dimension of the output, defaults to `dimension` if not provided.
        n_q (int): Number of vector quantizers used.
        q_dropout (bool): Random quantizer drop out at train time.
        no_quantization_rate (float): Gives the probability of applying no quantization at all
            at train time. The RVQ codebooks will still get the input value to learn the proper codebook.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        threshold_usage_ratio (float): Defines the threshold for the cluster usage under which a centroid
            is replaced. This is expressed as a fraction of the usage a centroid would get under
            a uniform distribution, so that it doesn't depend on the batch size etc.
        replaced_usage_ratio (float): When replacing a centroid, use this as an initial centroid usage,
            to avoid the centroid getting replaced too quickly.
        codebook_offset (int): Offset to use for the codebook indices. This is useful when using multiple quantizers
            such as in SplitResidualVectorQuantizer.
        force_projection (bool): Whether to force input and output projections even when dimension is constant.
        generator_seed (int or None): seed used to initialize the RNG used for no quantization.
    """

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        threshold_usage_ratio: float = 0.1,
        replaced_usage_ratio: float = 1.0,
        codebook_offset: int = 0,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.rng_dropout = random.Random(1234)
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            threshold_usage_ratio=threshold_usage_ratio,
            replaced_usage_ratio=replaced_usage_ratio,
            codebook_offset=codebook_offset,
        )

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        n_q = self.n_q
        x = self.input_proj(x)
        if self.training and self.q_dropout:
            n_q = self.rng_dropout.randint(1, self.n_q)
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        quantized, codes, commit_loss, metrics = self.vq(x, n_q=n_q)
        B, _, _ = quantized.shape
        if self.training and self.no_quantization_rate > 0:
            mask = (torch.rand(B, 1, 1, device=x.device) <= self.no_quantization_rate).float()
            quantized = x * mask + (1 - mask) * quantized
        quantized = self.output_proj(quantized)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss), metrics=metrics)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        if x.shape[-1] == 0:
            return torch.empty((x.shape[0], n_q, 0), device=x.device, dtype=torch.int64)

        x = self.input_proj(x)
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.

        All elements must be 0 <= c < self.cardinality, otherwise a dramatic CUDA crash
        occurs. We can't check this condition though, to avoid a synchronization point.
        """
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = self.output_proj(quantized)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n >= 0 and n <= self.max_n_q
        self.n_q = n

    @property
    def cardinality(self) -> int:
        return self.bins


class SplitResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            codebook_offset=1,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def _renorm_and_add(
        self,
        first_val: torch.Tensor,
        rest_val: torch.Tensor,
        n_q_semantic: int,
        n_q_acoustic: int,
    ):
        """Renormalizes values from `rvq_first` and `rvq_rest` and adds them.

        This allows correcting statistics that are normalized by the number of quantizers. To renormalize, we use the
        number of quantizers that are actually used, e.g. taking into account quantizer dropout.
        """
        n_q = n_q_semantic + n_q_acoustic
        renorm_first_val = first_val * n_q_semantic / n_q
        renorm_rest_val = rest_val * n_q_acoustic / n_q
        return renorm_first_val + renorm_rest_val

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        semantic_result = self.rvq_first(x, frame_rate)
        if self.n_q == self.n_q_semantic:
            return semantic_result
        acoustic_result = self.rvq_rest(x, frame_rate)
        full_quantized_emb = semantic_result.x + acoustic_result.x
        full_quantized_codes = torch.cat(
            [semantic_result.codes, acoustic_result.codes], dim=1
        )
        # This is the actual number of quantizers used,  e.g. taking into account quantizer dropout.
        n_q_semantic = semantic_result.codes.shape[1]
        n_q_acoustic = acoustic_result.codes.shape[1]
        full_quantized_bandwidth = semantic_result.bandwidth + acoustic_result.bandwidth
        full_quantized_penalty = self._renorm_and_add(
            semantic_result.penalty, acoustic_result.penalty, n_q_semantic, n_q_acoustic
        )
        full_quantized_metrics = semantic_result.metrics
        for key, value in acoustic_result.metrics.items():
            if key in full_quantized_metrics:
                full_quantized_metrics[key] = self._renorm_and_add(
                    full_quantized_metrics[key], value, n_q_semantic, n_q_acoustic
                )
            else:
                full_quantized_metrics[key] = value
        return QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            full_quantized_bandwidth,
            penalty=full_quantized_penalty,
            metrics=full_quantized_metrics,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        codes = self.rvq_first.encode(x)
        if self.n_q > self.n_q_semantic:
            acoustic_codes = self.rvq_rest.encode(x)
            codes = torch.cat([codes, acoustic_codes], dim=1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized

    @property
    def total_codebooks(self):
        return self.rvq_first.max_n_q + self.rvq_rest.max_n_q

    @property
    def num_codebooks(self):
        return self.rvq_first.num_codebooks + self.rvq_rest.num_codebooks

    @property
    def n_q(self):
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def dimension(self):
        return self.rvq_first.dimension

    @property
    def semantic_quantizer(self) -> ResidualVectorQuantizer:
        """This returns the quantizer that models the first level of the hierarchy (typically semantic)."""
        return self.rvq_first

    @property
    def acoustic_quantizer(self) -> ResidualVectorQuantizer:
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic)."""
        return self.rvq_rest

    def set_num_codebooks(self, n: int):
        assert n >= self.n_q_semantic and n <= self.total_codebooks, (n, self.n_q_semantic, self.total_codebooks)
        self.rvq_rest.set_num_codebooks(n - self.n_q_semantic)

    @property
    def cardinality(self) -> int:
        assert self.rvq_rest.cardinality == self.rvq_first.cardinality
        return self.rvq_first.cardinality


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    time_before_heads: bool = False,
):
    """
    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offset (int): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
        time_before_heads (bool):  if True, expected [B, T, H, D], else [B, H, T ,D]
    """

    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape
    assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
    if time_before_heads:
        ts = ts.view(B, -1, 1, 1)
    else:
        ts = ts.view(B, 1, -1, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(*dims, D), ko.view(*dims, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
    """

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offset, self.max_period, time_before_heads)


def gating_forward_kernel(
    weight_in: torch.Tensor, weight_out: torch.Tensor, activation, x: torch.Tensor
):
    x = F.linear(x, weight_in)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = F.linear(x, weight_out)
    return x


def gating_forward_generic(
    linear_in: nn.Module,
    linear_out: nn.Module,
    activation,
    x: torch.Tensor
):
    x = linear_in(x)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = activation(x[..., 0, :]) * x[..., 1, :]
    x = linear_out(x)
    return x


class ActivationGating(nn.Module):
    """
    Gating FFN layer, using the given activation.
    Args:
        dim (int): dimension of the input and output of the transformer.
        activation (any callable Tensor to Tensor): activation function to use.
        **factory_kwargs: other kwargs passed to the linear layer, in particular device and dtype.
    """

    _fsdp_final = True

    def __init__(self, dim: int, dim_feedforward: int, activation, quantized: bool = False, **factory_kwargs):
        super().__init__()
        # We should have 8 d^2 param, instead we will have
        # 2 * h * d + h * d = 3 h * d = 8 d^2
        # so h = 8 d / 3 but following Hervé's advice we use 21 / 8 as an approx.
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3

        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)

        # We try to follow the default PyTorch MHA convention, to easily compare results.

        self.activation = activation

    def forward(self, x: torch.Tensor):
        if isinstance(self.linear_in, nn.Linear):
            assert isinstance(self.linear_out, nn.Linear)
            return gating_forward_kernel(
                self.linear_in.weight, self.linear_out.weight, self.activation, x
            )
        else:
            return gating_forward_generic(
                self.linear_in,
                self.linear_out,
                self.activation,
                x
            )


def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(torch.nn.functional, name)
    elif name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def _make_gating(
    name: str, dim: int, dim_feedforward: int,
    **factory_kwargs
) -> nn.Module:
    return ActivationGating(
        dim, dim_feedforward, _get_activation(name), **factory_kwargs
    )


def make_gating(
    name: str, dim: int, dim_feedforward: int, **factory_kwargs
) -> nn.Module:
    gating = _make_gating(name, dim, dim_feedforward, **factory_kwargs)
    if isinstance(gating.linear_in, nn.Linear):
        max_params = 2 * dim * dim_feedforward
        params = sum(p.numel() for p in gating.parameters())
        assert (
            params <= max_params
        ), f"{name} gating has {params} params, max is {max_params}"
    return gating


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
    dtype: Optional[torch.dtype],
    eps: float,
):
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full(
        [], max_period, device=positions.device, dtype=dtype
    )  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def set_attention_context(model: nn.Module, context: Optional[int] = None) -> None:
    """Deactivates or changes the context span (in time steps) in a model.
    Args:
        model (nn.Module): model over which to look for attentions.
        context (int or None): new temporary context value.

    ..Note:: this is not a context manager but a plain function changing the context forever.
        Initially, it was a context manager, but that led to interesting bugs when using
        activation checkpointing, with the context being inconsistent between the forward
        and backward.
    """
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context


class KVCacheResult(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


class RingKVCache:
    """Efficient streaming KVCache to be compatible with Cuda Graph.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads in the attention.
        dim_per_head (int): Dimension per head.
        device (torch.device): Device on which to initialize the cache.
        dtype (torch.dtype): dtype to use for the cache.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        respect_exec_mask: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.respect_exec_mask = respect_exec_mask
        if self.respect_exec_mask:
            self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.end_offset[:] = torch.where(
            reset_mask,
            torch.zeros_like(self.end_offset),
            self.end_offset,
        )

    def complete(self, k: torch.Tensor, v: torch.Tensor, exec_mask: torch.Tensor) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        assert T > 0
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype)
        indexes = indexes + self.end_offset.view(-1, 1)
        indexes = indexes % self.capacity
        if self.respect_exec_mask:
            # indexes is [B, T]
            # k is [B, H, T, D]
            # cache is [B, H, T', D]
            this_indexes = indexes.view(B, 1, T, 1)
            this_indexes = this_indexes.expand(-1, H, T, D)
            self.cache[0].scatter_(2, this_indexes, k)
            self.cache[1].scatter_(2, this_indexes, v)
        else:
            self.cache[0].index_copy_(2, indexes[0], k)
            self.cache[1].index_copy_(2, indexes[0], v)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )

        # end_index correspond to the actual index where the last value was written.
        last_offset = self.end_offset.view(-1, 1) + T - 1
        end_index = last_offset % self.capacity
        delta = indexes - end_index

        # We know that if `index == end_index`, then we should output `self.end_offset`.
        # If `index = end_index - 1` we should output `self.end_offset - 1`
        # If `index = end_index - n` we should output `self.end_offset - n`
        # Now, for `index == end_index + 1` , we actually have the oldest entry in the cache,
        # so we should output `end_index + 1 - self.capacity`

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        if self.respect_exec_mask:
            self.end_offset[:] = torch.where(
                exec_mask,
                self.end_offset + T,
                self.end_offset)
        else:
            self.end_offset.add_(T)
        invalid = indexes >= self.end_offset.view(-1, 1)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


def apply_weights_per_step(modules: nn.ModuleList, schedule: list[int] | None,
                           x: torch.Tensor, offset: int | None) -> torch.Tensor:
    """Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        modules (nn.ModuleList): apply weights per step.
        schedule (list[int] or None): schedule for weight sharing.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    """

    if len(modules) == 1:
        return modules[0](x)

    assert offset is not None, "Out of sync execution with weights per step."

    ys: list[torch.Tensor] = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        if schedule is not None:
            module_index = schedule[module_index]
        y = modules[module_index](x[:, t: t + 1])
        ys.append(y)
    out = torch.cat(ys, 1)
    return out


class StreamingMultiheadAttention(nn.Module):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Number of time steps the attention can access to.
            When causal, can access `context` time steps into the past, and when non causal,
            can access `context // 2` steps in the past, and the same in the future.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        weights_per_step_schedule (list[int] | None): if provided, some steps will share weights when
            `weights_per_step` is True, e.g. step `I` will use weights `schedule[I]`.
        cross_attention (bool): True if this is to be used as a cross attention.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        cross_attention: bool = False,
        cache_cross_attention: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.cross_attention = cross_attention
        self.cache_cross_attention = cache_cross_attention
        if cross_attention:
            assert not weights_per_step, "weights_per_step not supported for cross attention."
            assert rope is None, "rope and cross_attention makes no sense."
            assert not causal, "causal and cross attention makes no sense."
            # We do not want to activate the streaming KVCache if we are a cross attention.
            # self.set_streaming_detached(True)

        out_dim = 3 * embed_dim
        mult = 1
        if weights_per_step:
            if weights_per_step_schedule:
                assert len(weights_per_step_schedule) == weights_per_step
                mult = max(weights_per_step_schedule) + 1
            else:
                mult = weights_per_step
        self.mult = mult

        # Split in one linear per step
        self.out_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )
        self.in_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )

        self._register_load_state_dict_pre_hook(StreamingMultiheadAttention._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            'in_proj_weight': 'in_projs.{i}.weight',
            'in_proj.weight': 'in_projs.{i}.weight',
            'in_proj.lora_A.weight': 'in_projs.{i}.lora_A.weight',
            'in_proj.lora_B.weight': 'in_projs.{i}.lora_B.weight',
            'out_proj.weight': 'out_projs.{i}.weight',
            'out_proj.lora_A.weight': 'out_projs.{i}.lora_A.weight',
            'out_proj.lora_B.weight': 'out_projs.{i}.lora_B.weight',
        }

        mult = module.mult
        # _scb suffix is for quantized data.
        for suffix in ['', '_scb']:
            for source, target in mappings.items():
                this_source = prefix + source + suffix
                if this_source in state_dict:
                    weight = state_dict[this_source]
                    _, *OD = weight.shape
                    weight = weight.view(mult, -1, *OD)
                    for i in range(mult):
                        this_target = prefix + target.format(i=i) + suffix
                        state_dict[this_target] = weight[i]
                    state_dict.pop(this_source)

    def _complete_kv(self, k, v) -> KVCacheResult:
        return KVCacheResult.from_kv(k, v)
        # state = self._streaming_state
        # if state is None or state.kv_cache is None:
        #     return KVCacheResult.from_kv(k, v)
        # else:
        #     return state.kv_cache.complete(k, v, state.exec_mask)

    def _compute_cross_attention(
            self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cross_attention
        assert key is value
        in_proj = self.in_projs[0]
        assert in_proj.bias is None
        assert isinstance(in_proj, nn.Linear)
        dim = in_proj.weight.shape[0] // 3
        kv = nn.functional.linear(key, in_proj.weight[dim:])
        k, v = rearrange(kv, "b t (p h d) -> p b h t d", p=2, h=self.num_heads)
        return k, v

    # def update_streaming_cross_attention_src(
    #         self, cross_attention_src: torch.Tensor) -> None:
    #     state = self._streaming_state
    #     assert state is not None
    #     assert self.cross_attention
    #     k, v = self._compute_cross_attention(cross_attention_src, cross_attention_src)
    #     if state.k_cross is None:
    #         state.k_cross = k
    #         state.v_cross = v
    #     else:
    #         assert state.v_cross is not None
    #         state.k_cross[:] = k
    #         state.v_cross[:] = v

    def _get_cross_attention(
            self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # state = self._streaming_state
        # if state is not None and state.k_cross is not None:
        #     assert state.v_cross is not None
        #     return state.k_cross, state.v_cross
        k, v = self._compute_cross_attention(key, value)
        # if state is not None and self.cache_cross_attention:
        #     state.k_cross = k
        #     state.v_cross = v
        return k, v

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # state = self._streaming_state
        B, T = query.shape[:2]

        offset = torch.zeros(B, device=query.device, dtype=torch.long)
        offset_cpu = 0
        # if state is None:
        #     offset = torch.zeros(B, device=query.device, dtype=torch.long)
        #     offset_cpu = 0
        # else:
        #     offset = state.offset
        #     offset_cpu = state.offset_cpu

        if self.cross_attention:
            assert len(self.in_projs) == 1
            in_proj = self.in_projs[0]
            assert in_proj.bias is None
            assert isinstance(in_proj, nn.Linear)
            dim = in_proj.weight.shape[0] // 3
            q = nn.functional.linear(query, in_proj.weight[:dim])
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k, v = self._get_cross_attention(key, value)
        else:
            projected = apply_weights_per_step(
                self.in_projs, self.weights_per_step_schedule, query, offset_cpu)

            q, k, v = rearrange(
                projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
            )
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v)
        pos_k = pos_k[:, None]
        if self.causal:
            pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(
                -1, 1)
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
            attn_bias = attn_bias[:, None]
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = apply_weights_per_step(
            self.out_projs, self.weights_per_step_schedule, x, offset_cpu)

        # if state is not None and not self.cross_attention:
        #     state.offset[:] = torch.where(
        #         state.exec_mask,
        #         state.offset + T,
        #         state.offset)
        #     state.offset_cpu += T
        return x


class StreamingTransformerLayer(nn.Module):
    """TransformerLayer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        norm (str): Normalization to use. Currently, only 'layer_norm' is supported.
        layer_scale (float, optional): If not None, LayerScale will be used with the given value as initial scale.
        gating (str): if provided, replaces FFN with special gating, like GLU, GSiGLU etc.
        weights_per_step (int): use different weights per time step. If non zero, should correspond to the
            number of possible time steps.
        weights_per_step_schedule (list[int] | None): if provided, some steps will share weights when
            `weights_per_step` is True, e.g. step `I` will use weights `schedule[I]`.
        skip_self_attn: If true, skips the self attention module and the norm
        cross_attention (bool): If True, expect to get secondary input for cross-attention.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        activation=F.gelu,
        skip_self_attn: bool = False,
        cross_attention: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: Dict[str, Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                weights_per_step_schedule=weights_per_step_schedule,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: Optional[nn.Module] = None
        self.linear1: Optional[nn.Module] = None
        self.linear2: Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        num_weights = 1
        if weights_per_step is not None:
            num_weights = weights_per_step
            if weights_per_step_schedule is not None:
                assert len(weights_per_step_schedule) == weights_per_step
                num_weights = max(weights_per_step_schedule) + 1
        if isinstance(dim_feedforward, list):
            assert dim_feedforward
            assert len(dim_feedforward) == num_weights, (
                "Length of dim_feedforward must match weights_per_step,"
                f" got {len(dim_feedforward)} != {num_weights}"
            )
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * num_weights
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.cross_attention: StreamingMultiheadAttention | None = None
        if cross_attention:
            self.cross_attention = StreamingMultiheadAttention(
                cross_attention=True, **attn_kwargs, **factory_kwargs)  # type: ignore
            # Cross attention norm is always a layer norm, for no specific reason.
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)  # type: ignore

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
            if cross_attention:
                self.layer_scale_cross = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
            if cross_attention:
                self.layer_scale_cross = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # state = self._streaming_state
        offset = 0
        # if state is not None:
        #     offset = state.offset_cpu
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        elif self.weights_per_step:
            assert isinstance(self.gating, nn.ModuleList)
            update = apply_weights_per_step(self.gating, self.weights_per_step_schedule, x, offset)
        else:
            update = self.gating(x)
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor):
        if self.skip_self_attn:
            return x
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig.to(update) + self.layer_scale_1(update)

    def _cross_attention_block(self, x: torch.Tensor,
                               cross_attention_src: torch.Tensor) -> torch.Tensor:
        assert self.cross_attention is not None
        x_orig = x
        x = self.norm_cross(x)
        # queries are from src, keys and values from cross_attention_src.
        update = self.cross_attention(x, cross_attention_src, cross_attention_src)
        return x_orig + self.layer_scale_cross(update)

    def forward(self, x: torch.Tensor, cross_attention_src: torch.Tensor | None = None):
        x = self._sa_block(x)
        if self.cross_attention is not None:
            assert cross_attention_src is not None
            x = self._cross_attention_block(x, cross_attention_src)
        else:
            assert cross_attention_src is None
        x = self._ff_block(x)
        # state = self._streaming_state
        # if state:
        #     state.offset_cpu += x.shape[1]
        return x


class StreamingTransformer(nn.Module):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        causal (bool): Causal mask applied automatically.
        context (int, optional): Receptive field for the causal mask, infinite if None.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, sin_rope, or none).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `StreamingTransformerLayer`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: Optional[Tuple[float, float]] = None,
        layer_class: Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        quantize: bool = False,
        checkpointing: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.checkpointing = checkpointing

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )
            # if quantize:
            #     # Quantizing layers one by one to avoid taking too much space during init.
            #     self.layers[-1].to(device=device, dtype=dtype)
            #     replace_linear_with_qlinear(self.layers[-1])

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        dtype_input = x.dtype
        offsets = torch.zeros(1, dtype=torch.long, device=x.device)
        # state = self._streaming_state
        # if state is None:
        #     offsets = torch.zeros(1, dtype=torch.long, device=x.device)
        # else:
        #     offsets = state.offsets

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        # if state is not None:
        #     state.offsets[:] = torch.where(
        #         state.exec_mask,
        #         state.offsets + T,
        #         state.offsets)
        return x.to(dtype_input)


class ProjectedTransformer(nn.Module):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(self, x, *args, **kwargs):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, *args, **kwargs)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])

class TransposedLayerNorm(nn.Module):
    """LayerNorm for [B, C, T] inputs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(**kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(1, 2)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return nn.utils.weight_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
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


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(
            nn.Conv1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        return x


@dataclass
class _StreamingConv1dState:
    batch_size: int
    device: torch.device
    previous: torch.Tensor
    first: torch.Tensor

    # def reset(self, reset_mask: torch.Tensor):
    #     super().reset(reset_mask)
    #     self.previous[:] = torch.where(reset_mask.view(-1, 1, 1), torch.zeros_like(self.previous), self.previous)
    #     self.first[:] = torch.where(reset_mask, torch.ones_like(self.first), self.first)


class StreamingConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ['constant', 'replicate'], pad_mode
        self.pad_mode = pad_mode
        assert causal
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).",
                stacklevel=2
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    @property
    def _stride(self) -> int:
        return self.conv.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.conv.conv.dilation[0]
        return (
            self._kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        stride = self._stride
        # Effective kernel size accounting for dilation.
        kernel = self._effective_kernel_size
        param = next(iter(self.parameters()))
        dtype = param.dtype
        device = param.device
        previous = torch.zeros(batch_size, self.conv.conv.in_channels, kernel - stride,
                               dtype=dtype, device=device)
        first = torch.ones(batch_size, device=device, dtype=torch.bool)
        return _StreamingConv1dState(batch_size, device, previous, first)

    def forward(self, x):
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        # state = self._streaming_state
        # if state is None:
        #     state = self._init_streaming_state(B)
        state = self._init_streaming_state(B)
        TP = state.previous.shape[-1]
        if TP and self.pad_mode == 'replicate':
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state.previous[:] = torch.where(
                state.first.view(-1, 1, 1), # & state.exec_mask.view(-1, 1, 1),
                init,
                state.previous)
        if TP:
            x = torch.cat([state.previous, x], dim=-1)
        y = self.conv(x)
        if TP:
            state.previous[:] = x[..., -TP:]
            # state.previous[:] = torch.where(
            #     state.exec_mask.view(-1, 1, 1),
            #     x[..., -TP:],
            #     state.previous)
            if self.pad_mode == 'replicate':
                state.first = torch.zeros_like(state.first)
                # state.first = torch.where(
                #     state.exec_mask,
                #     torch.zeros_like(state.first),
                #     state.first,
                # )
        return y


class StreamingConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        assert trim_right_ratio == 1.
        assert causal
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    @property
    def _stride(self) -> int:
        return self.convtr.convtr.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.convtr.convtr.kernel_size[0]

    def forward(self, x):
        B, C, T = x.shape
        K = self._kernel_size
        S = self._stride
        # state = self._streaming_state

        y = self.convtr(x)
        y = unpad1d(y, (0, K - S))
        # if state is None:
        #     y = unpad1d(y, (0, K - S))
        # else:
        #     PT = state.partial.shape[-1]
        #     if PT > 0:
        #         y[..., :PT] += state.partial
        #         bias = self.convtr.convtr.bias
        #         for_partial = y[..., -PT:]
        #         if bias is not None:
        #             for_partial -= bias[:, None]
        #         state.partial[:] = torch.where(
        #             state.exec_mask.view(-1, 1, 1),
        #             for_partial,
        #             state.partial)
        #         y = y[..., :-PT]
        return y


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    If `causal` is True, the output uses a causal convolution.
    """

    def __init__(
        self,
        stride: int,
        dimension: Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            pad_mode="replicate",
        )
        if not learnt:
            actual_conv = self.conv.conv.conv
            actual_conv.weight.requires_grad_(False)
            actual_conv.weight.data.fill_(1.0 / (2 * stride))

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.conv(x)
        if not self.learnt:
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(
        self,
        stride: int,
        dimension: Optional[int] = None,
        causal: bool = False,
        learnt: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.learnt = learnt
        self.channel_wise = channel_wise
        groups = 1
        if learnt:
            assert dimension is not None, "Dimension required for learnt convolutions."
            in_channels = dimension
            out_channels = dimension
            if channel_wise:
                groups = dimension
        else:
            in_channels = 1
            out_channels = 1

        self.convtr = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
        )
        if not learnt:
            actual_convtr = self.convtr.convtr.convtr
            actual_convtr.weight.requires_grad_(False)
            actual_convtr.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor):
        batch_size = len(x)
        if not self.learnt:
            x = rearrange(x, "b c t -> (b c) () t")
        y = self.convtr(x)
        if not self.learnt:
            x_for_normalization = torch.ones_like(x[:1])
            normalization = self.convtr(x_for_normalization)
            y = y / normalization
            y = rearrange(y, "(b c) () t -> b c t", b=batch_size)
        return y


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: List[int] = [3, 1],
        dilations: List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: Dict[str, Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations, strict=False)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamingConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamingConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        u, v = self.shortcut(x), self.block(x)
        assert u.shape == v.shape, (u.shape, v.shape, x.shape)
        return u + v


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
        mask_fn (nn.Module): Optional mask function to apply after convolution layers.
        mask_position (int): Position of the mask function, with mask_position == 0 for the first convolution layer,
            mask_position == 1 for the first conv block, etc.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        mask_fn: Optional[nn.Module] = None,
        mask_position: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = 1
        model: List[nn.Module] = [
            StreamingConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        if mask_fn is not None and mask_position == 0:
            model += [mask_fn]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2
            if mask_fn is not None and mask_position == i + 1:
                model += [mask_fn]

        model += [
            act(**activation_params),
            StreamingConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: Optional[str] = None,
        final_activation_params: Optional[dict] = None,
        norm: str = "none",
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
            self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: List[nn.Module] = [
            StreamingConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = (
                "none"
                if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                else norm
            )
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamingConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y



class MimiModel(nn.Module):
    """Mimi model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: Optional[nn.Module] = None,
        decoder_transformer: Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        if freeze_quantizer:
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        extra_metrics: Dict[str, torch.Tensor] = {}

        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        emb = self.encoder(x)
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
        # Checking that we have the proper length given the advertised frame rate.
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )

        q_res = self.quantizer(emb, self.frame_rate)
        emb = q_res.x
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = out
        q_res.metrics.update(extra_metrics)
        return q_res

    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"

        # state = self._streaming_state
        frame_size = self.frame_size

        x = pad_for_conv1d(x, frame_size, frame_size)
        emb = self.encoder(x)
        # if state is None:
        #     # The underlying convolutions no longer accept partial inputs,
        #     # `x` needs to be exactly a multiple of the frame size,
        #     # reproducing the previous padding behavior here.
        #     x = pad_for_conv1d(x, frame_size, frame_size)
        #     emb = self.encoder(x)
        # else:
        #     if x.shape[-1] % frame_size != 0 or x.shape[-1] == 0:
        #         raise RuntimeError(
        #             f"Invalid input x of length {x.shape[-1]}. The length must be "
        #             f"a positive multiple of the frame size {frame_size}. "
        #             "You are responsible for buffering accordingly before feeding audio to Mimi.")
        #     emb = state.graphed_encoder(x).clone()
        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
            # if state is None:
            #     (emb,) = self.encoder_transformer(emb)
            # else:
            #     assert state.graphed_tr_enc is not None
            #     (emb,) = state.graphed_tr_enc(emb)
        emb = self._to_framerate(emb)
        return emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x)
        codes = self.quantizer.encode(emb)
        return codes

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        # state = self._streaming_state
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)
            # if state is None:
            #     (emb,) = self.decoder_transformer(emb)
            # else:
            #     assert state.graphed_tr_dec is not None
            #     (emb,) = state.graphed_tr_dec(emb)
        out = self.decoder(emb)
        # if state is None:
        #     out = self.decoder(emb)
        # else:
        #     out = state.graphed_decoder(emb).clone()
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)

class MimiDecoder(nn.Module):
    def __init__(
        self,
        model_repo: str = "kyutai/moshiko-pytorch-bf16",
        model_path: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
        num_codebooks: int = 32,
        mimi_config: dict | None = None,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.num_codebooks = num_codebooks

        if mimi_config is None:
            mimi_config = _mimi_config

        mimi_weight_path = hf_hub_download(model_repo, model_path)

        encoder = SEANetEncoder(**mimi_config['seanet'])
        decoder = SEANetDecoder(**mimi_config['seanet'])
        encoder_transformer = ProjectedTransformer(
            device=device, **mimi_config['transformer']
        )
        decoder_transformer = ProjectedTransformer(
            device=device, **mimi_config['transformer']
        )
        quantizer = SplitResidualVectorQuantizer(
            **mimi_config['quantizer'],
        )
        self.model = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config['channels'],
            sample_rate=mimi_config['sample_rate'],
            frame_rate=mimi_config['frame_rate'],
            encoder_frame_rate=mimi_config['sample_rate'] / encoder.hop_length,
            causal=True,
            resample_method="conv",
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        ).to(device=device)
        self.model.eval()
        if Path(mimi_weight_path).suffix in (".safetensors", ".sft", ".sfts"):
            state = load_file(mimi_weight_path, device=str(device))
            self.model.load_state_dict(state)
        else:
            pkg = torch.load(mimi_weight_path, "cpu")
            self.model.load_state_dict(pkg["model"])
        self.model.set_num_codebooks(num_codebooks)

    @property
    def sample_rate(self):
        return self.model.sample_rate

    def encode(
        self,
        audio: torch.Tensor,
    ):
        return self.model.encode(audio)

    def decode(
        self,
        codes: torch.Tensor,
    ):
        return self.model.decode(codes)
