# Adopted from https://github.com/xingchensong/S3Tokenizer

# Copyright (c)  (Mddct: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple

import onnx
import torch
from einops import rearrange


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    1 for non-padded part and 0 for padded part.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.

    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).

    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).

    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks =
            [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
             [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
             [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)

    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e10
    return mask


def _rename_weights(weights_dict: dict):
    """
    Rename onnx weights to pytorch format.

    Parameters
    ----------
    weight_dict: dict
        The dict containing weights in onnx format

    Returns
    -------
    A new weight dict containing the weights in pytorch format.
    """
    new_weight_dict = {}
    for k, v in weights_dict.items():
        if "quantizer" in k:  # vq or fsq
            if k == "/quantizer/rq/model/layers.0/_codebook/Pow_1":
                new_weight_dict["quantizer._codebook.embed"] = v
            elif "project_down" in k:  # v2
                new_weight_dict[k] = v
        elif "positional_embedding" in k:  # positional emb
            new_weight_dict[k] = v
        elif "conv" in k:  # 1/2 or 1/4 subsample
            new_weight_dict[k] = v
        else:  # transformer blocks
            assert "blocks" in k
            new_k = (
                k[1:]
                .replace("/", ".")
                .replace("MatMul", "weight")
                .replace("Add_1", "bias")
                .replace("Mul", "weight")
                .replace("Add", "bias")
                .replace("mlp.mlp", "mlp")
            ).replace("fsmn_block.Conv", "fsmn_block.weight")

            new_weight_dict[f"encoder.{new_k}"] = v
    return new_weight_dict


def onnx2torch(onnx_path: str, torch_path: str = None, verbose: bool = False):
    """
    Open an onnx file and convert to pytorch format.

    Parameters
    ----------
    onnx_path: str
        The onnx file to open, typically `speech_tokenizer_v1.onnx`

    torch_path: str
        The path to save the torch-formated checkpoint.

    verbose: bool
        Logging info or not.

    Returns
    -------
    A checkpoint dict containing the weights and their names, if torch_path is
    None. Otherwise save checkpoint dict to the desired path.
    """
    onnx_model = onnx.load(onnx_path)
    weights_dict = {}
    initializer_map = {initializer.name: initializer for initializer in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_map:
                ln_bias_name, ln_weight_name = None, None  # for v2 ln
                initializer = initializer_map[input_name]
                if input_name in [
                    "onnx::Conv_1519",
                    "encoders.conv1.weight",
                    "onnx::Conv_2216",
                ]:  # v1_50hz, v1_25hz, v2_25hz
                    weight_name = "encoder.conv1.weight"
                elif input_name in [
                    "onnx::Conv_1520",
                    "encoders.conv1.bias",
                    "onnx::Conv_2217",
                ]:  # v1_50hz, v1_25hz, v2_25hz
                    weight_name = "encoder.conv1.bias"
                elif input_name in [
                    "onnx::Conv_1521",
                    "encoders.conv2.weight",
                    "onnx::Conv_2218",
                ]:
                    weight_name = "encoder.conv2.weight"
                elif input_name in [
                    "onnx::Conv_1522",
                    "encoders.conv2.bias",
                    "onnx::Conv_2219",
                ]:
                    weight_name = "encoder.conv2.bias"
                elif input_name == "encoders.positional_embedding":
                    weight_name = "encoder.positional_embedding"
                elif input_name == "quantizer.project_in.bias":
                    weight_name = "quantizer._codebook.project_down.bias"
                elif input_name == "onnx::MatMul_2536":
                    weight_name = "quantizer._codebook.project_down.weight"
                elif node.op_type == "LayerNormalization":  # in input_name:
                    ln_name = node.name.replace("/LayerNormalization", "")
                    ln_weight_name = ln_name + ".weight"
                    ln_bias_name = ln_name + ".bias"
                else:
                    weight_name = node.name
                if ln_weight_name is not None and ln_bias_name is not None:
                    ln_inputs = node.input
                    scale_name = ln_inputs[1]
                    bias_name = ln_inputs[2]
                    scale = (
                        onnx.numpy_helper.to_array(initializer_map[scale_name]).copy()
                        if scale_name in initializer_map
                        else None
                    )
                    bias = (
                        onnx.numpy_helper.to_array(initializer_map[bias_name]).copy()
                        if bias_name in initializer_map
                        else None
                    )
                    scale.flags.writeable = True
                    bias.flags.writeable = True
                    weight_tensor = torch.from_numpy(scale)
                    bias_tensor = torch.from_numpy(bias)

                    weights_dict[ln_bias_name] = bias_tensor
                    weights_dict[ln_weight_name] = weight_tensor
                else:
                    weight_array = onnx.numpy_helper.to_array(initializer).copy()
                    weight_array.flags.writeable = True
                    weight_tensor = torch.from_numpy(weight_array)
                    if len(weight_tensor.shape) > 2 or weight_name in ["encoder.positional_embedding"]:
                        weights_dict[weight_name] = weight_tensor
                    else:
                        weights_dict[weight_name] = weight_tensor.t()

    new_weights_dict = _rename_weights(weights_dict)
    if verbose:
        for k, v in new_weights_dict.items():
            print(f"{k} : {v.shape} {v.dtype}")
        print(f"PyTorch weights saved to {torch_path}")
    del weights_dict, onnx_model
    if torch_path:
        torch.save(new_weights_dict, torch_path)
    else:
        return new_weights_dict


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8

    use_sdpa: bool = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, : D // 2], xq[:, :, :, D // 2 :]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]

    half_l, half_r = xk[:, :, :, : D // 2], xk[:, :, :, D // 2 :]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


class FSQCodebook(torch.nn.Module):
    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        # h = ((self.level - 1) * h).round()  # range [-k, k]
        powers = torch.pow(self.level, torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1]).int()
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("There is no official up project component provided")


class FSQVectorQuantization(torch.nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class FSMNMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()

        self.n_head = n_head
        self.query = torch.nn.Linear(n_state, n_state)
        self.key = torch.nn.Linear(n_state, n_state, bias=False)
        self.value = torch.nn.Linear(n_state, n_state)
        self.out = torch.nn.Linear(n_state, n_state)

        self.fsmn_block = torch.nn.Conv1d(
            n_state, n_state, kernel_size, stride=1, padding=0, groups=n_state, bias=False
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)

        self.use_sdpa = use_sdpa

    def forward_fsmn(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:  # time2 > 0
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), fsm_memory
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
            output = output.transpose(1, 2).contiguous().view(q.size(0), -1, D)  # (batch, time1, d_model)
            return output, None, fsm_memory

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size, use_sdpa=use_sdpa)
        self.attn_ln = torch.nn.LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_mlp), torch.nn.GELU(), torch.nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = torch.nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(torch.nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = torch.nn.Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = torch.nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.freqs_cis = precompute_freqs_cis(64, 1024 * 2)
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = mask_to_bias(mask, x.dtype)

        tmp = torch.view_as_real(freqs_cis)
        cos, sin = tmp[:, :, 0], tmp[:, :, 1]

        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[: x.size(1)])

        return x, x_len


class S3TokenizerV2(torch.nn.Module):
    """S3 tokenizer v2 implementation (inference-only).
    Args:
        config (ModelConfig): Config
    """

    def __init__(self, name: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.name = name  # Store model name for token_rate determination
        if "v1" not in name:
            assert "v2" in name
            # TODO(Mddct): make it configureable
            config.n_codebook_size = 3**8
        self.config = config
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

        self.init_from_onnx(name)

    def forward(self, mel: torch.Tensor, mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: torch.Tensor, mel_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize mel spectrogram to tokens, with automatic long audio handling.

        Args:
            mel: mel spectrogram tensor, shape (batch_size, n_mels, T)
            mel_len: mel length tensor, shape (batch_size,)

        Returns:
            code: quantized tokens, shape (batch_size, T')
            code_len: token length, shape (batch_size,)
        """
        # Check if any audio in the batch exceeds 30 seconds
        # Assuming 16kHz sample rate and hop_length=160, 30s = 30*16000/160 = 3000 frames
        max_frames = 3000

        # Check which samples are long audio
        long_audio_mask = mel_len > max_frames

        if long_audio_mask.any():
            # Has long audio - need special processing
            # return self._quantize_mixed_batch(mel, mel_len, long_audio_mask,
            #                                   max_frames)
            raise NotImplementedError("Long audio handling is not implemented yet.")
        else:
            # All short audio - use original method
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return code, code_len

    @property
    def device(self):
        return next(self.parameters()).device

    def init_from_onnx(self, onnx_path: str):
        ckpt = onnx2torch(onnx_path, None, False)
        self.load_state_dict(ckpt, strict=True)

    def init_from_pt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        self.load_state_dict(ckpt, strict=True)

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
