import torch
import transformers.configuration_utils
from torch import nn
from transformers import AutoModel

# The MOSS-Audio-Tokenizer remote code uses the old spelling "PreTrainedConfig"
# which was renamed to "PretrainedConfig" in newer transformers versions.
if not hasattr(transformers.configuration_utils, "PreTrainedConfig"):
    transformers.configuration_utils.PreTrainedConfig = transformers.configuration_utils.PretrainedConfig


class MossAudioTokenizerDecoder(nn.Module):
    """Wrapper for OpenMOSS-Team/MOSS-Audio-Tokenizer decode functionality."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            trust_remote_code=True,
        )
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

        self.sample_rate = 24000
        self.num_codebooks = 32
        self.downsample_rate = 1920  # 24000 / 12.5 Hz

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform.

        Args:
            codes: (batch, 32, time) audio codes

        Returns:
            (batch, 1, audio_length) waveform tensor
        """
        batch_size, n_codebooks, time = codes.shape
        device = codes.device

        # Transpose to (NQ, B, T) for the tokenizer API
        codes = codes.transpose(0, 1)  # (32, batch, time)

        # Create padding mask (all valid)
        padding_mask = torch.ones(batch_size, time, device=device, dtype=torch.bool)

        dec = self.model.decode(
            codes,
            padding_mask=padding_mask,
            return_dict=True,
        )
        audio = dec.audio  # (batch, 1, audio_length)
        return audio
