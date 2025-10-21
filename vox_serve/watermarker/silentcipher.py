import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from pydub import AudioSegment
from scipy import stats as st
from torch import nn

from ..utils import get_logger

logger = get_logger(__name__)


class Layer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.conv(x) * torch.sigmoid(self.gate(x)))


class Encoder(nn.Module):
    def __init__(self, out_dim=32, n_layers=3, message_dim=0, message_band_size=None, n_fft=None):
        super(Encoder, self).__init__()
        assert message_band_size is not None
        assert n_fft is not None
        self.message_band_size = message_band_size
        main = [Layer(dim_in=1, dim_out=32, kernel_size=3, stride=1, padding=1)]

        for _ in range(n_layers - 2):
            main.append(Layer(dim_in=32, dim_out=32, kernel_size=3, stride=1, padding=1))
        main.append(Layer(dim_in=32, dim_out=out_dim, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*main)
        self.linear = nn.Linear(message_dim, message_band_size)
        self.n_fft = n_fft

    def forward(self, x):
        h = self.main(x)
        return h

    def transform_message(self, msg):
        output = self.linear(msg.transpose(2, 3)).transpose(2, 3)
        if self.message_band_size != self.n_fft // 2 + 1:
            output = torch.nn.functional.pad(output, (0, 0, 0, self.n_fft // 2 + 1 - self.message_band_size))
        return output


class CarrierDecoder(nn.Module):
    def __init__(self, config, conv_dim, n_layers=4, message_band_size=1024):
        super(CarrierDecoder, self).__init__()
        self.config = config
        self.message_band_size = message_band_size
        layers = [Layer(dim_in=conv_dim, dim_out=96, kernel_size=3, stride=1, padding=1)]

        for _ in range(n_layers - 2):
            layers.append(Layer(dim_in=96, dim_out=96, kernel_size=3, stride=1, padding=1))

        layers.append(Layer(dim_in=96, dim_out=1, kernel_size=1, stride=1, padding=0))

        self.main = nn.Sequential(*layers)

    def forward(self, x, message_sdr):
        h = self.main(x)

        if self.config.ensure_negative_message:
            h = torch.abs(h)

        h[:, :, self.message_band_size :, :] = 0

        if not self.config.no_normalization:
            h = h / torch.mean(h**2, dim=2, keepdim=True) ** 0.5 / (10 ** (message_sdr / 20))

        return h


class MsgDecoder(nn.Module):
    def __init__(self, message_dim=0, message_band_size=None, channel_dim=128, num_layers=10):
        super(MsgDecoder, self).__init__()
        assert message_band_size is not None
        self.message_band_size = message_band_size

        main = [nn.Dropout(0), Layer(dim_in=1, dim_out=channel_dim, kernel_size=3, stride=1, padding=1)]
        for _ in range(num_layers - 2):
            main += [
                nn.Dropout(0),
                Layer(dim_in=channel_dim, dim_out=channel_dim, kernel_size=3, stride=1, padding=1),
            ]
        main += [nn.Dropout(0), Layer(dim_in=channel_dim, dim_out=message_dim, kernel_size=3, stride=1, padding=1)]
        self.main = nn.Sequential(*main)
        self.linear = nn.Linear(self.message_band_size, 1)

    def forward(self, x):
        h = self.main(x[:, :, : self.message_band_size])
        h = self.linear(h.transpose(2, 3)).squeeze(3).unsqueeze(1)
        return h


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class STFT(torch.nn.Module, metaclass=Singleton):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_len = hop_length
        self.win_len = filter_length
        self.window = torch.hann_window(self.win_len).to("cuda")
        self.num_samples = -1

    def transform(self, x):
        x = torch.nn.functional.pad(x, (0, self.win_len - x.shape[1] % self.win_len))
        fft = torch.stft(
            x, self.filter_length, self.hop_len, self.win_len, window=self.window.to(x.device), return_complex=True
        )

        real_part, imag_part = fft.real, fft.imag

        squared = real_part**2 + imag_part**2
        additive_epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
        magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(additive_epsilon)

        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data)).float()
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = magnitude * torch.cos(phase) + 1j * magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            recombine_magnitude_phase,
            self.filter_length,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window.to(magnitude.device),
        ).unsqueeze(1)  # , length=self.num_samples
        padding = self.win_len - (self.num_samples % self.win_len)
        inverse_transform = inverse_transform[:, :, :-padding]
        return inverse_transform


class Model:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device

        self.n_messages = config.n_messages
        self.model_type = config.model_type
        self.message_dim = config.message_dim
        self.message_len = config.message_len

        # model dimensions
        self.enc_conv_dim = 16
        self.enc_num_repeat = 3
        self.dec_c_num_repeat = self.enc_num_repeat
        self.dec_m_conv_dim = 1
        self.dec_m_num_repeat = 8
        self.encoder_out_dim = 32
        self.dec_c_conv_dim = 32 * 3

        self.enc_c = Encoder(
            n_layers=self.config.enc_n_layers,
            message_dim=self.message_dim,
            out_dim=self.encoder_out_dim,
            message_band_size=self.config.message_band_size,
            n_fft=self.config.N_FFT,
        )

        self.dec_c = CarrierDecoder(
            config=self.config,
            conv_dim=self.dec_c_conv_dim,
            n_layers=self.config.dec_c_n_layers,
            message_band_size=self.config.message_band_size,
        )

        self.dec_m = [
            MsgDecoder(message_dim=self.message_dim, message_band_size=self.config.message_band_size)
            for _ in range(self.n_messages)
        ]
        # ------ make parallel ------
        self.enc_c = self.enc_c.to(self.device)
        self.dec_c = self.dec_c.to(self.device)
        self.dec_m = [m.to(self.device) for m in self.dec_m]

        self.average_energy_VCTK = torch.tensor(0.002837200844477648, device=self.device)
        self.stft = STFT(self.config.N_FFT, self.config.HOP_LENGTH)
        self.stft.to(self.device)
        self.load_models(config.load_ckpt)
        self.sr = self.config.SR

    def letters_encoding(self, patch_len, message_lst):
        """
        Encodes a list of messages into a compact representation and a padded representation.

        Args:
            patch_len (int): The length of the patch.
            message_lst (list): A list of messages to be encoded.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - message: A padded representation of the messages, where each message is repeated to match the patch
                    length.
                - message_compact: A compact representation of the messages, where each message is encoded as a one-hot
                    vector.

        Raises:
            AssertionError: If the length of any message in message_lst is not equal to self.config.message_len - 1.
        """

        message = []
        message_compact = []
        for i in range(self.n_messages):
            assert len(message_lst[i]) == self.config.message_len - 1
            index = np.concatenate((np.array(message_lst[i]) + 1, [0]))
            one_hot = np.identity(self.message_dim)[index]
            message_compact.append(one_hot)
            if patch_len % self.message_len == 0:
                message.append(np.tile(one_hot.T, (1, patch_len // self.message_len)))
            else:
                _ = np.tile(one_hot.T, (1, patch_len // self.message_len))
                _ = np.concatenate([_, one_hot.T[:, 0 : patch_len % self.message_len]], axis=1)
                message.append(_)
        message = np.stack(message)
        message_compact = np.stack(message_compact)
        # message = np.pad(message, ((0, 0), (0, 129 - self.message_dim), (0, 0)), 'constant')
        return message, message_compact

    def get_best_ps(self, y_one_sec):
        """
        Calculates the best phase shift value for watermark decoding.

        Args:
            y_one_sec (numpy.ndarray): Input audio signal.

        Returns:
            int: The best phase shift value.

        """

        def check_accuracy(pred_values):
            accuracy = 0
            for i in range(pred_values.shape[1]):
                unique, counts = np.unique(pred_values[:, i], return_counts=True)
                accuracy += np.max(counts) / pred_values.shape[0]

            return accuracy / pred_values.shape[1]

        y = torch.FloatTensor(y_one_sec).unsqueeze(0).unsqueeze(0).to(self.device)
        max_accuracy = 0
        final_phase_shift = 0

        for ps in range(0, self.config.HOP_LENGTH, 10):
            carrier, _ = self.stft.transform(y[0:1, 0:1, ps:].squeeze(1))
            carrier = carrier[:, None]

            for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                msg_reconst = self.dec_m[i](carrier)
                pred_values = torch.argmax(msg_reconst[0, 0], dim=0).data.cpu().numpy()
                pred_values = pred_values[
                    0 : int(msg_reconst.shape[3] / self.config.message_len) * self.config.message_len
                ]
                pred_values = pred_values.reshape([-1, self.config.message_len])
                cur_acc = check_accuracy(pred_values)
                if cur_acc > max_accuracy:
                    max_accuracy = cur_acc
                    final_phase_shift = ps

        return final_phase_shift

    def get_confidence(self, pred_values, message):
        """
        Calculates the confidence of the predicted values based on the provided message.

        Parameters:
        pred_values (numpy.ndarray): The predicted values.
        message (str): The message used for prediction.

        Returns:
        float: The confidence score.

        Raises:
        AssertionError: If the length of the message is not equal to the number of columns in pred_values.

        """
        assert len(message) == pred_values.shape[1], f"{len(message)} | {pred_values.shape}"
        return np.mean((pred_values == message[None]).astype(np.float32)).item()

    def sdr(self, orig, recon):
        """
        Calculate the Signal-to-Distortion Ratio (SDR) between the original and reconstructed signals.

        Parameters:
        orig (numpy.ndarray): The original signal.
        recon (numpy.ndarray): The reconstructed signal.

        Returns:
        float: The Signal-to-Distortion Ratio (SDR) value.

        """

        rms1 = (np.mean(orig**2)) ** 0.5
        rms2 = (np.mean((orig - recon) ** 2)) ** 0.5
        sdr = 20 * np.log10(rms1 / rms2)
        return sdr

    def load_audio(self, path):
        """
        Load an audio file from the given path and return the audio array and sample rate.

        Args:
            path (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the audio array and sample rate.

        """
        audio = AudioSegment.from_file(path)
        audio_array, sr = (
            (
                np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels))
                / (1 << (8 * audio.sample_width - 1))
            ),
            audio.frame_rate,
        )
        if audio_array.shape[1] == 1:
            audio_array = audio_array[:, 0]

        return audio_array, sr

    def encode(self, in_path, out_path, message_list, message_sdr=None, calc_sdr=True, disable_checks=False):
        """
        Encodes a message into an audio file.

        Parameters:
        - in_path (str): The path to the input audio file.
        - out_path (str): The path to save the output audio file.
        - message_list (list): A list of messages to be encoded into the audio file.
        - message_sdr (float, optional): The Signal-to-Distortion Ratio (SDR) of the message. Defaults to None.
        - calc_sdr (bool, optional): Whether to calculate the SDR of the encoded audio. Defaults to True.
        - disable_checks (bool, optional): Whether to disable input checks. Defaults to False.

        Returns:
        - dict: A dictionary containing the status of the encoding process, the SDR value(s), the time taken for
            encoding, and the time taken per second of audio.

        """
        y, orig_sr = self.load_audio(in_path)
        start = time.time()
        encoded_y, sdr = self.encode_wav(
            y,
            orig_sr,
            message_list=message_list,
            message_sdr=message_sdr,
            calc_sdr=calc_sdr,
            disable_checks=disable_checks,
        )
        time_taken = time.time() - start
        sf.write(out_path, encoded_y, orig_sr)

        if isinstance(sdr, list):
            return {
                "status": True,
                "sdr": [f"{sdr_i:.2f}" for sdr_i in sdr],
                "time_taken": time_taken,
                "time_taken_per_second": time_taken / (y.shape[0] / orig_sr),
            }
        else:
            return {
                "status": True,
                "sdr": f"{sdr:.2f}",
                "time_taken": time_taken,
                "time_taken_per_second": time_taken / (y.shape[0] / orig_sr),
            }

    def decode(self, path, phase_shift_decoding):
        """
        Decode the audio file at the given path using phase shift decoding.

        Parameters:
        path (str): The path to the audio file.
        phase_shift_decoding (bool): Flag indicating whether to use phase shift decoding.

        Returns:
        dictionary: A dictionary containing the decoded message status and value
        """

        y, orig_sr = self.load_audio(path)

        return self.decode_wav(y, orig_sr, phase_shift_decoding)

    def encode_wav(self, y_multi_channel, orig_sr, message_list, message_sdr=36, calc_sdr=False, disable_checks=False):
        """
        Encodes a multi-channel audio waveform with a given message.

        Args:
            y_multi_channel (torch.Tensor): The multi-channel audio waveform to be encoded.
            orig_sr (int): The original sampling rate of the audio waveform.
            message_list (list): The list of messages to be encoded. Each message may correspond to a channel in the
                audio waveform.
            message_sdr (float, optional): The signal-to-distortion ratio (SDR) of the message. If not provided, the
                default SDR from the configuration is used.
            calc_sdr (bool, optional): Flag indicating whether to calculate the SDR of the encoded waveform. Defaults
                to True.
            disable_checks (bool, optional): Flag indicating whether to disable input audio checks. Defaults to False.

        Returns:
            tuple: A tuple containing the encoded multi-channel audio waveform and the SDR (if calculated).

        Raises:
            AssertionError: If the number of messages does not match the number of channels in the input audio
                waveform.
        """

        # Convert input to torch tensor if not already
        if not isinstance(y_multi_channel, torch.Tensor):
            y_multi_channel = torch.tensor(y_multi_channel, dtype=torch.float32)

        single_channel = False
        if len(y_multi_channel.shape) == 1:
            single_channel = True
            y_multi_channel = y_multi_channel.unsqueeze(1)

        if message_sdr is None:
            message_sdr = self.config.message_sdr
            logger.info(f"Using the default SDR of {self.config.message_sdr} dB")

        if isinstance(message_list[0], int):
            message_list = [message_list] * y_multi_channel.shape[1]

        y_watermarked_multi_channel = []
        sdrs = []

        assert len(message_list) == y_multi_channel.shape[1], (
            f"Mismatch: {len(message_list)} messages vs {y_multi_channel.shape[1]} channels"
        )

        for channel_i in range(y_multi_channel.shape[1]):
            y = y_multi_channel[:, channel_i]
            message = message_list[channel_i]

            with torch.no_grad():
                orig_y = y.clone()
                if orig_sr != self.sr:
                    if orig_sr > self.sr:
                        logger.warning(
                            f"Reducing the sampling rate of the original audio from {orig_sr} -> {self.sr}. "
                            f"High frequency components may be lost!"
                        )
                    y = torchaudio.functional.resample(y.view(1, -1), orig_freq=orig_sr, new_freq=self.sr).squeeze()

                original_power = torch.mean(y**2)

                if not disable_checks:
                    if original_power == 0:
                        logger.warning(
                            "The input audio has a power of 0. This means the audio is likely just silence. "
                            "Skipping encoding."
                        )
                        return orig_y, 0

                y = y * torch.sqrt(self.average_energy_VCTK / original_power)
                y = y.unsqueeze(0).unsqueeze(0).to(self.device)
                carrier, carrier_phase = self.stft.transform(y.squeeze(1))
                carrier = carrier[:, None]
                carrier_phase = carrier_phase[:, None]

                def binary_encode(mes):
                    binary_message = "".join(["{0:08b}".format(mes_i) for mes_i in mes])
                    four_bit_msg = []
                    for i in range(len(binary_message) // 2):
                        four_bit_msg.append(int(binary_message[i * 2 : i * 2 + 2], 2))
                    return four_bit_msg

                binary_encoded_message = binary_encode(message)

                msgs, msgs_compact = self.letters_encoding(carrier.shape[3], [binary_encoded_message])
                msg_enc = torch.tensor(msgs, device=self.device, dtype=torch.float32).unsqueeze(0)

                carrier_enc = self.enc_c(carrier)  # encode the carrier
                msg_enc = self.enc_c.transform_message(msg_enc)

                merged_enc = torch.cat((carrier_enc, carrier.repeat(1, 32, 1, 1), msg_enc.repeat(1, 32, 1, 1)), dim=1)

                message_info = self.dec_c(merged_enc, message_sdr)
                if self.config.frame_level_normalization:
                    message_info = message_info * (torch.mean(carrier**2, dim=2, keepdim=True) ** 0.5)  # *time_weighing
                elif self.config.utterance_level_normalization:
                    message_info = message_info * (
                        torch.mean(carrier**2, dim=(2, 3), keepdim=True) ** 0.5
                    )  # *time_weighing

                if self.config.ensure_negative_message:
                    message_info = -message_info
                    carrier_reconst = torch.nn.functional.relu(
                        message_info + carrier
                    )  # decode carrier, output in stft domain
                elif self.config.ensure_constrained_message:
                    message_info[message_info > carrier] = carrier[message_info > carrier]
                    message_info[-message_info > carrier] = -carrier[-message_info > carrier]
                    carrier_reconst = message_info + carrier  # decode carrier, output in stft domain
                    assert torch.all(carrier_reconst >= 0), "negative values found in carrier_reconst"
                else:
                    carrier_reconst = torch.abs(message_info + carrier)  # decode carrier, output in stft domain

                self.stft.num_samples = y.shape[2]

                y = self.stft.inverse(carrier_reconst.squeeze(1), carrier_phase.squeeze(1))[0, 0]
                y = y * torch.sqrt(
                    original_power / self.average_energy_VCTK
                )  # Noise has a power of 5% power of VCTK samples

                if orig_sr != self.sr:
                    y = torchaudio.functional.resample(y.view(1, -1), orig_freq=self.sr, new_freq=orig_sr).squeeze()

                    # Sometimes resampling leads to off-by-one sequence length errors
                    seq_len = y_multi_channel.shape[0]
                    y = y[:seq_len]

                if calc_sdr:
                    sdr = self.sdr(orig_y, y)
                else:
                    sdr = 0

            y_watermarked_multi_channel.append(y.unsqueeze(1))
            sdrs.append(sdr)

        y_watermarked_multi_channel = torch.cat(y_watermarked_multi_channel, dim=1)

        if single_channel:
            y_watermarked_multi_channel = y_watermarked_multi_channel[:, 0]
            sdrs = sdrs[0]

        return y_watermarked_multi_channel

    def decode_wav(self, y_multi_channel, orig_sr, phase_shift_decoding):
        """
        Decode the given audio waveform to extract hidden messages.

        Args:
            y_multi_channel (torch.Tensor): The multi-channel audio waveform.
            orig_sr (int): The original sample rate of the audio waveform.
            phase_shift_decoding (str): Flag indicating whether to perform phase shift decoding.

        Returns:
            dict or list: A list of dictionary containing the decoded messages, confidences, and status for each
                channel if the input is multi-channel.
                Otherwise, a dictionary containing the decoded messages, confidences, and status for a single channel.

        Raises:
            Exception: If the decoding process fails.

        """

        # Convert input to torch tensor if not already
        if not isinstance(y_multi_channel, torch.Tensor):
            y_multi_channel = torch.tensor(y_multi_channel, dtype=torch.float32)

        single_channel = False
        if len(y_multi_channel.shape) == 1:
            single_channel = True
            y_multi_channel = y_multi_channel.unsqueeze(1)

        results = []

        for channel_i in range(y_multi_channel.shape[1]):
            y = y_multi_channel[:, channel_i]
            try:
                with torch.no_grad():
                    if orig_sr != self.sr:
                        y = torchaudio.functional.resample(y.view(1, -1), orig_freq=orig_sr, new_freq=self.sr).squeeze()

                    original_power = torch.mean(y**2)
                    y = y * torch.sqrt(
                        self.average_energy_VCTK / original_power
                    )  # Noise has a power of 5% power of VCTK samples

                    if phase_shift_decoding and phase_shift_decoding != "false":
                        ps = self.get_best_ps(y)
                    else:
                        ps = 0

                    y = y[ps:].unsqueeze(0).unsqueeze(0).to(self.device)
                    carrier, _ = self.stft.transform(y.squeeze(1))
                    carrier = carrier[:, None]

                    msg_reconst_list = []
                    confidence = []

                    for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                        msg_reconst = self.dec_m[i](carrier)
                        pred_values = torch.argmax(msg_reconst[0, 0], dim=0).data.cpu().numpy()
                        pred_values = pred_values[
                            0 : int(msg_reconst.shape[3] / self.config.message_len) * self.config.message_len
                        ]
                        pred_values = pred_values.reshape([-1, self.config.message_len])

                        ord_values = st.mode(pred_values, keepdims=False).mode
                        end_char = np.min(np.nonzero(ord_values == 0)[0])
                        confidence.append(self.get_confidence(pred_values, ord_values))

                        if end_char == self.config.message_len:
                            ord_values = ord_values[: self.config.message_len - 1]
                        else:
                            ord_values = np.concatenate([ord_values[end_char + 1 :], ord_values[:end_char]], axis=0)

                        # pred_values = ''.join([chr(v + 64) for v in ord_values])
                        msg_reconst_list.append((ord_values - 1).tolist())

                    def convert_to_8_bit_segments(msg_list):
                        segment_message_list = []
                        for msg_list_i in msg_list:
                            binary_format = "".join([f"{mes_i:02b}" for mes_i in msg_list_i])
                            eight_bit_segments = [
                                int(binary_format[i * 8 : i * 8 + 8], 2) for i in range(len(binary_format) // 8)
                            ]
                            segment_message_list.append(eight_bit_segments)
                        return segment_message_list

                    msg_reconst_list = convert_to_8_bit_segments(msg_reconst_list)

                results.append({"messages": msg_reconst_list, "confidences": confidence, "status": True})
            except Exception as e:
                logger.error(f"Decoding failed for channel {channel_i}: {e}")
                results.append({"messages": [], "confidences": [], "error": "Could not find message", "status": False})

        if single_channel:
            results = results[0]

        return results

    def convert_dataparallel_to_normal(self, checkpoint):
        return {i[len("module.") :] if i.startswith("module.") else i: checkpoint[i] for i in checkpoint}

    def load_models(self, ckpt_dir):
        self.enc_c.load_state_dict(
            self.convert_dataparallel_to_normal(
                torch.load(os.path.join(ckpt_dir, "enc_c.ckpt"), map_location=self.device)
            )
        )
        self.dec_c.load_state_dict(
            self.convert_dataparallel_to_normal(
                torch.load(os.path.join(ckpt_dir, "dec_c.ckpt"), map_location=self.device)
            )
        )
        for i, m in enumerate(self.dec_m):
            m.load_state_dict(
                self.convert_dataparallel_to_normal(
                    torch.load(os.path.join(ckpt_dir, f"dec_m_{i}.ckpt"), map_location=self.device)
                )
            )


def get_model(
    model_type="44.1k",
    ckpt_path="../Models/44_1_khz/73999_iteration",
    config_path="../Models/44_1_khz/73999_iteration/hparams.yaml",
    device="cpu",
):
    if model_type == "44.1k":
        if not os.path.exists(ckpt_path) or not os.path.exists(config_path):
            logger.info("ckpt path or config path does not exist! Downloading the model from the Hugging Face Hub...")
            from huggingface_hub import snapshot_download

            folder_dir = snapshot_download(repo_id="sony/silentcipher")
            ckpt_path = os.path.join(folder_dir, "44_1_khz/73999_iteration")
            config_path = os.path.join(folder_dir, "44_1_khz/73999_iteration/hparams.yaml")

        config = yaml.safe_load(open(config_path))
        config = argparse.Namespace(**config)
        config.load_ckpt = ckpt_path
        model = Model(config, device)
    elif model_type == "16k":
        if not os.path.exists(ckpt_path) or not os.path.exists(config_path):
            logger.info("ckpt path or config path does not exist! Downloading the model from the Hugging Face Hub...")
            from huggingface_hub import snapshot_download

            folder_dir = snapshot_download(repo_id="sony/silentcipher")
            ckpt_path = os.path.join(folder_dir, "16_khz/97561_iteration")
            config_path = os.path.join(folder_dir, "16_khz/97561_iteration/hparams.yaml")

        config = yaml.safe_load(open(config_path))
        config = argparse.Namespace(**config)
        config.load_ckpt = ckpt_path

        model = Model(config, device)
    else:
        logger.error("Please specify a valid model_type [44.1k, 16k]")

    return model
