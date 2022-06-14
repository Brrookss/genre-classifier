"""Contains functionality for input feature engineering."""

import librosa
import numpy as np


def audio_to_image_representation(input: np.ndarray) -> np.ndarray:
    """Converts input in the form of audio data to image representation.

    Input is expected to be representative of audio data in the
        form: (features, time).

    Conversion swaps features and time dimensions and adds an outermost
        dimension, resulting in a representation analogous to an image:
        (width, height, channels). The channels dimension only contains
        1 value, resulting in the image being grayscale specifically.

    :param input: input representative of audio
    :return: input representative of an image
    """
    audio_features = 0
    time = 1
    channels = 2

    width_height = np.swapaxes(input, audio_features, time)
    width_height_channels = np.expand_dims(width_height, channels)
    return width_height_channels


def waveform_to_normalized_mel_spectrogram(
    input: np.ndarray,
    sampling_rate: int
) -> np.ndarray:
    """Converts waveform to a normalized mel spectrogram representation.

    :param input: waveform of input
    :param sampling_rate: sampling rate of input
    :return: mel spectrogram representation of waveform normalized to [-1, 1]
    """
    waveform_normalized = librosa.util.utils.normalize(input)
    short_time_fourier_transform = np.abs(
        librosa.core.stft(waveform_normalized, n_fft=2048, hop_length=512)
    )
    mel = librosa.feature.melspectrogram(sr=sampling_rate,
                                         S=short_time_fourier_transform**2)
    mel_log = np.log(mel + 1e-9)
    mel_normalized = librosa.util.normalize(mel_log)
    return mel_normalized
