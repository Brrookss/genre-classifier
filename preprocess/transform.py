"""Applies feature engineering to dataset inputs.

usage: transform.py [-h] [-s SEGMENT] [-o OUTFILE] dataset_fp
"""

import argparse

import librosa
import numpy as np
import sys

sys.path.append("..")  # Required for following module imports

from constants import SAMPLES  # noqa: E402
from dataset.constants import TRACK_SAMPLING_RATE_HZ  # noqa: E402
from segment import segment  # noqa: E402


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_fp",
                        help="path to .npz dataset")
    parser.add_argument("-s", "--segment", type=int,
                        help="number of segments to divide tracks into")
    parser.add_argument("-o", "--outfile",
                        help="path to new location dataset is to be saved")

    args = parser.parse_args()
    return args


def transform(inputs: np.ndarray) -> np.ndarray:
    """Applies feature engineering to inputs.

    :param inputs: array of dataset inputs
    :return: array of inputs with feature engineering applied
    """
    num_inputs = inputs.shape[SAMPLES]
    inputs_transformed = []

    for i in range(num_inputs):
        transformed = waveform_to_normalized_mel_spectrogram(inputs[i])
        inputs_transformed.append(transformed)

    inputs_transformed = np.array(inputs_transformed)
    return inputs_transformed


def waveform_to_normalized_mel_spectrogram(input: np.ndarray) -> np.ndarray:
    """Converts waveform to a normalized mel spectrogram representation.

    :param input: waveform of a single input
    :return: mel spectrogram representation of waveform normalized to [-1, 1]
    """
    waveform_normalized = librosa.util.utils.normalize(input)
    short_time_fourier_transform = librosa.core.stft(waveform_normalized,
                                                     n_fft=2048,
                                                     hop_length=512)
    mel = librosa.feature.melspectrogram(sr=TRACK_SAMPLING_RATE_HZ,
                                         S=short_time_fourier_transform**2)
    mel_log = np.log(mel + 1e-9)
    mel_normalized = librosa.util.normalize(mel_log)
    return mel_normalized


def audio_to_image_representation(inputs: np.ndarray) -> np.ndarray:
    """Converts inputs in the form of audio data to image representation.

    Inputs are expected to be representative of audio data in the
    form: (samples, features, time). In this context, samples indicates each
    input in the dataset; samples does not relate to the term as it is used in
    the context of audio engineering.

    Conversion swaps features and time dimensions and adds an outermost
    dimension, resulting in a representation analogous to an image:
    (samples, width, height, channels). The channels dimension only contains
    1 value, resulting in the image being grayscale specifically.

    :param inputs: array of dataset inputs representative of audio
    :return: array of inputs representative of images
    """
    time = 1
    audio_features = 2
    channels = 3

    samples_width_height = np.swapaxes(inputs, time, audio_features)
    samples_width_height_channels = np.expand_dims(samples_width_height,
                                                   channels)
    return samples_width_height_channels


def main():
    """Applies feature engineering to dataset inputs."""
    args = get_arguments()

    with np.load(args.dataset_fp) as data:
        inputs = data["inputs"]
        labels = data["labels"]

    if args.segment:
        inputs, labels = segment(inputs, labels, args.segment)

    inputs = transform(inputs)
    inputs = audio_to_image_representation(inputs)
    np.savez(args.outfile or args.dataset_fp, inputs=inputs, labels=labels)


if __name__ == "__main__":
    main()
