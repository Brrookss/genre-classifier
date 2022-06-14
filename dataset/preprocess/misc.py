"""Contains miscellaneous functionality for input preprocessing."""

from typing import Sequence

import librosa
import numpy as np

import constants


def verify_and_fix_length(waveform: Sequence[int]) -> np.ndarray:
    """Checks signal length and adjusts to the expected length, if necessary.

    Expected length is defined as the product of TRACK_SAMPLING_RATE_HZ and
        TRACK_DURATION_SECONDS.

    Tracks containing an incompatible number of samples are zero padded or
        truncated accordingly.

    :param waveform: audio data corresponding to a waveform
    :return: audio data matching the expected length
    """
    expected_num_samples = (constants.TRACK_SAMPLING_RATE_HZ
                            * constants.TRACK_DURATION_SECONDS)

    if len(waveform) != expected_num_samples:
        waveform = librosa.util.fix_length(waveform, size=expected_num_samples)
    return waveform
