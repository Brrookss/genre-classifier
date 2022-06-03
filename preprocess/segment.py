"""Splits dataset inputs and corresponding labels."""

from typing import Tuple

import numpy as np

from constants import FEATURES
from constants import SAMPLES


def segment(
    inputs: np.ndarray,
    labels: np.ndarray,
    num_segments: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits dataset inputs and corresponding labels.

    Number of segments is expected to be at least 1 and less than the
    amount of feature data contained in each individual input to
    prevent undersized segments.

    :param inputs: array of dataset inputs
    :param labels: array of dataset labels
    :param num_segments: expected number of segments per track
    :raises ValueError: number of segments is outside allowed range
    :return: tuple of segmented arrays of inputs and labels
    """
    num_inputs = inputs.shape[SAMPLES]
    num_features = inputs.shape[FEATURES]
    segment_size = num_features // num_segments

    if num_segments < 1 or num_segments > num_features:
        raise ValueError(f"{num_segments} segments is not valid")

    input_segments = []
    label_segments = []

    for i in range(num_inputs):
        input = inputs[i]
        label = labels[i]
        offset = 0

        for _ in range(num_segments):
            segment = input[offset: offset + segment_size]
            offset += segment_size

            input_segments.append(segment)
            label_segments.append(label)

    expected_num_segments = num_inputs * num_segments
    assert len(input_segments) == len(label_segments) == expected_num_segments

    input_segments = np.array(input_segments)
    label_segments = np.array(label_segments)
    return input_segments, label_segments
