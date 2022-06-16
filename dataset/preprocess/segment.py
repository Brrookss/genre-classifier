"""Contains functionality for segmenting inputs and labels."""

from typing import Any, Sequence, Tuple

import numpy as np


def segment(
    input: Sequence[Any],
    num_segments: int,
    label: Any = None
) -> Tuple[Sequence[Any], Sequence[Any]]:
    """Splits input along its first dimension.

    A corresponding sequence of labels is also returned. Sequence elements
        are None if a label is not provided, otherwise the elements will be
        the label.

    To ensure equal segment sizes, the final segment may be omitted if the
        number of input features - as given in the first dimension - is not
        evenly divisible by the expected number of segments. No exception is
        raised under these conditions.

    :param input: input sequence
    :param num_segments: expected number of segments to split data into
    :param label: label corresponding to input, optional
    :return: tuple of (sequence of input segments, sequence of labels)
    """
    input = np.array(input)
    features = 0
    num_features = remainder = input.shape[features]
    segment_size = num_features // num_segments

    input_segments = []
    label_segments = []
    offset = 0

    while remainder >= segment_size:
        segment = input[offset: offset + segment_size]

        input_segments.append(segment)
        label_segments.append(label)

        remainder -= segment_size
        offset += segment_size
    return input_segments, label_segments
