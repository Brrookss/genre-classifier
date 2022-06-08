"""Contains functionality for splitting inputs and labels."""

import sys
from typing import Any, Sequence, Tuple

import numpy as np


def segment(
    input: Sequence[Any],
    num_segments: int,
    label: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits input and if provided, the corresponding label.

    :param input: input sequence
    :param num_segments: expected number of segments to split data into
    :param label: label corresponding to input, optional
    :return: tuple of (array of input segments, array of labels)
    """
    try:
        input_segments = np.split(input, num_segments)
    except ValueError as error:
        print(
            f"{error}: segment of differing size will be removed",
            file=sys.stderr
        )

        input_segments = np.array_split(input, num_segments)
        num_segments -= 1
        input_segments = input_segments[:num_segments]

    label_segments = [label for _ in range(num_segments)]
    assert len(input_segments) == len(label_segments) == num_segments

    input_segments = np.array(input_segments)
    label_segments = np.array(label_segments)
    return input_segments, label_segments
