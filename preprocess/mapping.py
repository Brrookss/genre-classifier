"""Maps dataset labels."""

from typing import Any, Sequence

import numpy as np


def get_mapping(labels: Sequence[int]) -> np.ndarray:
    """Creates an index-based mapping of the dataset labels.

    Unique values are extracted from the labels. To ensure consistency in the
    output, the mapping array is sorted.

    :param labels: array of dataset labels
    :return: array of sorted unique values from labels
    """
    mapping = np.unique(labels)

    assert is_sorted(mapping)
    return mapping


def is_sorted(array: Sequence[Any]) -> bool:
    """Determines if array is sorted in non-descending order.

    :param array: array to be checked
    :return: truth value of sortedness
    """
    i = 0

    while i + 1 < len(array):
        if array[i] > array[i + 1]:
            return False
        i += 1
    return True
