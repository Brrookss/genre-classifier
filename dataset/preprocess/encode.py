"""Contains functionality for label encoding."""

from typing import Any, Dict, Sequence

import numpy as np


def integer_encode_mapping(labels: Sequence[Any]) -> Dict[Any, int]:
    """Converts sequence of labels to a dictionary of index-based values.

    To ensure consistent encoding, labels are sorted before the creation of the
        mapping dictionary; the values of the mapping dictionary correspond to
        the label indices after sorting occurs.

    :param labels: sequence of labels to be encoded
    :return: dictionary of label keys and index values
    """
    encoded = {}

    unique_labels = np.unique(labels)
    assert is_sorted(unique_labels)

    for index, label in enumerate(unique_labels):
        encoded[label] = index
    return encoded


def is_sorted(seq: Sequence[Any]) -> bool:
    """Determines if sequence is sorted in non-descending order.

    :param seq: sequence to be checked
    :return: truth value associated with sortedness
    """
    for i in range(len(seq) - 1):
        if seq[i] > seq[i + 1]:
            return False
    return True


def one_hot_encode(integer_label: int, num_categories: int) -> np.ndarray:
    """Converts integer label to a one-hot encoded array.

    Datatype of the encoded array is int32.

    :param integer_label: categorical label in the range [0, num_categories)
    :param num_categories: number of categorical targets
    :return: one-hot encoded array
    """
    encoded = np.zeros(num_categories, dtype=np.int32)
    encoded[integer_label] = 1
    return encoded
