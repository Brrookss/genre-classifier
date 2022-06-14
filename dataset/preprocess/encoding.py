"""Contains functionality for label encoding."""

from typing import Any, Dict, Sequence

import numpy as np


def integer_encode(labels: Sequence[Any]) -> Dict[Any, int]:
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
