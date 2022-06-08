"""Contains steps to map dataset labels."""

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def get_mapping(labels: Sequence[Any]) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Creates an index-based mapping of the dataset labels.

    Unique elements are extracted from the labels. To ensure consistency in the
        output, the mapping array is sorted. A dictionary is created, storing
        the unique elements as keys and corresponding indices in the sorted
        array as values.

    :param labels: array of dataset labels
    :return:
        tuple containing array of sorted unique elements from labels and
        matching dictionary of key unique elements and value indices
    """
    mapping_array = np.unique(labels)
    assert is_sorted(mapping_array)
    mapping_dictionary = to_index_dictionary(mapping_array)
    return mapping_array, mapping_dictionary


def to_index_dictionary(array: Sequence[Any]) -> Dict[Any, int]:
    """Creates a dictionary of sequence elements and corresponding indices.

    :param array: sequence to be represented as an index-based dictionary
    :return: dictionary of key elements and value indices
    """
    labels = dict()

    for i, label in enumerate(array):
        labels[label] = i
    return labels


def is_sorted(array: Sequence[Any]) -> bool:
    """Determines if array is sorted in non-descending order.

    :param array: sequence to be checked
    :return: truth value of sortedness
    """
    for i in range(len(array) - 1):
        if array[i] > array[i + 1]:
            return False
    return True
