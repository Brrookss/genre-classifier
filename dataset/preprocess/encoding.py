"""Contains functionality for label encoding."""

from typing import Any, Dict, Sequence

import numpy as np


def integer_encode(
    array: Sequence[Any], mapping_dictionary: Dict[Any, int]
) -> np.ndarray:
    """Converts sequence elements to an array of corresponding integers.

    :param array: sequence to be categorically encoded
    :param mapping_dictionary: mapping of array element keys and integer values
    :return: array of integer representations of the provided elements
    """
    encoded = np.zeros(len(array), dtype=np.int32)

    for i, element in enumerate(array):
        encoded[i] = mapping_dictionary[element]
    return encoded
