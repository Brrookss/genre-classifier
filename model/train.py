"""Trains a neural network model on a dataset."""

import argparse
import sys
from typing import Any, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append("..")  # Required for following module imports

from constants import BATCH_SIZE  # noqa: E402
from constants import EPOCHS  # noqa: E402
from constants import TRAIN_SET_FRACTION  # noqa: E402


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_fp",
                        help="path to .npz dataset")
    parser.add_argument("model_fp",
                        help="path to preconfigured model")
    parser.add_argument("-e", "--export",
                        help="path to location model is to be saved")

    args = parser.parse_args()
    return args


def divide_dataset(
    inputs: Sequence[Any],
    labels: Sequence[Any],
    train_set_fraction: float
) -> Tuple[Sequence[Any], Sequence[Any],
           Sequence[Any], Sequence[Any],
           Sequence[Any], Sequence[Any]]:
    """Divides dataset into train, validate, and test subsets.

    Dataset is shuffled during the creation of the subsets. The fraction
    of the dataset allocated to the train subset must be provided; the
    remainder is evenly split between the validate and test subsets.

    :param inputs: sequence of dataset inputs
    :param labels: sequence of dataset labels
    :param train_set_fraction: proportion of dataset allocated to train subset
    :return:
        tuple of train, validate, and test inputs, followed by train, validate,
        and test labels
    """
    train_inputs, remainder_inputs, train_labels, remainder_labels = (
        train_test_split(inputs, labels,
                         train_size=train_set_fraction, shuffle=True)
    )
    validate_inputs, test_inputs, validate_labels, test_labels = (
        train_test_split(remainder_inputs, remainder_labels,
                         test_size=0.5, shuffle=True)
    )
    return (train_inputs, validate_inputs, test_inputs,
            train_labels, validate_labels, test_labels)


def main():
    """Trains a neural network model on a dataset."""
    args = get_arguments()

    with np.load(args.dataset_fp) as data:
        inputs = data["inputs"]
        labels = data["labels"]

    (train_inputs, validate_inputs, test_inputs,
     train_labels, validate_labels, test_labels) = (
         divide_dataset(inputs, labels, TRAIN_SET_FRACTION)
     )

    model = tf.keras.models.load_model(args.model_fp)
    model.fit(train_inputs, train_labels,
              batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=(validate_inputs, validate_labels))
    model.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)

    model.save(args.export or args.model_fp)


if __name__ == "__main__":
    main()
