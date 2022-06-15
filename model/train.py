"""Trains a neural network model on a dataset.

Example:
    python train.py [-h] [-e EXPORT] model_filepath dataset_filepath
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import constants


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filepath",
                        help="path to preconfigured model")
    parser.add_argument("dataset_filepath",
                        help="path to .npz file of dataset")
    parser.add_argument("-e", "--export",
                        help="path to location model is to be saved")

    args = parser.parse_args()
    return args


def main():
    """Trains a neural network model on a dataset."""
    args = get_arguments()

    with np.load(args.dataset_filepath) as dataset:
        inputs = dataset["inputs"]
        labels = dataset["labels"]

    train_inputs, remainder_inputs, train_labels, remainder_labels = (
        train_test_split(inputs, labels,
                         train_size=constants.TRAIN_SET_FRACTION, shuffle=True)
    )
    validate_inputs, test_inputs, validate_labels, test_labels = (
        train_test_split(remainder_inputs, remainder_labels,
                         test_size=0.5, shuffle=True)
    )
    del inputs

    model = tf.keras.models.load_model(args.model_filepath)
    model.fit(train_inputs, train_labels,
              batch_size=constants.BATCH_SIZE, epochs=constants.EPOCHS,
              validation_data=(validate_inputs, validate_labels))
    del train_inputs
    del validate_inputs

    model.evaluate(test_inputs, test_labels, batch_size=constants.BATCH_SIZE)
    model.save(args.export or args.model_filepath)


if __name__ == "__main__":
    main()
