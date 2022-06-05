"""Builds and compiles a neural network model.

usage: configure.py [-h] [-s] dataset_filepath export
"""

import argparse
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf

sys.path.append("..")

from model.constants import LEARNING_RATE  # noqa: E402
from preprocess.mapping import get_mapping  # noqa: E402


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_filepath",
                        help="path to .npz file of dataset")
    parser.add_argument("export",
                        help="path to location model is to be saved")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="suppress display of model summary")

    args = parser.parse_args()
    return args


def build(
    input_shape: Tuple[int, int, int],
    num_outputs: int
) -> tf.keras.Model:
    """Stacks layers to build a 2D convolutional neural network model.

    Model is composed of a convolutional block followed by a fully connected
    block. Maxpooling is implemented in the convolutional block to downsample
    the input. Softmax activation is used in the last layer to output a
    probability distribution across all possible categories.

    :return: neural network model with layers added
    """
    model = tf.keras.Sequential()

    # Convolutional block
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D((32), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.BatchNormalization())

    # Fully connected block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_outputs, activation="softmax"))

    return model


def main():
    """Builds and compiles a neural network model."""
    args = get_arguments()

    with np.load(args.dataset_filepath) as data:
        inputs = data["inputs"]
        labels = data["labels"]

    input_shape = inputs.shape[1:]
    mapping = get_mapping(labels)

    model = build(input_shape, len(mapping))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    if not args.silent:
        model.summary()
    model.save(args.export)


if __name__ == "__main__":
    main()
