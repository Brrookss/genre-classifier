"""Builds and compiles a neural network model.

Example:
    python configure.py [-h] [-s] dataset_filepath export
"""

import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf

from constants import LEARNING_RATE


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_filepath",
                        help="path to .npz file of dataset")
    parser.add_argument("export",
                        help="path to location model is to be saved")

    args = parser.parse_args()
    return args


def build(
    input_shape: Tuple[int, int, int],
    num_categories: int
) -> tf.keras.Model:
    """Stacks layers to build a neural network model.

    Model is composed of five two-dimensional convolutional blocks followed by
        a fully connected block. Resizing and max pooling layers are
        implemented to downsample the input.

    :param input_shape: dimension of a single input
    :param num_categories: number of catagorical targets
    :return: neural network model with layers added
    """
    model = tf.keras.Sequential()

    # Convolutional block 1
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Resizing(128, 128))
    model.add(tf.keras.layers.Conv2D((64), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convolutional block 2
    model.add(tf.keras.layers.Conv2D((64), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convoutional block 3
    model.add(tf.keras.layers.Conv2D((64), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convoutional block 4
    model.add(tf.keras.layers.Conv2D((64), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convoutional block 5
    model.add(tf.keras.layers.Conv2D((64), (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Fully connected block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_categories))
    return model


def main():
    """Builds and compiles a neural network model."""
    args = get_arguments()

    with np.load(args.dataset_filepath, mmap_mode="r") as dataset:
        inputs = dataset["inputs"]
        labels = dataset["labels"]

    features = categories = 1
    input_shape = inputs.shape[features:]
    num_categories = labels.shape[categories]

    model = build(input_shape, num_categories)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    model.summary()
    model.save(args.export)


if __name__ == "__main__":
    main()
