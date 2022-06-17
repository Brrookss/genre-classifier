"""Contains functionality to display neural network model training metrics."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf


def display_test_accuracy(
    predictions: Sequence[Sequence[int]],
    labels: Sequence[Sequence[int]]
) -> None:
    """Displays a fraction corresponding to model accuracy during testing.

    :param predictions: sequence of one-hot encoded predictions
    :param labels: sequence of one-hot encoded labels
    :return: None
    """
    predicted = []
    actual = []

    for one_hot_prediction, one_hot_label in zip(predictions, labels):
        prediction = np.argmax(one_hot_prediction)
        label = np.argmax(one_hot_label)

        predicted.append(prediction)
        actual.append(label)

    correct = np.where(np.array(predicted) == np.array(actual))[0]
    accuracy = len(correct) / len(predicted)
    print(f"test_accuracy: {accuracy:.4f}")


def display_train_accuracy(history: tf.keras.callbacks.History) -> None:
    """Displays a plot corresponding to model accuracy during training.

    :param history: object returned by model after training
    :return: None
    """
    _, ax = plt.subplots()

    ax.plot(history.history["accuracy"], label="Training Dataset")
    ax.plot(history.history["val_accuracy"], label="Validation Dataset")

    ax.set_title("Accuracy Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.show()


def display_confusion(matrix: Sequence[Sequence[int]]) -> None:
    """Displays a contingency table comparing predicted and actual results.

    :param matrix: two-dimensional sequence representing a contingency table
    :return: None
    """
    _, ax = plt.subplots()

    df = pd.DataFrame(matrix)
    sn.heatmap(df, annot=True, fmt="d")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.show()


def display_loss(history: tf.keras.callbacks.History) -> None:
    """Displays a plot corresponding to model loss during training.

    :param history: object returned by model after training
    :return: None
    """
    _, ax = plt.subplots()

    ax.plot(history.history["loss"], label="Training Dataset")
    ax.plot(history.history["val_loss"], label="Validation Dataset")

    ax.set_title("Loss Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()


def get_confusion_matrix_from_one_hot(
    predictions: Sequence[Sequence[int]],
    labels: Sequence[Sequence[int]]
) -> Sequence[Sequence[int]]:
    """Creates a confusion matrix using one-hot encoded predictions and labels.

    :param predictions: sequence of one-hot encoded predictions
    :param labels: sequence of one-hot encoded labels
    :return: two-dimensional sequence representing a contingency table
    """
    predicted = []
    actual = []

    for one_hot_prediction, one_hot_label in zip(predictions, labels):
        prediction = np.argmax(one_hot_prediction)
        label = np.argmax(one_hot_label)

        predicted.append(prediction)
        actual.append(label)

    matrix = tf.math.confusion_matrix(actual, predicted)
    return matrix
