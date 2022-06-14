"""Loads tracks and corresponding per track metadata to create a dataset.

Dataset created is not raw data; input preprocessing, label encoding, and
    segmentation are applied.

Structure of files pointed to by tracks_filepath and metadata_filepath
    are expected to match the structure of the audio data and per track
    metadata files sourced by the Free Music Archive (FMA) dataset.

Example:
    python create.py [-h] tracks_filepath metadata_filepath outfile
"""

import argparse
import csv
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import audioread
import librosa
import numpy as np

import constants
from preprocess.encoding import integer_encode
from preprocess.feature_engineering import audio_to_image_representation
from preprocess.feature_engineering import waveform_to_normalized_mel_spectrogram  # noqa: E501
from preprocess.misc import verify_and_fix_length
from preprocess.segmenting import segment


def apply_feature_engineering(waveform: Sequence[int]) -> np.ndarray:
    """Applies feature engineering to the waveform of a track.

    :param waveform: audio data sequence corresponding to a waveform
    :return: preprocessed audio data
    """
    mel_spectrogram = waveform_to_normalized_mel_spectrogram(
        waveform, constants.TRACK_SAMPLING_RATE_HZ
    )
    image = audio_to_image_representation(mel_spectrogram)
    preprocessed = image

    return preprocessed


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_filepath",
                        help="path to directory of tracks")
    parser.add_argument("metadata_filepath",
                        help="path to .csv file of per track metadata")
    parser.add_argument("outfile",
                        help="path to location dataset is to be saved")

    args = parser.parse_args()
    return args


def get_column(csv_file: csv.reader, column_name: str) -> int:
    """Finds column number of the matching column name.

    File pointer is modified during function call; seek() must be used
        accordingly to reset position.

    :param csv_file: reader object of .csv file to search
    :param column_name: name of column to locate
    :raises ValueError: provided column name was not found
    :return: column number of matching column name
    """
    for row_number, row in enumerate(csv_file):
        if row_number == constants.CSV_HEADER_ROWS_NUM:
            break

        for column_number, column in enumerate(row):
            if column == column_name:
                return column_number
    raise ValueError(f"Unable to locate column '{column_name}' in .csv file")


def get_metadata_mapping(metadata_filepath: str) -> Dict[str, str]:
    """Associates input names with label names.

    File pointer is modified during function call.

    :param metadata_filepath: path to .csv file of per input metadata
    :return: dictionary containing input name keys and label name values
    """
    labels = {}

    with open(metadata_filepath, "r") as file:
        metadata = csv.reader(file)

        input_column = get_column(metadata, constants.INPUT_NAME)
        file.seek(0)
        label_column = get_column(metadata, constants.LABEL_NAME)
        file.seek(0)

        for _, row in enumerate(metadata, start=constants.CSV_HEADER_ROWS_NUM):
            input = row[input_column]
            label = row[label_column]

            if input and label:
                labels[input] = label
    return labels


def get_track_filepaths(directory: str) -> list:
    """Searches directory for files containing audio format extensions.

    Allowed audio format extensions are defined by TRACK_EXTENSIONS.

    :param directory: directory to be searched
    :return: list of paths to audio files
    """
    filepaths = []

    for directory_filepath, _, files in os.walk(directory):
        for file in files:
            if "." in file:
                suffix = 1
                extension = file.split(".")[suffix]

                if extension in constants.TRACK_EXTENSIONS_SUPPORTED:
                    filepath = os.path.join(directory_filepath, file)
                    filepaths.append(filepath)
    return filepaths


def get_unique_dataset_labels(
    track_filepaths: Sequence[str],
    metadata_mapping: Dict[str, str]
) -> Sequence[str]:
    """Creates a sequence of unique dataset labels.

    To ensure consistent output, the labels sequence is returned in sorted
        order.

    :param track_filepaths: sequence of track filepaths
    :param metadata_mapping:
        dictionary containing input name keys and label name values
    :return: ordered sequence of unique dataset labels
    """
    unique_dataset_labels = set()

    for track in track_filepaths:
        name = trim_track_name(track)
        label = metadata_mapping[name]

        if label not in unique_dataset_labels:
            unique_dataset_labels.add(label)

    dataset_mapping = sorted(unique_dataset_labels)
    return dataset_mapping


def load_and_preprocess_track(track_filepath: str) -> Tuple[np.ndarray, str]:
    """Loads track waveform and applies preprocessing steps.

    Failure to load track, e.g., if the audio file is corrupted, results
        in None being returned.

    Provided filepath is returned unmodified to maintain a reference to the
        preprocessed data.

    :param track_filepath: path to audio file
    :return: tuple of (preprocessed audio data, track filepath)
    """
    waveform = load_track(track_filepath)

    if waveform is not None:
        waveform = verify_and_fix_length(waveform)
        preprocessed = apply_feature_engineering(waveform)
    else:
        preprocessed = None

    return preprocessed, track_filepath


def load_track(track_filepath: str) -> np.ndarray:
    """Loads track waveform.

    Failure to load track, e.g., if the audio file is corrupted, results
        in None being returned.

    :param track_filepath: path to audio file
    :return: array of track waveform or None
    """
    try:
        waveform, _ = librosa.load(track_filepath,
                                   sr=constants.TRACK_SAMPLING_RATE_HZ,
                                   duration=constants.TRACK_DURATION_SECONDS)
    except audioread.exceptions.NoBackendError:
        print(f"NoBackendError: unable to load file '{track_filepath}'",
              file=sys.stderr)
        waveform = None
    finally:
        return waveform


def trim_track_name(path: str) -> str:
    """Removes prepended file path, leading zeros, and file extension.

    Example:
        '/foo/bar/baz/001230.mp3' returns '1230'

    :param path: path to track file
    :return: track name, exclusively
    """
    track = Path(path).name
    track = track.lstrip("0")

    if "." in track:
        root = 0
        track = track.split(".")[root]
    return track


def main():
    """Loads tracks and per track metadata. Input preprocessing, label
    encoding, and segmentation are applied to create a dataset.
    """
    args = get_arguments()

    filepaths = get_track_filepaths(args.tracks_filepath)
    input_to_label_mapping = get_metadata_mapping(args.metadata_filepath)

    unique_labels = get_unique_dataset_labels(filepaths,
                                              input_to_label_mapping)
    encoded_labels = integer_encode(unique_labels)

    with multiprocessing.Pool() as pool:
        dataset_inputs = []
        dataset_labels = []

        for preprocessed_track, filepath in pool.imap(
            load_and_preprocess_track, filepaths, chunksize=16
        ):
            if preprocessed_track is None:
                continue

            name = trim_track_name(filepath)
            genre = input_to_label_mapping[name]
            genre_encoded = encoded_labels[genre]

            segmented_track, segmented_genre = segment(
                preprocessed_track, constants.SEGMENTS_NUM, genre_encoded
            )
            del preprocessed_track

            dataset_inputs.extend(segmented_track)
            dataset_labels.extend(segmented_genre)
    np.savez(args.outfile, inputs=dataset_inputs, labels=dataset_labels)


if __name__ == "__main__":
    main()
