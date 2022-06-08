"""Loads tracks and corresponding per track metadata to create a dataset.

Dataset created is not raw data; input preprocessing and label encoding are
    applied.

Structure of files pointed to by tracks_filepath and metadata_filepath
    are expected to match the structure of the audio data and per track
    metadata files sourced by the Free Music Archive (FMA) dataset.

Example:
    python3 configure.py [-h] tracks_filepath metadata_filepath outfile
"""

import argparse
import csv
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any, Dict, Sequence, Tuple

import audioread
import librosa
import numpy as np

import constants
from mapping import get_mapping
from preprocess.encoding import integer_encode
from preprocess.feature_engineering import audio_to_image_representation
from preprocess.feature_engineering import waveform_to_normalized_mel_spectrogram  # noqa: E501


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
    number = -1
    found = False

    for row_number, row in enumerate(csv_file):
        if row_number == constants.CSV_HEADER_ROWS_NUM:
            break

        for column_number, column in enumerate(row):
            if column == column_name:
                number = column_number
                found = True

    if not found:
        raise ValueError(
            f"Unable to locate column '{column_name}' in .csv file"
        )
    return number


def get_genres(
    track_paths: Sequence[str], metadata_filepath: str
) -> Sequence[str]:
    """Creates a sequence of genres corresponding to the tracks.

    Genre is identified by the track name as provided in the file path.

    :param track_paths: sequence of file paths to tracks
    :param metadata_filepath: path to .csv file of per track metadata
    :return: sequence of genres
    """
    genres = []
    labels = get_labels(metadata_filepath)

    for track in track_paths:
        name = trim_track_name(track)
        genres.append(labels[name])
    return genres


def get_labels(metadata_filepath: str) -> Dict[str, str]:
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


def get_track_paths(directory: str) -> list:
    """Searches directory for files containing audio format extensions.

    Allowed audio format extensions are defined by TRACK_EXTENSIONS.

    :param directory: directory to be searched
    :return: list of paths to audio files
    """
    suffix = 1
    paths = []

    for directory_path, _, files in os.walk(directory):
        for file in files:
            if "." in file:
                extension = file.split(".")[suffix]
                if extension in constants.TRACK_EXTENSIONS:
                    file_path = os.path.join(directory_path, file)
                    paths.append(file_path)
    return paths


def load_track(track_path: str) -> np.ndarray:
    """Loads track waveform.

    Failure to load track results in None being returned.

    :param track_path: path to audio file
    :return: array representing track waveform or None
    """
    try:
        waveform, _ = librosa.load(track_path,
                                   sr=constants.TRACK_SAMPLING_RATE_HZ,
                                   duration=constants.TRACK_DURATION_SECONDS)
    except audioread.exceptions.NoBackendError:
        print(f"Error loading '{track_path}'", file=sys.stderr)
        waveform = None
    else:
        print(f"Loaded '{track_path}'")
    finally:
        return waveform


def preprocess(track_path: str) -> Tuple[np.ndarray, str]:
    """Loads track before applying feature engineering.

    Provided file path is returned unmodified to maintain a reference to the
        newly preprocessed data, such as if used in a parallelized map
        function.

    :param track_path: path to audio file
    :return: tuple of (preprocessed data, track file path)
    """
    waveform = load_track(track_path)

    if waveform is not None:
        waveform = verify_length(waveform)
        mel_spectrogram = waveform_to_normalized_mel_spectrogram(
            waveform, constants.TRACK_SAMPLING_RATE_HZ
        )
        grayscale = audio_to_image_representation(mel_spectrogram)
        preprocessed = grayscale
    else:
        preprocessed = None

    return preprocessed, track_path


def trim_track_name(path: str) -> str:
    """Removes prepended file path, leading zeros, and file extension.

    Example:
        '/foo/bar/baz/001230.mp3' returns '1230'

    :param path: path to track file
    :return:
        track name with prepended file path, leading zeros, and extension
        removed
    """
    root = 0

    track = Path(path).name  # Removes parent directories
    track = track.lstrip("0")
    if "." in track:
        track = track.split(".")[root]
    return track


def unpack_and_clean(
    pairs: Sequence[Tuple[Any, Any]]
) -> Tuple[Sequence[Any], Sequence[Any]]:
    """Decouples sequence of paired data; incomplete pairs are omitted.

    :param pairs: sequence of tuples
    :return: tuple of (first items sequence, second items sequence)
    """
    first_items = []
    second_items = []

    for first, second in pairs:
        if first is not None and second is not None:
            first_items.append(first)
            second_items.append(second)

    return first_items, second_items


def verify_length(waveform: Sequence[int]) -> np.ndarray:
    """Checks signal length and adjusts to the expected length.

    Expected length is defined as the product of TRACK_SAMPLING_RATE_HZ and
        TRACK_DURATION_SECONDS.

    Tracks containing an incompatible number of samples are zero padded or
        truncated accordingly.

    :param waveform: signal to be checked
    :return: signal of the expected length
    """
    samples = 0
    expected_num_samples = (constants.TRACK_SAMPLING_RATE_HZ
                            * constants.TRACK_DURATION_SECONDS)

    if waveform.shape[samples] != expected_num_samples:
        waveform = librosa.util.fix_length(waveform, size=expected_num_samples)
    return waveform


def main():
    """Loads tracks and per track metadata. Input preprocessing and label
    encoding are applied to create dataset.
    """
    args = get_arguments()
    track_paths = get_track_paths(args.tracks_filepath)

    with multiprocessing.Pool() as pool:
        track_data = pool.map(preprocess, track_paths)

    tracks_preprocessed, track_paths = unpack_and_clean(track_data)
    genres = get_genres(track_paths, args.metadata_filepath)

    _, mapping_dictionary = get_mapping(genres)
    labels_encoded = integer_encode(genres, mapping_dictionary)

    assert len(tracks_preprocessed) == len(labels_encoded)
    np.savez(args.outfile, inputs=tracks_preprocessed, labels=labels_encoded)


if __name__ == "__main__":
    main()
