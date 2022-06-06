"""Loads tracks and corresponding per track metadata sourced by the Free Music
Archive (FMA) dataset to create NumPy arrays of track waveforms and genres.

Example:
    python3 create.py [-h] tracks_filepath metadata_filepath outfile
"""

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import audioread
import librosa
import numpy as np

from constants import CSV_HEADER_ROWS_NUM
from constants import INPUT_NAME
from constants import LABEL_NAME
from constants import TRACK_DURATION_SECONDS
from constants import TRACK_EXTENSIONS
from constants import TRACK_SAMPLING_RATE_HZ


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


def get_labels(metadata_filepath: str) -> Dict[str, str]:
    """Associates input names with label names.

    :param metadata_filepath: path to .csv file of per input metadata
    :return: dictionary containing input name keys and label name values
    """
    labels = {}

    with open(metadata_filepath, "r") as file:
        metadata = csv.reader(file)

        file.seek(0)
        input_column = get_column(metadata, INPUT_NAME)
        file.seek(0)
        label_column = get_column(metadata, LABEL_NAME)
        file.seek(0)

        for _, row in enumerate(metadata, start=CSV_HEADER_ROWS_NUM):
            input = row[input_column]
            label = row[label_column]

            if not input or not label:
                continue
            labels[input] = label
    return labels


def get_column(csv_file: csv.reader, column_name: str) -> int:
    """Gets column number of the matching column name.

    Function modifies position of file pointer; seek() must be used accordingly
    to reset position.

    :param csv_file: reader object of .csv file to search
    :param column_name: name of column to locate
    :raises ValueError: provided column name was not found
    :return: column number corresponding to matching column name
    """
    for row_number, row in enumerate(csv_file):
        if row_number == CSV_HEADER_ROWS_NUM:
            break

        for column_number, column in enumerate(row):
            if column == column_name:
                return column_number

    raise ValueError(f"Unable to locate column '{column_name}' in .csv file")


def get_track_paths(directory: str) -> list:
    """Searches directory for files containing audio format extensions.

    Audio format extensions to be searched for are defined by TRACK_EXTENSIONS.

    :param directory: directory to be searched
    :return: list of paths to audio files
    """
    suffix = 1
    paths = []

    for directory_path, _, files in os.walk(directory):
        for file in files:
            if "." in file:
                extension = file.split(".")[suffix]
                if extension in TRACK_EXTENSIONS:
                    file_path = os.path.join(directory_path, file)
                    paths.append(file_path)
    return paths


def create(
    track_paths: Sequence[str],
    genres: Dict[str, str]
) -> Tuple[Sequence[Sequence[int]], Sequence[str]]:
    """Creates lists of track waveforms and corresponding genres.

    :param track_paths: sequence of track file paths
    :param genres: dictionary of track name keys and genre labels
    :return: tuple of lists containing input track waveforms and genre labels
    """
    num_samples = 0  # Represents index, not actual number of audio features

    # Represents actual number of audio features
    expected_num_samples = TRACK_SAMPLING_RATE_HZ * TRACK_DURATION_SECONDS

    inputs = []
    labels = []

    for path in track_paths:
        try:
            waveform, _ = librosa.load(path,
                                       sr=TRACK_SAMPLING_RATE_HZ,
                                       duration=TRACK_DURATION_SECONDS)
        except audioread.exceptions.NoBackendError:
            print(f"Failed to load '{path}'", file=sys.stderr)
        else:
            if waveform.shape[num_samples] < expected_num_samples:
                # Zero padding
                waveform = librosa.util.fix_length(waveform,
                                                   size=expected_num_samples)
            track = trim_track_name(path)
            genre = genres[track]

            inputs.append(waveform)
            labels.append(genre)
    return inputs, labels


def trim_track_name(path: str) -> str:
    """Removes prepended file path, leading zeros, and file extension.

    Example:
        '/foo/bar/baz/001230.mp3' returns '1230'

    :param path: file path to track
    :return:
        track name with prepended file path, leading zeros, and extension
        removed
    """
    root = 0

    track = Path(path).name  # Excludes parent directories
    track = track.lstrip("0")
    if "." in track:
        track = track.split(".")[root]
    return track


def main():
    """Loads tracks and per track metadata sourced by the Free Music Archive
    (FMA) dataset to create NumPy arrays of track waveforms and genres.

    Structure of files pointed to by tracks_filepath and metadata_filepath
    are expected to match the structure of the audio data and per track
    metadata files sourced by the Free Music Archive (FMA) dataset.
    """
    args = get_arguments()

    track_paths = get_track_paths(args.tracks_filepath)
    genres = get_labels(args.metadata_filepath)

    inputs, labels = create(track_paths, genres)
    assert len(inputs) == len(labels)
    np.savez(args.outfile, inputs=inputs, labels=labels)


if __name__ == "__main__":
    main()
