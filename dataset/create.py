"""Loads tracks and corresponding per track metadata sourced by the
Free Music Archive (FMA) dataset to create NumPy arrays of track waveforms
and genres.

usage: python3 create.py [-h] tracks_fp metadata_fp outfile
"""

import argparse
import csv
from os import listdir
from os import path
from typing import Dict, Sequence, Tuple

import librosa
import numpy as np

from constants import CSV_HEADER_ROWS_NUM
from constants import INPUT_NAME
from constants import LABEL_NAME
from constants import TRACK_DURATION_SECONDS
from constants import TRACK_EXTENSION
from constants import TRACK_SAMPLING_RATE_HZ


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_fp",
                        help="path to directory of tracks")
    parser.add_argument("metadata_fp",
                        help="path to .csv file of per track metadata")
    parser.add_argument("outfile",
                        help="path to location dataset is to be saved")

    args = parser.parse_args()
    return args


def create(
    tracks_fp: str,
    metadata_fp: str
) -> Tuple[Sequence[int], Sequence[str]]:
    """Creates lists of track waveforms and corresponding genres.

    Structure of files pointed to by tracks_fp and metadata_fp are expected
    to match the structure of the MP3-encoded audio data and per track metadata
    files sourced by the Free Music Archive (FMA) dataset, respectively.

    :param tracks_fp: path to directory of tracks
    :param metadata_fp: path to .csv file of per track metadata
    :return: tuple of lists containing input track waveforms and genre labels
    """
    num_samples = 0  # Index
    expected_num_samples = TRACK_SAMPLING_RATE_HZ * TRACK_DURATION_SECONDS

    inputs = []
    labels = []

    genres = get_labels(metadata_fp)
    tracks_directory = listdir(tracks_fp)

    for file in tracks_directory:
        subdirectory_fp = path.join(tracks_fp, file)

        if not path.isdir(subdirectory_fp):
            continue

        track_files = listdir(subdirectory_fp)

        for track in track_files:
            track_fp = path.join(subdirectory_fp, track)

            waveform, _ = librosa.load(track_fp,
                                       sr=TRACK_SAMPLING_RATE_HZ,
                                       duration=TRACK_DURATION_SECONDS)

            if waveform.shape[num_samples] < expected_num_samples:
                waveform = librosa.util.fix_length(waveform,
                                                   size=expected_num_samples)

            track_name = trim_track_name(track, TRACK_EXTENSION)
            genre = genres[track_name]

            inputs.append(waveform)
            labels.append(genre)

    assert len(inputs) == len(labels)
    return inputs, labels


def get_labels(metadata_fp: str) -> Dict[str, str]:
    """Associates input names with label names.

    :param metadata_fp: path to .csv file of per input metadata
    :return: dictionary containing input name keys and label name values
    """
    labels = {}

    with open(metadata_fp, "r") as file:
        metadata = csv.reader(file)

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
    """Gets the column number of the provided column name.

    Function modifies position of file pointer; seek() must be used accordingly
    to reset position.

    :param csv_file: reader object of .csv file to search
    :param column_name: name of column to locate
    :raises ValueError: provided column name was not found
    :return: column number corresponding to provided column name
    """
    found = False

    for r, row in enumerate(csv_file):
        if r == CSV_HEADER_ROWS_NUM:
            break

        for c, column in enumerate(row):
            if column == column_name:
                number = c
                found = True

    if not found:
        raise ValueError(
            f"Unable to locate column '{column_name}' in .csv file"
        )
    return number


def trim_track_name(track: str, extension: str) -> str:
    """Removes leading zeros and file extension from track name.

    :param track: name of track
    :param extension: file extension to be removed
    :return: track name with leading zeros and extension removed
    """
    root = 0

    track = track.lstrip("0")
    if track.endswith(extension):
        track = track.split(".")[root]
    return track


def main():
    """Loads tracks and per track metadata sourced by the
    Free Music Archive (FMA) dataset to create NumPy arrays
    of track waveforms and genres.
    """
    args = get_arguments()

    inputs, labels = create(args.tracks_fp, args.metadata_fp)
    np.savez(args.outfile, inputs=inputs, labels=labels)


if __name__ == "__main__":
    main()
