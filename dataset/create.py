"""Loads tracks and corresponding per track metadata sourced by the
Free Music Archive (FMA) dataset to create NumPy arrays of track waveforms
and genres.

Usage: python3 create.py [-h] tracks_fp metadata_fp dataset_fp
"""

import argparse
import csv
from os import listdir
from os import path
from typing import Dict, Tuple

import librosa
import numpy as np

CSV_HEADER_ROWS = 3
TRACK_DURATION = 29.9


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    TODO: Add description to ArgumentParser()
    TODO: Add help statements to arguments

    :return: object associated with command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_fp")
    parser.add_argument("metadata_fp")
    parser.add_argument("dataset_fp")

    args = parser.parse_args()
    return args


def create(tracks_fp: str, metadata_fp: str) -> Tuple[np.array, np.array]:
    """Creates NumPy arrays of track waveforms and corresponding genres.

    Structure of files pointed to by tracks_fp and metadata_fp are expected
    to match the structure of the MP3-encoded audio data and per track metadata
    files sourced by the Free Music Archive (FMA) dataset, respectively.

    :param tracks_fp: file path to directory of tracks
    :param metadata_fp: file path to .csv file of track metadata
    :return: NumPy arrays containing input track waveforms and genre labels
    """
    inputs = []
    labels = []

    genres = get_genres(metadata_fp)
    tracks_directory = listdir(tracks_fp)

    for file in tracks_directory:
        subdirectory_fp = path.join(tracks_fp, file)

        if not path.isdir(subdirectory_fp):
            continue

        track_files = listdir(subdirectory_fp)
        for track in track_files:
            track_fp = path.join(subdirectory_fp, track)

            waveform, _ = librosa.load(track_fp, sr=None,
                                       duration=TRACK_DURATION)

            track_name = trim_track_name(track)
            genre = genres[track_name]

            inputs.append(waveform)
            labels.append(genre)

    assert len(inputs) == len(labels)
    inputs = np.array(inputs)
    labels = np.array(labels)
    # TODO: Create zipped NumPy array containing inputs and labels
    return inputs, labels


def get_genres(metadata_fp: str) -> Dict[str, str]:
    """Associates track names with their genre names.
    
    :param metadata_fp: file path to .csv file of track metadata
    :returns: dictionary containing track name keys and genre name values
    """
    genres = {}

    with open(metadata_fp, "r") as file:
        metadata = csv.reader(file)

        id_column = get_column(metadata, "track_id")
        file.seek(0)
        genre_column = get_column(metadata, "genre_top")
        file.seek(0)

        for _, row in enumerate(metadata, start=CSV_HEADER_ROWS):
            id = row[id_column]
            genre = row[genre_column]

            if not id or not genre:
                continue
            genres[id] = genre
    return genres


def get_column(csv_file: csv.reader, column_name: str) -> int:
    """Gets the column number of the provided column name.

    Function modifies position of file pointer; seek() must be used accordingly
    to reset position.

    :param csv_file: reader object of .csv file to search
    :param column_name: name of column to locate
    :raises: ValueError if unable to locate provided column name
    :return: column number corresponding to provided column name
    """
    found = False

    for r, row in enumerate(csv_file):
        if r == CSV_HEADER_ROWS:
            break

        for c, column in enumerate(row):
            if column == column_name:
                number = c
                found = True

    if not found:
        raise ValueError("Unable to locate column in .csv file.")
    return number


def trim_track_name(track: str) -> str:
    """Removes leading zeros and extension in track name.
    
    :param track: name of track
    :return: track name with leading zeros and extension removed
    """
    extension = ".mp3"
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
    create(args.tracks_fp, args.metadata_fp)


if __name__ == "__main__":
    main()
