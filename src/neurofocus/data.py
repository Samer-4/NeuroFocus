from pathlib import Path
import pickle

import numpy as np


EEG_CHANNELS = 32
SAMPLING_RATE = 128


def load_deap(data_dir: str | Path):
    """
    Load DEAP preprocessed participant files.

    Returns
    -------
    eeg : np.ndarray
        EEG data with shape:
        (participants, trials, 32 EEG channels, samples)

    labels : np.ndarray
        DEAP ratings with shape:
        (participants, trials, 4)

        Label order:
        0 = Valence
        1 = Arousal
        2 = Dominance
        3 = Liking
    """
    data_dir = Path(data_dir)

    files = sorted(data_dir.glob("s*.dat"))

    if not files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in {data_dir}. "
            "Expected files such as s01.dat."
        )

    all_eeg = []
    all_labels = []

    for path in files:
        with path.open("rb") as file:
            participant = pickle.load(file, encoding="latin1")

        eeg = participant["data"][:, :EEG_CHANNELS, :]
        labels = participant["labels"]

        all_eeg.append(eeg)
        all_labels.append(labels)

    return np.asarray(all_eeg), np.asarray(all_labels)
