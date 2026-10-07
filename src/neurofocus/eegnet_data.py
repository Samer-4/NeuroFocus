from pathlib import Path

import numpy as np

from neurofocus.data import load_deap


FS = 128
BASELINE_SECONDS = 3
BASELINE_SAMPLES = FS * BASELINE_SECONDS


def prepare_eegnet_data(
    data_dir: str | Path,
    threshold: float = 5.0,
):
    """
    Prepare DEAP EEG for EEGNet.

    Returns
    -------
    X : np.ndarray
        Shape:
        (trials, 1, 32, 7680)

    y_valence : np.ndarray
        0 = negative
        1 = positive

    y_arousal : np.ndarray
        0 = low arousal
        1 = high arousal

    groups : np.ndarray
        Participant ID for every trial.
        Used to prevent participant leakage.
    """

    eeg, labels = load_deap(data_dir)

    # Remove first 3 seconds of baseline EEG.
    eeg = eeg[:, :, :, BASELINE_SAMPLES:]

    participants = eeg.shape[0]
    trials = eeg.shape[1]
    channels = eeg.shape[2]
    samples = eeg.shape[3]

    # Flatten participant + trial dimensions.
    X = eeg.reshape(
        participants * trials,
        channels,
        samples,
    ).astype(np.float32)

    # EEGNet expects:
    # (examples, 1, channels, time)
    X = X[:, np.newaxis, :, :]

    valence = labels[:, :, 0].reshape(-1)
    arousal = labels[:, :, 1].reshape(-1)

    y_valence = (
        valence >= threshold
    ).astype(np.int64)

    y_arousal = (
        arousal >= threshold
    ).astype(np.int64)

    groups = np.repeat(
        np.arange(participants),
        trials,
    )

    return X, y_valence, y_arousal, groups
