import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt


BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def segment_trial(
    trial: np.ndarray,
    window_size: int = 256,
    step_size: int = 128,
):
    """
    Split one EEG trial into overlapping 2-second windows.
    """
    for start in range(
        0,
        trial.shape[-1] - window_size + 1,
        step_size,
    ):
        yield trial[:, start:start + window_size]


def band_power_features(
    segment: np.ndarray,
    fs: int = 128,
) -> dict[str, float]:
    """
    Extract Theta, Alpha, Beta, and Gamma band power
    from all 32 EEG channels.
    """
    features = {}

    for channel_idx, signal in enumerate(segment):

        frequencies, psd = welch(
            signal,
            fs=fs,
            nperseg=fs,
        )

        for band_name, (low, high) in BANDS.items():

            mask = (
                (frequencies >= low)
                & (frequencies <= high)
            )

            power = np.mean(psd[mask])

            features[
                f"Ch{channel_idx + 1}_{band_name}"
            ] = float(power)

    return features


def differential_entropy_features(
    segment: np.ndarray,
    fs: int = 128,
) -> dict[str, float]:
    """
    Extract differential entropy from Theta, Alpha,
    Beta, and Gamma bands across all EEG channels.

    32 channels x 4 bands = 128 DE features.
    """
    features = {}

    for channel_idx, signal in enumerate(segment):

        for band_name, (low, high) in BANDS.items():

            sos = butter(
                4,
                [low, high],
                btype="bandpass",
                fs=fs,
                output="sos",
            )

            filtered = sosfiltfilt(
                sos,
                signal,
            )

            variance = np.var(
                filtered,
                ddof=1,
            )

            variance = max(
                variance,
                1e-12,
            )

            de = 0.5 * np.log(
                2 * np.pi * np.e * variance
            )

            features[
                f"Ch{channel_idx + 1}_{band_name}_de"
            ] = float(de)

    return features


def trial_features(
    trial: np.ndarray,
    fs: int = 128,
    window_size: int = 256,
    step_size: int = 128,
    feature_type: str = "bandpower",
) -> dict[str, float]:
    """
    Create one 128-feature representation
    for an entire EEG trial.

    feature_type:
        "bandpower" = original PSD band-power features
        "de"        = differential entropy features
    """
    baseline_samples = 3 * fs
    trial = trial[:, baseline_samples:]

    if feature_type == "bandpower":
        extractor = band_power_features
    elif feature_type == "de":
        extractor = differential_entropy_features
    else:
        raise ValueError(
            "feature_type must be 'bandpower' or 'de'"
        )

    window_features = []

    for segment in segment_trial(
        trial,
        window_size=window_size,
        step_size=step_size,
    ):
        window_features.append(
            extractor(segment, fs=fs)
        )

    feature_df = pd.DataFrame(window_features)

    return feature_df.mean().to_dict()


def build_feature_table(
    eeg: np.ndarray,
    deap_labels: np.ndarray,
    threshold: float = 5.0,
    fs: int = 128,
    window_size: int = 256,
    step_size: int = 128,
    feature_type: str = "bandpower",
) -> pd.DataFrame:
    """
    Build one ML-ready row per DEAP trial.
    """
    rows = []

    for participant in range(eeg.shape[0]):

        for trial in range(eeg.shape[1]):

            features = trial_features(
                eeg[participant, trial],
                fs=fs,
                window_size=window_size,
                step_size=step_size,
                feature_type=feature_type,
            )

            valence = float(
                deap_labels[participant, trial, 0]
            )

            arousal = float(
                deap_labels[participant, trial, 1]
            )

            valence_class = int(
                valence >= threshold
            )

            arousal_class = int(
                arousal >= threshold
            )

            affective_state = (
                2 * valence_class
                + arousal_class
            )

            features.update({
                "participant": participant,
                "trial": trial,
                "valence": valence,
                "arousal": arousal,
                "valence_class": valence_class,
                "arousal_class": arousal_class,
                "affective_state": affective_state,
            })

            rows.append(features)

    return pd.DataFrame(rows)
