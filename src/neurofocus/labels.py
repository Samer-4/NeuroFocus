import numpy as np


CLASS_NAMES = {
    0: "negative_low_arousal",
    1: "negative_high_arousal",
    2: "positive_low_arousal",
    3: "positive_high_arousal",
}


def affective_state(
    valence: float,
    arousal: float,
    threshold: float = 5.0,
) -> int:
    """
    Convert valence and arousal ratings into one of four affective states.

    DEAP ratings use a 1-9 scale.
    Ratings >= 5 are treated as high/positive.
    """
    positive = valence >= threshold
    high_arousal = arousal >= threshold

    return int(2 * positive + high_arousal)


def make_trial_labels(
    labels: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    Convert all DEAP valence/arousal ratings into four-class labels.

    Expected input shape:
        (participants, trials, 4)

    Returns:
        (participants, trials)
    """
    valence = labels[..., 0]
    arousal = labels[..., 1]

    positive = valence >= threshold
    high_arousal = arousal >= threshold

    return (
        2 * positive.astype(int)
        + high_arousal.astype(int)
    ).astype(int)
