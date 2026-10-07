from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REFERENCE_PATH = Path("demo/reference_profile.json")
VALENCE_MODEL_PATH = Path("models/valence_model.joblib")
AROUSAL_MODEL_PATH = Path("models/arousal_model.joblib")

BANDS = ["theta", "alpha", "beta", "gamma"]

BAND_LABELS = {
    "theta": "Theta (4–8 Hz)",
    "alpha": "Alpha (8–13 Hz)",
    "beta": "Beta (13–30 Hz)",
    "gamma": "Gamma (30–45 Hz)",
}

CHANNELS = [f"Ch{i}" for i in range(1, 33)]


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

valence_model = joblib.load(VALENCE_MODEL_PATH)
arousal_model = joblib.load(AROUSAL_MODEL_PATH)


# ---------------------------------------------------------
# Build a non-participant-specific reference profile
# ---------------------------------------------------------
#
# The public demo does NOT redistribute raw DEAP EEG trials.
#
# We use the median of each engineered feature across the
# local feature table to construct an aggregate reference
# profile. User controls modify this aggregate profile.
#
# This means the demo exposes neither raw EEG nor an
# individual participant's trial.
# ---------------------------------------------------------

feature_columns = [
    f"Ch{channel}_{band}"
    for channel in range(1, 33)
    for band in BANDS
]

import json

with REFERENCE_PATH.open() as f:
    reference_data = json.load(f)

reference_profile = pd.Series(reference_data)

missing = [column for column in feature_columns if column not in reference_profile.index]

if missing:
    raise ValueError(
        f"Missing expected EEG features: {missing[:5]}"
    )

reference_profile = reference_profile[feature_columns]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def build_feature_vector(theta, alpha, beta, gamma):
    """
    Modify the aggregate reference EEG profile using
    user-selected band multipliers.
    """

    multipliers = {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    vector = reference_profile.copy()

    for band, multiplier in multipliers.items():
        band_columns = [
            column
            for column in feature_columns
            if column.endswith(f"_{band}")
        ]

        vector.loc[band_columns] *= multiplier

    return vector


def make_heatmap(vector):
    """
    Display the 32-channel × 4-band feature representation.
    Log scaling is used only for visualization because EEG
    band-power values are strongly right-skewed.
    """

    matrix = np.zeros((32, 4))

    for channel_idx in range(32):
        for band_idx, band in enumerate(BANDS):
            column = f"Ch{channel_idx + 1}_{band}"
            matrix[channel_idx, band_idx] = vector[column]

    display_matrix = np.log10(matrix + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))

    image = ax.imshow(
        display_matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title("32-Channel EEG Frequency Profile")
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("EEG Channel")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Theta", "Alpha", "Beta", "Gamma"])

    ax.set_yticks(range(32))
    ax.set_yticklabels(CHANNELS, fontsize=7)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10(Band Power)")

    fig.tight_layout()

    return fig


def make_band_plot(vector):
    """
    Aggregate each frequency band across all 32 channels.
    Median is used because the feature distributions are
    highly right-skewed.
    """

    values = []

    for band in BANDS:
        columns = [
            column
            for column in feature_columns
            if column.endswith(f"_{band}")
        ]

        values.append(float(np.median(vector[columns])))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Theta", "Alpha", "Beta", "Gamma"],
        values,
    )

    ax.set_title("Median Band Power Across 32 EEG Channels")
    ax.set_ylabel("Band Power")

    fig.tight_layout()

    return fig


def get_probability(model, X):
    """
    Return probability/confidence when supported.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        prediction = int(model.predict(X)[0])

        classes = list(model.classes_)
        index = classes.index(prediction)

        return prediction, float(probabilities[index])

    prediction = int(model.predict(X)[0])

    return prediction, None


def predict(theta, alpha, beta, gamma):
    """
    Run both final NeuroFocus models on the synthetic
    aggregate EEG feature profile.
    """

    vector = build_feature_vector(
        theta,
        alpha,
        beta,
        gamma,
    )

    X = pd.DataFrame(
        [vector.values],
        columns=feature_columns,
    )

    valence_prediction, valence_confidence = get_probability(
        valence_model,
        X,
    )

    arousal_prediction, arousal_confidence = get_probability(
        arousal_model,
        X,
    )

    valence_label = (
        "Positive" if valence_prediction == 1 else "Negative"
    )

    arousal_label = (
        "High" if arousal_prediction == 1 else "Low"
    )

    if valence_confidence is not None:
        valence_text = (
            f"### Valence: **{valence_label}**\n"
            f"Model confidence: **{valence_confidence:.1%}**"
        )
    else:
        valence_text = f"### Valence: **{valence_label}**"

    if arousal_confidence is not None:
        arousal_text = (
            f"### Arousal: **{arousal_label}**\n"
            f"Model confidence: **{arousal_confidence:.1%}**"
        )
    else:
        arousal_text = f"### Arousal: **{arousal_label}**"

    heatmap = make_heatmap(vector)
    band_plot = make_band_plot(vector)

    return (
        valence_text,
        arousal_text,
        heatmap,
        band_plot,
    )


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------

with gr.Blocks(title="NeuroFocus") as demo:

    gr.Markdown(
        """
        # 🧠 NeuroFocus

        ### EEG-Based Affective State Classification

        Explore how changes in EEG frequency-band activity
        affect predictions from models trained on the
        **DEAP dataset**.

        NeuroFocus predicts two affective dimensions:

        - **Valence:** Negative ↔ Positive
        - **Arousal:** Low ↔ High

        The final models were evaluated using EEG from
        **completely unseen participants**.
        """
    )

    gr.Markdown(
        """
        > **How to use this demo**
        >
        > Each slider changes the relative strength of one EEG
        > frequency band compared with an aggregate reference
        > profile.
        >
        > `1.0×` represents the reference level.
        > Values below 1 reduce that band; values above 1
        > increase it.
        """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("## 🎛️ EEG Frequency Controls")

            theta = gr.Slider(
                minimum=0.25,
                maximum=3.0,
                value=1.0,
                step=0.05,
                label=BAND_LABELS["theta"],
            )

            alpha = gr.Slider(
                minimum=0.25,
                maximum=3.0,
                value=1.0,
                step=0.05,
                label=BAND_LABELS["alpha"],
            )

            beta = gr.Slider(
                minimum=0.25,
                maximum=3.0,
                value=1.0,
                step=0.05,
                label=BAND_LABELS["beta"],
            )

            gamma = gr.Slider(
                minimum=0.25,
                maximum=3.0,
                value=1.0,
                step=0.05,
                label=BAND_LABELS["gamma"],
            )

            predict_button = gr.Button(
                "Run NeuroFocus",
                variant="primary",
            )

            reset_button = gr.Button("Reset")

        with gr.Column():

            gr.Markdown("## 🔮 Model Prediction")

            valence_output = gr.Markdown()
            arousal_output = gr.Markdown()

            gr.Markdown(
                """
                **Final models**

                Valence → Band Power + Logistic Regression

                Arousal → Band Power + RBF SVM
                """
            )

    gr.Markdown("## 📊 EEG Feature Explorer")

    with gr.Row():

        heatmap_output = gr.Plot(
            label="32-Channel Frequency Profile"
        )

        band_output = gr.Plot(
            label="Frequency-Band Summary"
        )

    gr.Markdown(
        """
        ---

        ## 🧪 What Did NeuroFocus Find?

        Three approaches were compared using validation
        participants that were completely separate from the
        training participants.

        | Representation | Valence Balanced Accuracy | Arousal Balanced Accuracy |
        |---|---:|---:|
        | **Band Power** | **58.50%** | **59.22%** |
        | Differential Entropy | 55.50% | 56.34% |
        | EEGNet-style Network | 50.00% | 53.75% |

        The strongest classical models were then frozen,
        retrained on the complete development set, and
        evaluated once on five held-out participants.

        | Final Test | Accuracy | Balanced Accuracy |
        |---|---:|---:|
        | Valence | 54.50% | **52.70%** |
        | Arousal | 49.50% | **52.50%** |

        ### Why are the final scores modest?

        EEG differs substantially between people.

        NeuroFocus deliberately prevents EEG from the test
        participants from appearing during training or model
        selection. The model therefore has to transfer what
        it learned from one group of brains to people it has
        never encountered.

        The validation-to-test drop demonstrates how difficult
        that cross-subject generalization problem is.

        ---

        ## 🔐 DEAP Data & Privacy

        NeuroFocus was developed using the **DEAP Dataset for
        Emotion Analysis using Physiological Signals**.

        This public demo does **not redistribute raw DEAP EEG
        recordings or individual participant trials**.

        The interactive controls operate on an aggregate
        feature profile and are intended to demonstrate model
        behavior rather than reproduce an individual
        participant's EEG recording.

        Users who want to reproduce the full experiment should
        obtain DEAP separately from the official dataset source.

        ---

        ## ⚠️ Important

        NeuroFocus is an experimental machine-learning project.

        It is **not a clinical system, diagnostic tool, or
        reliable emotion detector**.

        The sliders demonstrate how the trained models respond
        to controlled changes in engineered EEG features.
        Predictions should not be interpreted as measurements
        of a person's actual emotional state.
        """
    )

    predict_button.click(
        fn=predict,
        inputs=[theta, alpha, beta, gamma],
        outputs=[
            valence_output,
            arousal_output,
            heatmap_output,
            band_output,
        ],
    )

    reset_button.click(
        fn=lambda: (1.0, 1.0, 1.0, 1.0),
        outputs=[theta, alpha, beta, gamma],
    )

    demo.load(
        fn=predict,
        inputs=[theta, alpha, beta, gamma],
        outputs=[
            valence_output,
            arousal_output,
            heatmap_output,
            band_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
