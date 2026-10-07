from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


NON_FEATURE_COLUMNS = [
    "participant",
    "trial",
    "valence",
    "arousal",
    "valence_class",
    "arousal_class",
    "affective_state",
]


def participant_split(
    df: pd.DataFrame,
    random_state: int = 42,
):
    """
    Split by participant so the model is evaluated
    on people it has never seen during training.
    """

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        outer.split(
            df,
            groups=df["participant"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state,
    )

    val_idx, test_idx = next(
        inner.split(
            temp_df,
            groups=temp_df["participant"],
        )
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


def get_features(df: pd.DataFrame):
    return df.drop(columns=NON_FEATURE_COLUMNS)


def build_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        ),
    ])


def evaluate(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "classification_report": classification_report(
            y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_target(
    train_df,
    val_df,
    target,
    display_name,
):
    print("\n" + "=" * 50)
    print(display_name)
    print("=" * 50)

    X_train = get_features(train_df)
    X_val = get_features(val_df)

    y_train = train_df[target]
    y_val = val_df[target]

    print("\nTraining distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation distribution:")
    print(y_val.value_counts().sort_index())

    model = build_model()

    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    metrics = evaluate(
        model,
        X_val,
        y_val,
    )

    print(
        "Validation accuracy:",
        f"{metrics['accuracy']:.4f}",
    )

    print(
        "Validation balanced accuracy:",
        f"{metrics['balanced_accuracy']:.4f}",
    )

    return model, metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="data/processed/eeg_features.csv",
    )

    args = parser.parse_args()

    print("Loading trial-level EEG features...")

    df = pd.read_csv(args.features)

    print(f"Trials: {len(df):,}")
    print(
        f"Participants: "
        f"{df['participant'].nunique()}"
    )

    train_df, val_df, test_df = participant_split(df)

    print("\nParticipant split:")

    for name, split in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test (untouched)", test_df),
    ]:
        participants = sorted(
            split["participant"].unique()
        )

        print(
            f"{name}: "
            f"{len(participants)} participants / "
            f"{len(split)} trials"
        )

        print(
            f"  IDs: {participants}"
        )

    valence_model, valence_metrics = train_target(
        train_df,
        val_df,
        target="valence_class",
        display_name="MODEL 1: NEGATIVE vs POSITIVE VALENCE",
    )

    arousal_model, arousal_metrics = train_target(
        train_df,
        val_df,
        target="arousal_class",
        display_name="MODEL 2: LOW vs HIGH AROUSAL",
    )

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        valence_model,
        "models/valence_svm.joblib",
    )

    joblib.dump(
        arousal_model,
        "models/arousal_svm.joblib",
    )

    results = {
        "valence": valence_metrics,
        "arousal": arousal_metrics,
    }

    Path(
        "models/validation_metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(
        "Valence balanced accuracy:",
        f"{valence_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Arousal balanced accuracy:",
        f"{arousal_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "\nHeld-out test participants were NOT evaluated."
    )


if __name__ == "__main__":
    main()
