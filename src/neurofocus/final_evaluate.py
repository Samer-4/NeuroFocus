from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurofocus.compare_models import (
    participant_split,
    get_features,
)


DATA_PATH = "data/processed/eeg_features.csv"
MODEL_DIR = Path("models")


def evaluate_model(
    model,
    train_df,
    val_df,
    test_df,
    target,
    name,
    output_path,
):
    """
    Train the frozen model choice on train + validation data,
    then evaluate exactly once on the untouched test participants.
    """

    development_df = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    X_dev = get_features(development_df)
    y_dev = development_df[target]

    X_test = get_features(test_df)
    y_test = test_df[target]

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Training on {len(development_df)} development trials..."
    )

    model.fit(X_dev, y_dev)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    print(f"\nFINAL TEST ACCURACY: {accuracy:.4f}")
    print(
        "FINAL TEST BALANCED ACCURACY: "
        f"{balanced_accuracy:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    joblib.dump(
        model,
        output_path,
    )

    print(f"Saved final model: {output_path}")

    return accuracy, balanced_accuracy


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(DATA_PATH)

    train_df, val_df, test_df = participant_split(df)

    print("FINAL FROZEN EVALUATION")
    print("-----------------------")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print(
        "Test participants:",
        sorted(test_df["participant"].unique()),
    )

    valence_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    arousal_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
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

    valence_acc, valence_bal = evaluate_model(
        valence_model,
        train_df,
        val_df,
        test_df,
        target="valence_class",
        name="VALENCE — LOGISTIC REGRESSION",
        output_path=MODEL_DIR / "valence_model.joblib",
    )

    arousal_acc, arousal_bal = evaluate_model(
        arousal_model,
        train_df,
        val_df,
        test_df,
        target="arousal_class",
        name="AROUSAL — RBF SVM",
        output_path=MODEL_DIR / "arousal_model.joblib",
    )

    print("\n" + "=" * 65)
    print("FINAL NEUROFOCUS RESULTS")
    print("=" * 65)

    print(
        f"Valence | Accuracy: {valence_acc:.4f} | "
        f"Balanced: {valence_bal:.4f}"
    )

    print(
        f"Arousal | Accuracy: {arousal_acc:.4f} | "
        f"Balanced: {arousal_bal:.4f}"
    )

    print("\nModel selection is now FINAL.")


if __name__ == "__main__":
    main()
