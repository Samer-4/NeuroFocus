import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


NON_FEATURE_COLUMNS = [
    "participant",
    "trial",
    "valence",
    "arousal",
    "valence_class",
    "arousal_class",
    "affective_state",
]


def participant_split(df, random_state=42):
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


def get_features(df):
    return df.drop(columns=NON_FEATURE_COLUMNS)


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                    max_iter=10000,
                ),
            ),
        ]),

        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def compare_target(
    train_df,
    val_df,
    target,
    display_name,
):
    print("\n" + "=" * 65)
    print(display_name)
    print("=" * 65)

    X_train = get_features(train_df)
    X_val = get_features(val_df)

    y_train = train_df[target]
    y_val = val_df[target]

    results = []

    for name, model in get_models().items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val,
            predictions,
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        })

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Balanced accuracy: {balanced_accuracy:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "balanced_accuracy",
        ascending=False,
    )

    print("\nRanking:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "balanced_accuracy": "{:.4f}".format,
            },
        )
    )

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default="data/processed/eeg_features.csv",
        help="Path to extracted EEG feature CSV.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.features)

    print(f"Features: {args.features}")

    train_df, val_df, test_df = participant_split(df)

    print("Dataset:")
    print(f"Train: {len(train_df)} trials")
    print(f"Validation: {len(val_df)} trials")
    print(f"Test: {len(test_df)} trials (untouched)")

    valence_results = compare_target(
        train_df,
        val_df,
        target="valence_class",
        display_name="VALENCE: NEGATIVE vs POSITIVE",
    )

    arousal_results = compare_target(
        train_df,
        val_df,
        target="arousal_class",
        display_name="AROUSAL: LOW vs HIGH",
    )

    print("\n" + "=" * 65)
    print("BEST VALIDATION MODELS")
    print("=" * 65)

    print(
        "\nValence:",
        valence_results.iloc[0]["model"],
        "-",
        f"{valence_results.iloc[0]['balanced_accuracy']:.4f}",
    )

    print(
        "Arousal:",
        arousal_results.iloc[0]["model"],
        "-",
        f"{arousal_results.iloc[0]['balanced_accuracy']:.4f}",
    )

    print(
        "\nFinal test participants have NOT been evaluated."
    )


if __name__ == "__main__":
    main()
