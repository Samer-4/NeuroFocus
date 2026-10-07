from pathlib import Path
import argparse

from neurofocus.data import load_deap
from neurofocus.features import build_feature_table
from neurofocus.labels import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Extract EEG features from DEAP."
    )

    parser.add_argument(
        "--data-dir",
        default="data_preprocessed_python",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--feature-type",
        choices=["bandpower", "de"],
        default="bandpower",
    )

    args = parser.parse_args()

    if args.output is None:
        if args.feature_type == "de":
            args.output = "data/processed/eeg_de_features.csv"
        else:
            args.output = "data/processed/eeg_features.csv"

    print("Loading DEAP data...")

    eeg, labels = load_deap(args.data_dir)

    print(f"EEG shape: {eeg.shape}")
    print(f"Labels shape: {labels.shape}")

    print(
        f"\nExtracting EEG {args.feature_type} features..."
    )

    features = build_feature_table(
        eeg,
        labels,
        threshold=args.threshold,
        feature_type=args.feature_type,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path}")
    print(f"Total EEG trials: {len(features):,}")

    metadata_columns = {
        "participant",
        "trial",
        "valence",
        "arousal",
        "valence_class",
        "arousal_class",
        "affective_state",
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in metadata_columns
    ]

    print(
        f"EEG feature columns: {len(feature_columns)}"
    )

    print("\nClass distribution:")

    counts = (
        features["affective_state"]
        .value_counts()
        .sort_index()
    )

    for class_id, count in counts.items():
        percentage = count / len(features) * 100

        print(
            f"{class_id} - {CLASS_NAMES[class_id]}: "
            f"{count:,} trials ({percentage:.1f}%)"
        )


if __name__ == "__main__":
    main()
