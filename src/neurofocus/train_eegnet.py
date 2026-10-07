from pathlib import Path
import argparse
import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neurofocus.eegnet_data import prepare_eegnet_data
from neurofocus.eegnet_model import EEGNet


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def participant_split(X, y, groups):
    """
    Same participant-level split strategy used
    by our classical ML experiments.
    """

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, temp_idx = next(
        outer.split(X, y, groups)
    )

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_relative_idx, test_relative_idx = next(
        inner.split(
            X[temp_idx],
            y[temp_idx],
            groups[temp_idx],
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    return train_idx, val_idx, test_idx


def normalize_from_train(X_train, X_val):
    """
    Normalize using ONLY training-set statistics.

    This prevents validation information from
    leaking into training.
    """

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True)

    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, X_val, mean, std


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            pred = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                y_batch.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        targets,
        predictions,
    )

    return accuracy, balanced_accuracy


def train(
    target="valence",
    epochs=30,
    patience=5,
    batch_size=16,
    learning_rate=0.001,
    limit=None,
):
    set_seed()

    print(f"Target: {target}")

    X, y_valence, y_arousal, groups = prepare_eegnet_data(
        "data_preprocessed_python"
    )

    if target == "valence":
        y = y_valence
    elif target == "arousal":
        y = y_arousal
    else:
        raise ValueError(
            "target must be 'valence' or 'arousal'"
        )

    train_idx, val_idx, test_idx = participant_split(
        X,
        y,
        groups,
    )

    print(
        f"Train: {len(train_idx)} trials / "
        f"{len(np.unique(groups[train_idx]))} participants"
    )

    print(
        f"Validation: {len(val_idx)} trials / "
        f"{len(np.unique(groups[val_idx]))} participants"
    )

    print(
        f"Test: {len(test_idx)} trials / "
        f"{len(np.unique(groups[test_idx]))} participants "
        "(UNTOUCHED)"
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    # Used only for a quick local sanity test.
    if limit is not None:
        X_train = X_train[:limit]
        y_train = y_train[:limit]

        X_val = X_val[:limit]
        y_val = y_val[:limit]

        print(
            f"\nSANITY TEST MODE: "
            f"limited to {limit} examples per split."
        )

    X_train, X_val, mean, std = normalize_from_train(
        X_train,
        X_val,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        batch_size,
        shuffle=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = EEGNet().to(device)

    class_counts = np.bincount(y_train)

    class_weights = (
        len(y_train)
        / (2.0 * class_counts)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_balanced_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * X_batch.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_accuracy, val_balanced_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Balanced Acc: "
            f"{val_balanced_accuracy:.4f}"
        )

        if (
            val_balanced_accuracy
            > best_balanced_accuracy
        ):
            best_balanced_accuracy = (
                val_balanced_accuracy
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )
            break

    model.load_state_dict(best_state)

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        f"models/eegnet_{target}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "target": target,
            "best_validation_balanced_accuracy":
                best_balanced_accuracy,
        },
        output_path,
    )

    print(
        f"\nBest validation balanced accuracy: "
        f"{best_balanced_accuracy:.4f}"
    )

    print(f"Saved: {output_path}")

    print(
        "\nFinal test participants were NOT evaluated."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["valence", "arousal"],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    train(
        target=args.target,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
