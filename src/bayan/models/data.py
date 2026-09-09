"""Lab 3A: dataset construction and leakage-safe grouped split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset():
    df = pd.read_csv(DATA_PATH)

    groups = df["citizen_group_id"]

    train_idx, temp_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=42,
        ).split(df, groups=groups)
    )

    train = df.iloc[train_idx].reset_index(drop=True)
    temp = df.iloc[temp_idx].reset_index(drop=True)

    valid_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=42,
        ).split(temp, groups=temp["citizen_group_id"])
    )

    validation = temp.iloc[valid_idx].reset_index(drop=True)
    test = temp.iloc[test_idx].reset_index(drop=True)

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }
