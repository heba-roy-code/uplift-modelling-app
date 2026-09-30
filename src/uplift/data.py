"""Loading, preparing and splitting the Hillstrom email dataset.

The core project frames the problem as a *binary* treatment ("received any email"
vs "no email") and a binary outcome ("conversion" by default). The three-arm
``segment`` column is kept in the raw frame for the multi-treatment extension.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hillstrom.csv.gz"
DATA_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"

CONTROL_ARM = "No E-Mail"
OUTCOMES = ("visit", "conversion", "spend")
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]


def load_hillstrom(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return the raw Hillstrom data (64,000 rows), downloading it if not on disk."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def make_uplift_frame(
    df: pd.DataFrame, outcome: str = "conversion"
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the raw data into features ``X``, binary treatment ``w`` and outcome ``y``.

    ``w`` is 1 for either email arm and 0 for the control group.
    ``history_segment`` is dropped because it is just a binned copy of ``history``.
    Categorical features are one-hot encoded (with the first level dropped).
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    X = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=int,
    )
    w = (df["segment"] != CONTROL_ARM).astype(int).rename("treatment")
    y = df[outcome].rename(outcome)
    return X, y, w


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Train/test split, stratified on treatment *and* outcome.

    Stratifying on the treatment/outcome combination keeps the treatment share and the
    (rare) conversion rate the same in both halves. Returns
    ``X_train, X_test, y_train, y_test, w_train, w_test``.
    """
    strata = w.astype(str) + "_" + (y > 0).astype(str)
    return train_test_split(X, y, w, test_size=test_size, random_state=seed, stratify=strata)
