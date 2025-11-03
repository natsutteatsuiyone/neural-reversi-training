"""Shared helpers for training ProbCut models."""

from __future__ import annotations

import argparse
from collections.abc import Collection
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def build_features(df: pd.DataFrame) -> np.ndarray:
    shallow = df["shallow_depth"].to_numpy(dtype=np.int64)
    deep = df["deep_depth"].to_numpy(dtype=np.int64)
    sd = shallow.astype(float)
    dd = deep.astype(float)
    return np.column_stack([
        sd,
        dd,
    ])


def oof_predictions_linreg(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Compute out-of-fold linear regression predictions to avoid leakage."""
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two samples to compute OOF predictions.")
    splits = min(max(2, n_splits), n_samples)
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    y_oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(X):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        y_oof[va] = model.predict(X[va])
    return y_oof


def smoothed_local_mae(sd: np.ndarray, dd: np.ndarray, abs_resid: np.ndarray, alpha: float) -> np.ndarray:
    """Compute a smoothed local MAE for each (shallow_depth, deep_depth) bucket."""
    g = pd.DataFrame({"sd": sd, "dd": dd, "abs": abs_resid})
    grp = g.groupby(["sd", "dd"])
    mean_by_key = grp["abs"].transform("mean")
    count_by_key = grp["abs"].transform("count").astype(float)
    global_abs = g["abs"].mean()
    smoothed = (mean_by_key * count_by_key + alpha * global_abs) / (count_by_key + alpha)
    return smoothed.to_numpy()


def fit_probcut_models(
    df: pd.DataFrame,
    n_splits: int,
    alpha: float,
    seed: int,
    epsilon: float,
) -> Tuple[LinearRegression, LinearRegression]:
    # Prepare feature matrix
    X = build_features(df)
    y = df["diff"].to_numpy(dtype=float)

    # Step 1: compute residuals with OOF predictions
    y_oof = oof_predictions_linreg(X, y, n_splits=n_splits, seed=seed)
    resid = y - y_oof

    sd = df["shallow_depth"].to_numpy(dtype=float)
    dd = df["deep_depth"].to_numpy(dtype=float)
    # Step 2: estimate local MAE and derive sigma under a normal assumption
    local_mae = smoothed_local_mae(sd, dd, np.abs(resid), alpha=alpha)
    std_targets = local_mae * np.sqrt(np.pi / 2.0)  # E|eps| = sigma * sqrt(2/pi)
    log_std_targets = np.log(std_targets + epsilon)

    # Step 3: predict log sigma out-of-fold and reuse it as weights
    log_sigma_oof = oof_predictions_linreg(X, log_std_targets, n_splits=n_splits, seed=seed)
    sigma_oof = np.exp(log_sigma_oof)
    var_pred = np.maximum(sigma_oof ** 2, 1e-12)
    w = 1.0 / var_pred
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    # Step 4: refit the mean model with WLS using the OOF variance
    mean_model = LinearRegression()
    mean_model.fit(Xw, yw)

    # Step 5: refit the final log-sigma model on all data for export
    std_model = LinearRegression()
    std_model.fit(X, log_std_targets)

    return mean_model, std_model


def format_float(x: float) -> str:
    return f"{float(x):.10f}"


def add_training_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI flags shared by the ProbCut training scripts."""
    parser.add_argument("--folds", type=int, default=5, help="K-fold for OOF residuals.")
    parser.add_argument("--alpha", type=float, default=5.0, help="Smoothing strength for local MAE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffling.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Stability constant for log std.")


def validate_dataframe(df: pd.DataFrame, required_columns: Collection[str]) -> None:
    """Validate that the dataframe matches the expectations of the training pipelines."""
    required_set = set(required_columns)
    missing_cols = required_set - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")
    if df.empty:
        raise ValueError("Input CSV is empty.")
    if df[list(required_set)].isnull().any().any():
        raise ValueError("Input data contains NaN values.")
    if (df["shallow_depth"] < 0).any() or (df["deep_depth"] < 0).any():
        raise ValueError("Depth values must be non-negative.")
    if (df["deep_depth"] < df["shallow_depth"]).any():
        raise ValueError("deep_depth must be >= shallow_depth.")


def load_dataframe(csv_path: Path | str, required_columns: Collection[str]) -> pd.DataFrame:
    """Load a CSV file and run shared validation checks."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    validate_dataframe(df, required_columns=required_columns)
    return df


__all__ = [
    "build_features",
    "oof_predictions_linreg",
    "smoothed_local_mae",
    "fit_probcut_models",
    "format_float",
    "add_training_cli_arguments",
    "validate_dataframe",
    "load_dataframe",
]
