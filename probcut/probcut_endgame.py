"""Endgame ProbCut model training (single set of parameters)."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

from probcut.probcut_common import fit_probcut_models, format_float

REQUIRED_COLUMNS = {"shallow_depth", "deep_depth", "diff"}


def train_endgame(
    df: pd.DataFrame,
    n_splits: int,
    alpha: float,
    seed: int,
    epsilon: float,
) -> Tuple[LinearRegression, LinearRegression]:
    """Train mean and std models for endgame."""
    return fit_probcut_models(
        df,
        n_splits=n_splits,
        alpha=alpha,
        seed=seed,
        epsilon=epsilon,
    )


def emit_rust_params(
    mean_model: LinearRegression,
    std_model: LinearRegression,
) -> None:
    """Emit a Rust constant for endgame parameters."""
    print("const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {")
    print(f"    mean_intercept: {format_float(mean_model.intercept_)},")
    print(f"    mean_coef_shallow: {format_float(mean_model.coef_[0])},")
    print(f"    mean_coef_deep: {format_float(mean_model.coef_[1])},")
    print(f"    std_intercept: {format_float(std_model.intercept_)},")
    print(f"    std_coef_shallow: {format_float(std_model.coef_[0])},")
    print(f"    std_coef_deep: {format_float(std_model.coef_[1])},")
    print("};")
