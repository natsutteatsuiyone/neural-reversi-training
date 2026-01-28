"""Midgame ProbCut model training (per-ply parameters)."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

from probcut.probcut_common import fit_probcut_models, format_float

REQUIRED_COLUMNS = {"ply", "shallow_depth", "deep_depth", "diff"}


def fit_models_for_ply(
    group: pd.DataFrame,
    n_splits: int,
    alpha: float,
    seed: int,
    epsilon: float,
) -> Tuple[LinearRegression, LinearRegression]:
    """Fit mean and std models for a single ply group."""
    return fit_probcut_models(
        group,
        n_splits=n_splits,
        alpha=alpha,
        seed=seed,
        epsilon=epsilon,
    )


def train_all_plies(
    df: pd.DataFrame,
    n_splits: int,
    alpha: float,
    seed: int,
    epsilon: float,
) -> Dict[int, Tuple[LinearRegression, LinearRegression]]:
    """Train models for all plies in the dataframe."""
    models: Dict[int, Tuple[LinearRegression, LinearRegression]] = {}

    for ply, group in df.groupby("ply"):
        group = group.reset_index(drop=True)
        try:
            mean_model, std_model = fit_models_for_ply(
                group,
                n_splits=n_splits,
                alpha=alpha,
                seed=seed,
                epsilon=epsilon,
            )
            models[int(ply)] = (mean_model, std_model)
        except Exception as e:
            import sys

            print(f"[WARN] Skipped ply {ply} due to error: {e}", file=sys.stderr)

    return models


def emit_rust_params(
    models: Dict[int, Tuple[LinearRegression, LinearRegression]],
    max_ply: int,
) -> None:
    """Emit a Rust constant array, falling back to the closest trained ply when needed."""
    learned = sorted(models.keys())
    if not learned:
        raise RuntimeError("No ply was learned. Check your CSV contents.")

    def nearest_ply(p: int) -> int:
        return min(learned, key=lambda q: abs(q - p))

    print(f"const PROBCUT_PARAMS: [ProbcutParams; {max_ply}] = [")
    for ply in range(max_ply):
        use_ply = ply if ply in models else nearest_ply(ply)
        mean_model, std_model = models[use_ply]

        print("    ProbcutParams {")
        print(f"        mean_intercept: {format_float(mean_model.intercept_)},")
        print(f"        mean_coef_shallow: {format_float(mean_model.coef_[0])},")
        print(f"        mean_coef_deep: {format_float(mean_model.coef_[1])},")
        print(f"        std_intercept: {format_float(std_model.intercept_)},")
        print(f"        std_coef_shallow: {format_float(std_model.coef_[0])},")
        print(f"        std_coef_deep: {format_float(std_model.coef_[1])},")
        print("    },")
    print("];")
