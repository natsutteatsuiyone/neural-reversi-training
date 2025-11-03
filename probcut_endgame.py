"""
Train ProbCut endgame parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

import probcut_common

build_features = probcut_common.build_features
oof_predictions_linreg = probcut_common.oof_predictions_linreg
smoothed_local_mae = probcut_common.smoothed_local_mae
fit_probcut_models = probcut_common.fit_probcut_models
format_float = probcut_common.format_float
load_dataframe = probcut_common.load_dataframe

REQUIRED_COLUMNS = {"shallow_depth", "deep_depth", "diff"}


def fit_models(
    df: pd.DataFrame,
    n_splits: int,
    alpha: float,
    seed: int,
    epsilon: float,
) -> Tuple[LinearRegression, LinearRegression]:
    return fit_probcut_models(
        df,
        n_splits=n_splits,
        alpha=alpha,
        seed=seed,
        epsilon=epsilon,
    )


def emit_rust_params(mean_model: LinearRegression, std_model: LinearRegression) -> None:
    # Match build_features ordering: [sd, dd, 1 / (1 + log1p(max(dd - sd, 0))), ((dd - sd) % 2 == 0)]
    print("const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {")
    print(f"    mean_intercept: {format_float(mean_model.intercept_)},")
    print(f"    mean_coef_shallow: {format_float(mean_model.coef_[0])},")
    print(f"    mean_coef_deep: {format_float(mean_model.coef_[1])},")
    print(f"    std_intercept: {format_float(std_model.intercept_)},")
    print(f"    std_coef_shallow: {format_float(std_model.coef_[0])},")
    print(f"    std_coef_deep: {format_float(std_model.coef_[1])},")
    print("};")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ProbCut endgame model from CSV data.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    probcut_common.add_training_cli_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv_path)
    try:
        df = load_dataframe(csv_path, REQUIRED_COLUMNS)
    except FileNotFoundError:
        print(f"Error: File {csv_path} does not exist.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error loading/validating data: {exc}", file=sys.stderr)
        return 1

    n_splits = max(2, args.folds)
    alpha = max(0.0, args.alpha)
    epsilon = args.epsilon
    seed = args.seed

    try:
        mean_model, std_model = fit_models(
            df,
            n_splits=n_splits,
            alpha=alpha,
            seed=seed,
            epsilon=epsilon,
        )
    except Exception as exc:
        print(f"Error fitting models: {exc}", file=sys.stderr)
        return 1

    emit_rust_params(mean_model, std_model)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
