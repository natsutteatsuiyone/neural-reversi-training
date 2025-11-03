"""
Train ProbCut parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression

import probcut_common

build_features = probcut_common.build_features
oof_predictions_linreg = probcut_common.oof_predictions_linreg
smoothed_local_mae = probcut_common.smoothed_local_mae
fit_probcut_models = probcut_common.fit_probcut_models
format_float = probcut_common.format_float
load_dataframe = probcut_common.load_dataframe


REQUIRED_COLUMNS = {"ply", "shallow_depth", "deep_depth", "diff"}


def fit_models_for_ply(group: pd.DataFrame, n_splits: int, alpha: float, seed: int, epsilon: float) -> Tuple[LinearRegression, LinearRegression]:
    return fit_probcut_models(
        group,
        n_splits=n_splits,
        alpha=alpha,
        seed=seed,
        epsilon=epsilon,
    )


def emit_rust_params(models: Dict[int, Tuple[LinearRegression, LinearRegression]], max_ply: int) -> None:
    """Emit a Rust constant array, falling back to the closest trained ply when needed."""
    learned = sorted(models.keys())
    if not learned:
        raise RuntimeError("No ply was learned. Check your CSV contents.")

    def nearest_ply(p: int) -> int:
        return min(learned, key=lambda q: abs(q - p))

    print("const PROBCUT_PARAMS: [ProbcutParams; {}] = [".format(max_ply))
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ProbCut models from CSV data.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--max-ply", type=int, default=60, help="Number of plies to emit in the Rust array.")
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

    models: Dict[int, Tuple[LinearRegression, LinearRegression]] = {}
    n_splits = max(2, args.folds)
    alpha = max(0.0, args.alpha)
    epsilon = args.epsilon
    seed = args.seed

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
            print(f"[WARN] Skipped ply {ply} due to error: {e}", file=sys.stderr)

    emit_rust_params(models, max_ply=args.max_ply)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
