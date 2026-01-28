"""Unified entry point for ProbCut parameter training.

This script trains ProbCut models and outputs Rust code for the parameters.

Usage:
    uv run scripts/probcut.py data.csv --variant midgame
    uv run scripts/probcut.py data.csv --variant endgame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probcut.probcut_common import add_training_cli_arguments, load_dataframe
from probcut.probcut_midgame import (
    REQUIRED_COLUMNS as MIDGAME_COLUMNS,
    emit_rust_params as emit_midgame_params,
    train_all_plies,
)
from probcut.probcut_endgame import (
    REQUIRED_COLUMNS as ENDGAME_COLUMNS,
    emit_rust_params as emit_endgame_params,
    train_endgame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ProbCut models and emit Rust parameters."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--variant",
        choices=("midgame", "endgame"),
        default="midgame",
        help="Training variant (default: midgame)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=60,
        help="Number of plies to emit in Rust array (midgame only)",
    )
    add_training_cli_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)

    required_columns = MIDGAME_COLUMNS if args.variant == "midgame" else ENDGAME_COLUMNS

    try:
        df = load_dataframe(csv_path, required_columns)
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
        if args.variant == "midgame":
            models = train_all_plies(
                df,
                n_splits=n_splits,
                alpha=alpha,
                seed=seed,
                epsilon=epsilon,
            )
            emit_midgame_params(models, max_ply=args.max_ply)
        else:
            mean_model, std_model = train_endgame(
                df,
                n_splits=n_splits,
                alpha=alpha,
                seed=seed,
                epsilon=epsilon,
            )
            emit_endgame_params(mean_model, std_model)
    except Exception as exc:
        print(f"Error fitting models: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
