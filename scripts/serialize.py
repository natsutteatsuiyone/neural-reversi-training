"""Unified entry point for neural network model serialization.

This script provides a single CLI for serializing all model variants
(large, small, WASM) into compressed binary format suitable for deployment.

Usage:
    uv run scripts/serialize.py --checkpoint ./ckpt/model.ckpt
    uv run scripts/serialize.py --checkpoint ./ckpt/model.ckpt --model_variant small
    uv run scripts/serialize.py --checkpoint ./ckpt/model.ckpt --model_variant wasm
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import model_lg
from models import model_sm
from models import model_wasm
import version
from models.serialization.serialize_common import (
    DEFAULT_COMPRESSION_LEVEL,
    load_checkpoint_into_model,
    write_compressed_output,
)
from models.serialization.serialize_lg import LargeNNWriter
from models.serialization.serialize_sm import SmallNNWriter
from models.serialization.serialize_wasm import WasmNNWriter


# Model variant configuration
MODEL_CONFIG = {
    "large": {
        "lit_model_cls": model_lg.LitReversiModel,
        "writer_cls": LargeNNWriter,
        "default_filename": f"eval-{version.get_version_hash()}.zst",
    },
    "small": {
        "lit_model_cls": model_sm.LitReversiSmallModel,
        "writer_cls": SmallNNWriter,
        "default_filename": f"eval_sm-{version.get_version_hash()}.zst",
    },
    "wasm": {
        "lit_model_cls": model_wasm.LitReversiWasmModel,
        "writer_cls": WasmNNWriter,
        "default_filename": f"eval_wasm-{version.get_version_hash()}.zst",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize a Reversi model to compressed binary format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--model_variant",
        choices=("large", "small", "wasm"),
        default="large",
        help="Model architecture to serialize (default: large)",
    )
    parser.add_argument(
        "--cl",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help=f"Compression level (default: {DEFAULT_COMPRESSION_LEVEL})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for serialized file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output filename (default: variant-specific name)",
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram display during serialization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MODEL_CONFIG[args.model_variant]

    # Use variant-specific default filename if not specified
    filename = args.filename or config["default_filename"]

    # Create parser for error reporting in load_checkpoint_into_model
    parser = argparse.ArgumentParser()

    lit_model = config["lit_model_cls"]()
    load_checkpoint_into_model(args.checkpoint, lit_model, parser)

    writer = config["writer_cls"](lit_model.model, show_hist=not args.no_hist)
    output_path = Path(args.output_dir) / filename
    write_compressed_output(writer.get_buffer(), output_path, args.cl)


if __name__ == "__main__":
    main()
