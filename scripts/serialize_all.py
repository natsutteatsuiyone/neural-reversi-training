"""Batch serialization of multiple checkpoint files.

This script provides a CLI for serializing all checkpoint files in a directory.

Usage:
    uv run scripts/serialize_all.py --ckpt-dir ./ckpt --output-dir ./weights
    uv run scripts/serialize_all.py --ckpt-dir ./ckpt/small --output-dir ./weights --model_variant small
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models import model_lg
from models import model_sm
from models import model_wasm
from models.serialization.serialize_common import (
    DEFAULT_COMPRESSION_LEVEL,
    normalize_state_dict_keys,
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
    },
    "small": {
        "lit_model_cls": model_sm.LitReversiSmallModel,
        "writer_cls": SmallNNWriter,
    },
    "wasm": {
        "lit_model_cls": model_wasm.LitReversiWasmModel,
        "writer_cls": WasmNNWriter,
    },
}


def serialize_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    compression_level: int,
    model_variant: str,
) -> tuple[Path, bool, str]:
    """Serialize a single checkpoint file."""
    output_filename = f"{checkpoint_path.stem}.zst"
    output_path = output_dir / output_filename

    config = MODEL_CONFIG[model_variant]

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        lit_model = config["lit_model_cls"]()

        if isinstance(ckpt, dict) and "current_model_state" in ckpt:
            base_state = normalize_state_dict_keys(ckpt["current_model_state"])
            lit_model.load_state_dict(base_state, strict=False)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_source = ckpt["state_dict"]
        else:
            state_source = ckpt

        state = normalize_state_dict_keys(state_source)
        lit_model.load_state_dict(state, strict=False)
        lit_model.eval()

        writer = config["writer_cls"](lit_model.model, show_hist=False)
        write_compressed_output(writer.get_buffer(), output_path, compression_level)

        return checkpoint_path, True, f"Serialized to {output_path}"

    except Exception as exc:
        return checkpoint_path, False, f"Error: {exc}"


def get_checkpoint_files(ckpt_dir: Path) -> list[Path]:
    """Get all checkpoint files from the specified directory."""
    return list(ckpt_dir.glob("*.ckpt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize all checkpoint files in a folder"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for serialized files",
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
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = get_checkpoint_files(ckpt_dir)

    if not checkpoint_files:
        print(f"No checkpoint files found in '{ckpt_dir}'.")
        return

    print(f"Processing {len(checkpoint_files)} checkpoint files...")

    def process_checkpoint(cp: Path) -> tuple[Path, bool, str]:
        return serialize_checkpoint(cp, output_dir, args.cl, args.model_variant)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        results = list(executor.map(process_checkpoint, checkpoint_files))

    success_count = 0
    fail_count = 0

    for checkpoint_path, success, output in results:
        if success:
            success_count += 1
            print(f"Success: {checkpoint_path}")
        else:
            fail_count += 1
            print(f"Failed: {checkpoint_path}")
            print(output)

    print(
        f"Processing complete: success={success_count}, "
        f"failed={fail_count}, total={len(checkpoint_files)}"
    )


if __name__ == "__main__":
    main()
