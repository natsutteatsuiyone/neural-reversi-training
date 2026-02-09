"""Batch serialization of multiple checkpoint files.

This script provides a CLI for serializing all checkpoint files in a directory.
Each checkpoint is processed in a separate subprocess to guarantee full memory
release between files.

Usage:
    uv run scripts/serialize_all.py --ckpt-dir ./ckpt --output-dir ./weights
    uv run scripts/serialize_all.py --ckpt-dir ./ckpt/small --output-dir ./weights --model_variant small
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.serialization.serialize_common import DEFAULT_COMPRESSION_LEVEL


def _serialize_one(
    checkpoint_path: Path,
    output_path: Path,
    compression_level: int,
    model_variant: str,
) -> tuple[bool, str]:
    """Serialize a single checkpoint (runs inside a child process)."""
    import torch

    from models import model_lg, model_sm, model_wasm
    from models.serialization.serialize_common import (
        normalize_state_dict_keys,
        write_compressed_output,
    )
    from models.serialization.serialize_lg import LargeNNWriter
    from models.serialization.serialize_sm import SmallNNWriter
    from models.serialization.serialize_wasm import WasmNNWriter

    model_config = {
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

    config = model_config[model_variant]

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    lit_model = config["lit_model_cls"]()

    if isinstance(ckpt, dict) and "current_model_state" in ckpt:
        base_state = normalize_state_dict_keys(ckpt["current_model_state"])
        lit_model.load_state_dict(base_state, strict=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = normalize_state_dict_keys(ckpt["state_dict"])
    else:
        state = normalize_state_dict_keys(ckpt)

    lit_model.load_state_dict(state, strict=False)
    lit_model.eval()

    writer = config["writer_cls"](lit_model.model, show_hist=False)
    write_compressed_output(writer.get_buffer(), output_path, compression_level)

    return True, f"Serialized to {output_path}"


def serialize_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    compression_level: int,
    model_variant: str,
) -> tuple[Path, bool, str]:
    """Serialize a single checkpoint file in a child process."""
    output_path = output_dir / f"{checkpoint_path.stem}.zst"

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        try:
            success, msg = pool.apply(
                _serialize_one,
                (checkpoint_path, output_path, compression_level, model_variant),
            )
            return checkpoint_path, success, msg
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

    success_count = 0
    fail_count = 0

    for cp in checkpoint_files:
        checkpoint_path, success, output = serialize_checkpoint(
            cp, output_dir, args.cl, args.model_variant
        )
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
