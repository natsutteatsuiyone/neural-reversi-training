import os
import argparse
import glob
import subprocess
import concurrent.futures
from typing import List, Tuple


def get_checkpoint_files(ckpt_dir: str) -> List[str]:
    """Get all checkpoint files from the specified directory"""
    return glob.glob(os.path.join(ckpt_dir, "*.ckpt"))


def serialize_checkpoint(args: Tuple[str, str, int]) -> Tuple[str, bool, str]:
    """Serialize a single checkpoint file"""
    checkpoint_path, output_dir, cl = args

    filename = os.path.basename(checkpoint_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}.zst")

    cmd = [
        "uv", "run", "serialize.py",
        "--checkpoint", checkpoint_path,
        "--cl", str(cl),
        "--output", output_path,
        "--no-hist"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return checkpoint_path, True, result.stdout
    except subprocess.CalledProcessError as e:
        return checkpoint_path, False, f"Error: {e.stderr}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialize all checkpoint files in a folder")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for serialized files"
    )
    parser.add_argument(
        "--cl",
        type=int,
        default=7,
        help="Compression level (default: 7)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_files = get_checkpoint_files(args.ckpt_dir)

    if not checkpoint_files:
        print(f"No checkpoint files found in the specified directory '{args.ckpt_dir}'.")
        return

    print(f"Processing {len(checkpoint_files)} checkpoint files...")

    process_args = [(cp, args.output_dir, args.cl) for cp in checkpoint_files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(serialize_checkpoint, process_args))

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

    print(f"Processing complete: success={success_count}, failed={fail_count}, total={len(checkpoint_files)}")


if __name__ == "__main__":
    main()
