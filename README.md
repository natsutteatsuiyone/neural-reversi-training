# Neural Reversi Training

## Prerequisites

- Python 3.14+
- CUDA-compatible GPU (compute capability 7.0+)
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

### 1. Clone

```bash
git clone https://github.com/natsutteatsuiyone/neural-reversi-training.git
cd neural-reversi-training
```

### 2. Environment Setup

```bash
uv venv
uv sync
```

### 3. Build CUDA Extension

Build the custom sparse linear CUDA extension:

```bash
uv run python sparse_linear/setup.py build_ext --inplace
```

### 4. Build Feature Dataset Extension (Optional, Recommended)

Build the C++ extension for faster data loading (~1.6x speedup):

```bash
uv run python feature_dataset/setup.py build_ext --inplace
```

The training script will automatically use the C++ implementation if available.

### 5. Optional: NVIDIA Apex (Recommended for Performance)

For optimal training performance, install NVIDIA Apex:

```bash
# Install NVIDIA Apex for optimized training
git clone https://github.com/NVIDIA/apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -v --no-build-isolation ./apex/
```

## Training

### Basic Training

```bash
uv run scripts/train.py --train_data ./data/train --val_data ./data/val
```

### Train the WASM Model

```bash
uv run scripts/train.py --train_data ./data/train --val_data ./data/val --model_variant wasm
```

### Resume Training from Checkpoint

```bash
uv run scripts/train.py --train_data ./data/train --val_data ./data/val --resume_from_checkpoint ./ckpt/reversi-10-0.0123.ckpt
```

### Command Line Arguments

**Required:**
- `--train_data`: Path to training data (*.zst)
- `--val_data`: Path to validation data (*.zst)

**Optional:**
- `--batch_size`: Training batch size (default: 16384)
- `--num_workers`: Data loading workers (default: 4)
- `--epochs`: Maximum epochs (default: 300)
- `--seed`: Random seed (default: 42)
- `--resume_from_checkpoint`: Checkpoint path to resume training
- `--resume_from_weights`: Weight file path to resume training
- `--lr`: Learning rate (default: 0.001)
- `--shuffle`: Shuffle the dataset (default: True)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--file_usage_ratio`: Ratio of files to use per epoch (default: 1.0)
- `--model_variant`: Model architecture to train: `large` (default), `small`, or `wasm` (the previous `--small` flag maps to `--model_variant small`)
- `--use_python_dataset`: Force use of Python dataset implementation instead of C++

### Output

- Checkpoints saved in `ckpt/` (or `ckpt/<model_variant>/` for non-large models)
- TensorBoard logs saved in `tb_logs/`
- Best models saved based on validation loss
- Last checkpoint automatically saved as `last.ckpt`

## Creating Weight Files

Convert trained checkpoint to a weight file:

```bash
# Large model (default)
uv run scripts/serialize.py --checkpoint ./ckpt/reversi-best.ckpt

# Small model
uv run scripts/serialize.py --checkpoint ./ckpt/small/reversi-best.ckpt --model_variant small

# WASM model
uv run scripts/serialize.py --checkpoint ./ckpt/wasm/reversi-best.ckpt --model_variant wasm
```

### Serialization Arguments

- `--checkpoint`: (Required) Checkpoint file path
- `--model_variant`: Model architecture: `large` (default), `small`, or `wasm`
- `--output_dir`: Output directory path (default: current directory)
- `--filename`: Output filename (default: auto-generated based on model variant and version)
- `--cl`: Compression level (default: 1)
- `--no-hist`: Disable histogram display

**Example with custom output:**
```bash
uv run scripts/serialize.py --checkpoint ./ckpt/reversi-best.ckpt --output_dir ./weights --filename my_model.zst
```

The output is a zstandard-compressed binary, compatible with the Neural Reversi engine.

## Training ProbCut Parameters

Train ProbCut models and generate Rust parameters:

```bash
# Midgame parameters (per-ply, default)
uv run scripts/probcut.py data.csv

# Endgame parameters
uv run scripts/probcut.py data.csv --variant endgame
```

### ProbCut Arguments

- `csv_path`: (Required) Path to the input CSV file
- `--variant`: Training variant: `midgame` (default) or `endgame`
- `--max-ply`: Number of plies to emit in Rust array (midgame only, default: 60)
- `--folds`: K-fold for OOF residuals (default: 5)
- `--alpha`: Smoothing strength for local MAE (default: 5.0)
- `--seed`: Random seed for KFold shuffling (default: 42)
- `--epsilon`: Stability constant for log std (default: 1e-8)

The output is Rust code printed to stdout that can be copied into the Neural Reversi engine.

## Preparing Training Data

Refer to [neural-reversi/datagen](https://github.com/natsutteatsuiyone/neural-reversi/tree/main/datagen).

## License

This project is licensed under the [GNU General Public License v3 (GPL v3)](LICENSE). By using or contributing to this project, you agree to comply with the terms of the license.

This project includes code originally licensed under GPL v3 from the following project:

- **[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)**
