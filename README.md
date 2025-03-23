# Neural Reversi Training

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv)

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/natsutteatsuiyone/neural-reversi-training.git
cd neural-reversi-training
```

### 2. Install Dependencies

```bash
uv venv
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
uv sync
```

## Training

### Basic Training

```bash
uv run train.py --train_data ./data/train --val_data ./data/val
```

### Resume Training from Checkpoint

```bash
uv run train.py --train_data ./data/train --val_data ./data/val --resume-from-checkpoint ./chkpt/reversi-10-0.0123.ckpt
```

### Command Line Arguments

**Required:**
- `--train_data`: Path to training data (*.zst)
- `--val_data`: Path to validation data (*.zst)

**Optional:**
- `--batch_size`: Training batch size (default: 16384)
- `--num_workers`: Data loading workers (default: 4)
- `--max_epochs`: Maximum epochs (default: 10000)
- `--val_check_interval`: Steps interval for validation (default: 5000)
- `--seed`: Random seed (default: 42)
- `--resume-from-checkpoint`: Checkpoint path to resume training
- `--lr`: Learning rate (default: 0.001)
- `--t_max`: CosineAnnealingLR T_max (default: 300)
- `--random_skipping`: Random skipping factor (default: 3)

### Output

- Checkpoints saved in `chkpt/`
- TensorBoard logs saved in `tb_logs/`
- Best models saved based on validation loss

## Creating Weight Files

Convert trained checkpoint to a weight file:

```bash
uv run serialize.py --checkpoint ./chkpt/reversi-best.ckpt --output ./eval.zst
```

### `serialize.py` Arguments

- `--checkpoint`: (Required) Checkpoint file path
- `--output`: Output weight file path (default: eval.zst)
- `--cl`: Compression level (default: 7)
- `--no-hist`: Disable histogram display

The output is a zstandard-compressed binary, compatible with the Neural Reversi engine.

## Preparing Training Data

Refer to [neural-reversi/datagen](https://github.com/natsutteatsuiyone/neural-reversi/tree/main/datagen).

## Workflow Summary

1. Train using `train.py`
2. Generate weight file with `serialize.py`

## License

This project is licensed under the [GNU General Public License v3 (GPL v3)](LICENSE). By using or contributing to this project, you agree to comply with the terms of the license.

This project includes code originally licensed under GPL v3 from the following project:

- **[nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)**
