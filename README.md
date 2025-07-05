# Neural Reversi Training

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU
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
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### 3. Optional: NVIDIA Apex (Recommended for Performance)

For optimal training performance, install NVIDIA Apex:

```bash
# Install NVIDIA Apex for optimized training
uv pip install -v --no-deps --no-cache-dir --disable-pip-version-check --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git@master \
    --config-settings " \
    --build-option=--cpp_ext \
    --cuda_ext"
```

## Training

### Basic Training

```bash
uv run train.py --train_data ./data/train --val_data ./data/val
```

### Resume Training from Checkpoint

```bash
uv run train.py --train_data ./data/train --val_data ./data/val --resume_from_checkpoint ./ckpt/reversi-10-0.0123.ckpt
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
- `--small`: Use small model architecture

### Performance Optimization

**For maximum training throughput:**

1. **Batch Size**: Increase `--batch_size` to saturate GPU memory (try 32768, 65536)
2. **Data Loading**: Adjust `--num_workers` based on CPU cores (typically 4-8)
3. **File Usage**: Use `--file_usage_ratio 0.8` for faster epochs during development
4. **Mixed Precision**: The training uses bf16-mixed precision automatically
5. **Model Compilation**: PyTorch compilation is enabled by default for faster inference

**Memory optimization:**
```bash
# For large datasets, reduce file usage ratio
uv run train.py --train_data ./data/train --val_data ./data/val --file_usage_ratio 0.5

# For limited GPU memory, reduce batch size
uv run train.py --train_data ./data/train --val_data ./data/val --batch_size 8192
```

### Output

- Checkpoints saved in `ckpt/` (or `ckpt/small/` for small models)
- TensorBoard logs saved in `tb_logs/`
- Best models saved based on validation loss
- Last checkpoint automatically saved as `last.ckpt`

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
