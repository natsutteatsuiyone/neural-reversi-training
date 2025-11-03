import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import FeatureDataset, custom_collate_fn
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import model
import model_sm
import model_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data directory",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to the validation data directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=65536,
        help="Batch size (applied inside FeatureDataset)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        dest="resume_from_checkpoint",
    )
    parser.add_argument(
        "--resume_from_weights",
        dest="resume_from_weights",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the dataset",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the optimizer",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--file_usage_ratio",
        type=float,
        default=1.0,
        help="Ratio of files to use per epoch (0.0 to 1.0)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
    )
    return parser.parse_args()


def prepare_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    num_workers: int,
    file_usage_ratio: float,
    shuffle: bool,
    small: bool,
) -> tuple:
    train_files = [str(p) for p in Path(train_dir).glob("*.zst")]
    val_files = [str(p) for p in Path(val_dir).glob("*.zst")]

    train_dataset = FeatureDataset(
        filepaths=train_files,
        batch_size=batch_size,
        file_usage_ratio=file_usage_ratio,
        num_features=model_common.NUM_FEATURES,
        num_feature_params=model_common.NUM_FEATURE_PARAMS,
        shuffle=shuffle,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataset = FeatureDataset(
        filepaths=val_files,
        batch_size=batch_size,
        file_usage_ratio=1.0,
        num_features=model_common.NUM_FEATURES,
        num_feature_params=model_common.NUM_FEATURE_PARAMS,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

def prepare_callbacks(small: bool) -> tuple:
    lr_monitor = LearningRateMonitor(logging_interval="step")

    dirpath = "ckpt/small" if small else "ckpt"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=32,
        monitor="val_loss",
        mode="min",
        dirpath=dirpath,
        filename="{epoch}-{val_loss:.8f}",
        save_on_train_epoch_end=True,
    )

    # checkpoint_callback = ModelCheckpoint(
    #     save_last=True,
    #     every_n_epochs=10,
    #     filename="{epoch}-{val_loss:.8f}",
    #     dirpath=dirpath,
    #     save_top_k=-1,
    # )

    return [checkpoint_callback, lr_monitor]

def main():
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    train_loader, val_loader = prepare_dataloaders(
        args.train_data,
        args.val_data,
        args.batch_size,
        args.num_workers,
        args.file_usage_ratio,
        args.shuffle,
        args.small,
    )

    if args.small:
        if args.resume_from_weights:
            reversi_model = model_sm.LitReversiSmallModel(lr=args.lr, t_max=args.epochs, weight_decay=args.weight_decay)
            checkpoint = torch.load(args.resume_from_weights, weights_only=True)
            reversi_model.load_state_dict(checkpoint["state_dict"])
        else:
            reversi_model = model_sm.LitReversiSmallModel(lr=args.lr, t_max=args.epochs, weight_decay=args.weight_decay)
    else:
        if args.resume_from_checkpoint:
            reversi_model = model.LitReversiModel(
                lr=args.lr, weight_decay=args.weight_decay, t_max=args.epochs
            )
        elif args.resume_from_weights:
            reversi_model = model.LitReversiModel(
                lr=args.lr, weight_decay=args.weight_decay, t_max=args.epochs
            )
            checkpoint = torch.load(args.resume_from_weights, weights_only=True)
            state = checkpoint["state_dict"]
            # Backward-compat: map old keys (no 'model.' prefix) to new ones
            if not any(k.startswith("model.") for k in state.keys()):
                state = {f"model.{k}": v for k, v in state.items()}
            reversi_model.load_state_dict(state, strict=False)
        else:
            reversi_model = model.LitReversiModel(
                lr=args.lr, weight_decay=args.weight_decay, t_max=args.epochs
            )

    logger = TensorBoardLogger("tb_logs", name="reversi_model")
    callbacks = prepare_callbacks(args.small)

    trainer = L.Trainer(
        callbacks=callbacks,
        log_every_n_steps=2000,
        logger=logger,
        max_epochs=args.epochs,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(
        reversi_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint or None,
    )

if __name__ == "__main__":
    main()
