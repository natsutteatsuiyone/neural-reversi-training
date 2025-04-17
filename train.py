import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import FeatureDataset, custom_collate_fn
from model import ReversiModel
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import random

from model_sm import ReversiSmallModel


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
        default=8192 * 4,
        help="Batch size (applied inside FeatureDataset)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10000,
        help="Maximum number of epochs",
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
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--t_max", type=int, default=300, help="CosineAnnealingLR's T_max"
    )
    parser.add_argument(
        "--random_skipping",
        type=int,
        default=3,
        help="Random skipping for FeatureDataset",
    )
    parser.add_argument(
        "--small_net",
        action="store_true",
    )
    return parser.parse_args()


def prepare_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    num_workers: int,
    random_skipping: int,
) -> tuple:
    train_files = [str(p) for p in Path(train_dir).glob("*.zst")]
    random.shuffle(train_files)
    val_files = [str(p) for p in Path(val_dir).glob("*.zst")]

    train_dataset = FeatureDataset(
        filepaths=train_files, batch_size=batch_size, random_skipping=random_skipping
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataset = FeatureDataset(
        filepaths=val_files, batch_size=batch_size, random_skipping=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def prepare_logger_and_callbacks() -> tuple:
    logger = TensorBoardLogger("tb_logs", name="reversi_model")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    epoch_checkpoint = ModelCheckpoint(
        save_top_k=30,
        monitor="val_loss",
        mode="min",
        dirpath="ckpt/",
        filename="{epoch}-{val_loss:.2f}",
        save_on_train_epoch_end=True,
        save_last=True,
    )

    return logger, [epoch_checkpoint, lr_monitor]


def main():
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    train_loader, val_loader = prepare_dataloaders(
        args.train_data,
        args.val_data,
        args.batch_size,
        args.num_workers,
        args.random_skipping,
    )

    if args.small_net:
        model = ReversiSmallModel(
            lr=args.lr,
            t_max=args.t_max,
        )
    else:
        if args.resume_from_checkpoint:
            model = ReversiModel.load_from_checkpoint(args.resume_from_checkpoint)
        elif args.resume_from_weights:
            checkpoint = torch.load(f=args.resume_from_weights, weights_only=True)
            model = ReversiModel(lr=args.lr, t_max=args.t_max)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model = ReversiModel(lr=args.lr, t_max=args.t_max)
    model = torch.compile(model)

    logger, callbacks = prepare_logger_and_callbacks()

    trainer = L.Trainer(
        callbacks=callbacks,
        log_every_n_steps=500,
        logger=logger,
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
