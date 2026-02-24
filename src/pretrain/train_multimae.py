import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pretrain.baselines_common import MODALITIES, build_loaders, move_batch, save_checkpoint, set_seed
from src.utils import plot_single_loss_curve


class MultiMAE1D(nn.Module):
    def __init__(self, signal_length: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(signal_length // 8),
                nn.Flatten(),
                nn.Linear(hidden_dim * (signal_length // 8), latent_dim),
            )
            for m in MODALITIES
        })
        self.decoders = nn.ModuleDict({m: nn.Linear(latent_dim, signal_length) for m in MODALITIES})

    def forward(self, x: Dict[str, torch.Tensor]):
        latents = {m: self.encoders[m](x[m]) for m in MODALITIES}
        shared = torch.stack(list(latents.values()), dim=0).mean(dim=0)
        recons = {m: self.decoders[m](shared).unsqueeze(1) for m in MODALITIES}
        return recons


def random_mask(batch: Dict[str, torch.Tensor], mask_ratio: float):
    masked, masks = {}, {}
    for m, x in batch.items():
        keep = (torch.rand_like(x) > mask_ratio).float()
        masked[m] = x * keep
        masks[m] = 1.0 - keep
    return masked, masks


def masked_mse(pred, target, mask):
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).pow(2) * mask).sum() / denom


def setup_logging(run_name: str, output_path: str) -> None:
    """Configure file + console logging, matching the pretraining MAE script style."""
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def evaluate(model, val_loader, device, mask_ratio):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = move_batch(batch, device)
            masked_batch, masks = random_mask(batch, mask_ratio)
            recons = model(masked_batch)
            loss = sum(masked_mse(recons[m], batch[m], masks[m]) for m in MODALITIES) / len(MODALITIES)
            total += loss.item()
    return total / max(1, len(val_loader))


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = os.path.join(args.output_path, args.run_name)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    loaders = build_loaders(args.data_path, args.fold_number, args.batch_size, args.val_split, args.seed, args.num_workers)

    sample_batch = next(iter(loaders.train_loader))
    signal_length = sample_batch[MODALITIES[0]].shape[-1]

    model = MultiMAE1D(signal_length, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logging.info("--- Starting New Training Run ---")
    logging.info(f"Run Name: {args.run_name}")
    logging.info("Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("\nModel Architecture:")
    logging.info(str(model))
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info("-" * 50)
    logging.info(f"Training set size: {len(loaders.train_loader.dataset)}")
    logging.info(f"Validation set size: {len(loaders.val_loader.dataset)}")

    best_val = float('inf')
    losses = []

    for epoch in range(1, args.num_epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.num_epochs} ---")
        model.train()
        train_total = 0.0
        pbar_train = tqdm(loaders.train_loader, desc=f"Training Epoch {epoch}")
        for batch in pbar_train:
            batch = move_batch(batch, device)
            masked_batch, masks = random_mask(batch, args.mask_ratio)
            recons = model(masked_batch)
            loss = sum(masked_mse(recons[m], batch[m], masks[m]) for m in MODALITIES) / len(MODALITIES)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            pbar_train.set_postfix(Loss=f"{loss.item():.4f}")

        train_loss = train_total / max(1, len(loaders.train_loader))
        val_loss = evaluate(model, loaders.val_loader, device, args.mask_ratio)
        current_lr = optimizer.param_groups[0]["lr"]

        losses.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })
        losses_df = pd.DataFrame(losses)
        np.savez(
            os.path.join(run_dir, "losses.npz"),
            epoch=losses_df["epoch"].values,
            train_loss=losses_df["train_loss"].values,
            val_loss=losses_df["val_loss"].values,
            lr=losses_df["lr"].values,
        )

        logging.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_model_path = os.path.join(models_dir, "best_ckpt.pt")
            save_checkpoint(model, optimizer, epoch, best_model_path)
            logging.info(f"Saved best checkpoint to {best_model_path}")

        if epoch % 5 == 0:
            plot_path = os.path.join(run_dir, "loss_curves.png")
            plot_single_loss_curve(losses_df, plot_path)
            logging.info(f"Saved loss plot to {plot_path}")

    logging.info("Training complete.")
    final_plot_path = os.path.join(run_dir, "loss_curves.png")
    plot_single_loss_curve(pd.DataFrame(losses), final_plot_path)
    logging.info(f"Saved final loss plot to {final_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiMAE-style masked multimodal reconstruction pretraining for WESAD.')
    parser.add_argument('--run_name', type=str, default='multimae_baseline')
    parser.add_argument('--output_path', type=str, default='./results/pretrain_baselines')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fold_number', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    setup_logging(args.run_name, args.output_path)
    main(args)
