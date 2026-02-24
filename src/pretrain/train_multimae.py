import argparse
import os
import sys
from typing import Dict

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pretrain.baselines_common import MODALITIES, build_loaders, move_batch, save_checkpoint, set_seed


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


def evaluate(model, val_loader, device, mask_ratio):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            masked_batch, masks = random_mask(batch, mask_ratio)
            recons = model(masked_batch)
            loss = sum(masked_mse(recons[m], batch[m], masks[m]) for m in MODALITIES) / len(MODALITIES)
            total += loss.item()
    return total / max(1, len(val_loader))


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    loaders = build_loaders(args.data_path, args.fold_number, args.batch_size, args.val_split, args.seed, args.num_workers)

    sample_batch = next(iter(loaders.train_loader))
    signal_length = sample_batch[MODALITIES[0]].shape[-1]

    model = MultiMAE1D(signal_length, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_dir = os.path.join(args.output_path, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_total = 0.0
        for batch in loaders.train_loader:
            batch = move_batch(batch, device)
            masked_batch, masks = random_mask(batch, args.mask_ratio)
            recons = model(masked_batch)
            loss = sum(masked_mse(recons[m], batch[m], masks[m]) for m in MODALITIES) / len(MODALITIES)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        train_loss = train_total / max(1, len(loaders.train_loader))
        val_loss = evaluate(model, loaders.val_loader, device, args.mask_ratio)
        print(f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(run_dir, 'best_ckpt.pt'))


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
    main(parser.parse_args())
