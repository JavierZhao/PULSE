import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.wesad_dataset import WESADDataset

MODALITIES = ["ecg", "bvp", "acc", "temp"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader


def build_loaders(
    data_path: str,
    fold_number: int,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
) -> DataBundle:
    dataset = WESADDataset(data_path=data_path, fold_number=fold_number, split="train")
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    return DataBundle(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


class SimpleTimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(input_dim // 8),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * (input_dim // 8), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


class MultiModalProjector(nn.Module):
    def __init__(self, signal_length: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict({m: SimpleTimeSeriesEncoder(signal_length, hidden_dim, proj_dim) for m in MODALITIES})

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: F.normalize(self.encoders[m](batch[m]), dim=-1) for m in MODALITIES}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def info_nce(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def pairwise_clip_loss(embeddings: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    keys = list(embeddings.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            zi = embeddings[keys[i]]
            zj = embeddings[keys[j]]
            logits_ij = zi @ zj.T / temperature
            logits_ji = zj @ zi.T / temperature
            losses.append(0.5 * (info_nce(logits_ij) + info_nce(logits_ji)))
    return torch.stack(losses).mean()


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {m: batch[m].to(device) for m in MODALITIES}
