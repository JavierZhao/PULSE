# VICRegEncoder: Variance-Invariance-Covariance Regularization for
# physiological time-series self-supervised pretraining.
#
# VICReg avoids representational collapse *without* negative pairs or
# momentum encoders by enforcing three complementary objectives on the
# embedding batch:
#   1. **Invariance** — MSE between representations of two augmented views
#      of the same sample (pull positives together).
#   2. **Variance** — hinge loss on the standard deviation of each embedding
#      dimension across the batch (prevents collapse to a single point).
#   3. **Covariance** — penalizes off-diagonal elements of the covariance
#      matrix (decorrelates dimensions, avoids informational collapse).
#
# Reference: Bardes, Ponce & LeCun – "VICReg: Variance-Invariance-Covariance
# Regularization for Self-Supervised Learning" (ICLR 2022).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# Re-use the same augmentor and encoder backbone from ContrastiveEncoder
from .ContrastiveEncoder import (
    SignalAugmentor,
    ContrastiveTransformerEncoder,
)


# ---------------------------------------------------------------------------
# VICReg loss components
# ---------------------------------------------------------------------------
def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Hinge loss encouraging the std of each dimension >= gamma."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean squared error between paired representations."""
    return F.mse_loss(z1, z2)


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalises off-diagonal elements of the batch covariance matrix."""
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


class VICRegLoss(nn.Module):
    """Combined VICReg loss."""

    def __init__(self, inv_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        z1, z2: (B, D) — representations of two views.
        Returns (total_loss, {inv, var, cov} component dict).
        """
        inv = invariance_loss(z1, z2)
        var = (variance_loss(z1, self.gamma) + variance_loss(z2, self.gamma)) / 2
        cov = (covariance_loss(z1) + covariance_loss(z2)) / 2

        total = self.inv_weight * inv + self.var_weight * var + self.cov_weight * cov
        return total, {"invariance": inv.item(), "variance": var.item(),
                       "covariance": cov.item()}


# ---------------------------------------------------------------------------
# Expander (projects encoder output → higher dim for VICReg, then a final
# projection maps back down — following the VICReg paper recommendation)
# ---------------------------------------------------------------------------
class Expander(nn.Module):
    """3-layer MLP expander (encoder_dim → expand_dim → expand_dim → expand_dim)."""

    def __init__(self, embed_dim: int, expand_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, expand_dim),
            nn.BatchNorm1d(expand_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expand_dim, expand_dim),
            nn.BatchNorm1d(expand_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expand_dim, expand_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# VICReg model wrapper
# ---------------------------------------------------------------------------
class VICRegModel(nn.Module):
    """
    VICReg self-supervised model for a single modality.

    Same finetuning-compatible API as CheapSensorMAE:
      propose_masking, forward_encoder, num_patches, modality_name, encoder
    """

    def __init__(self, modality_name: str, sig_len: int = 3840,
                 window_len: int = 96, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, norm_layer=nn.LayerNorm,
                 private_mask_ratio: float = 0.5,
                 expand_dim: int = 2048,
                 inv_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0,
                 # Ignored kwargs for API compatibility
                 decoder_embed_dim=512, decoder_depth=4,
                 decoder_num_heads=16, decoder_mlp_ratio=4.0,
                 norm_pix_loss=True):
        super().__init__()
        print(f"Initializing VICRegModel for {modality_name}")
        self.modality_name = modality_name
        self.window_len = window_len
        self.num_patches = sig_len // window_len

        self.encoder = ContrastiveTransformerEncoder(
            sig_len=sig_len, window_len=window_len, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            private_mask_ratio=private_mask_ratio,
        )

        self.augmentor = SignalAugmentor()
        self.expander = Expander(embed_dim, expand_dim)
        self.criterion = VICRegLoss(inv_weight, var_weight, cov_weight)

    # ---- Finetuning-compatible interface ----
    def propose_masking(self, batch_size, num_patches, mask_ratio, devic):
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=devic)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_shuffle, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, ids_shuffle, ids_restore,
                        ids_keep):
        return self.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep)

    # ---- VICReg training interface ----
    def encode_and_expand(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal and produce expanded representation."""
        private, shared, _ = self.encoder(x)
        cls_repr = (private + shared)[:, 0, :]  # CLS token
        return self.expander(cls_repr)

    def vicreg_loss(self, x: torch.Tensor):
        """Create two augmented views and compute VICReg loss."""
        view1 = self.augmentor(x)
        view2 = self.augmentor(x)
        z1 = self.encode_and_expand(view1)
        z2 = self.encode_and_expand(view2)
        return self.criterion(z1, z2)
