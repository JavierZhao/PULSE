# ContrastiveEncoder: SimCLR-style contrastive learning for physiological signals
#
# Instead of reconstructing masked patches (MAE), this approach learns
# representations by maximizing agreement between differently-augmented views
# of the same signal (positive pairs) while pushing apart different signals
# (negative pairs) using the NT-Xent (Normalized Temperature-scaled Cross
# Entropy) loss.
#
# Multi-modal extension: cross-modal contrastive learning aligns
# representations of co-occurring signals from different modalities.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .model_utils.pos_embed import get_1d_sincos_pos_embed
from timm.models.vision_transformer import Block


# ---------------------------------------------------------------------------
# Signal augmentations for contrastive views
# ---------------------------------------------------------------------------
class SignalAugmentor(nn.Module):
    """Stochastic augmentation pipeline for 1-D physiological signals."""

    def __init__(self, noise_std: float = 0.05, scale_range: tuple = (0.8, 1.2),
                 shift_max: int = 50, mask_ratio: float = 0.15):
        super().__init__()
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L)"""
        B, C, L = x.shape
        aug = x.clone()

        # 1. Additive Gaussian noise
        if self.noise_std > 0:
            aug = aug + torch.randn_like(aug) * self.noise_std

        # 2. Random scaling
        lo, hi = self.scale_range
        scales = torch.empty(B, 1, 1, device=x.device).uniform_(lo, hi)
        aug = aug * scales

        # 3. Temporal shift (circular)
        if self.shift_max > 0:
            shifts = torch.randint(-self.shift_max, self.shift_max + 1, (B,))
            for i in range(B):
                aug[i] = torch.roll(aug[i], shifts[i].item(), dims=-1)

        # 4. Random temporal masking (zero-out contiguous segment)
        if self.mask_ratio > 0:
            mask_len = int(L * self.mask_ratio)
            starts = torch.randint(0, L - mask_len, (B,))
            for i in range(B):
                aug[i, :, starts[i]:starts[i] + mask_len] = 0.0

        return aug


# ---------------------------------------------------------------------------
# Projection head (maps encoder output → low-dim contrastive space)
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss (SimCLR)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, proj_dim) — L2-normalized projections of two views.
        Returns scalar NT-Xent loss.
        """
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_select(mask).view(2 * B, -1)  # (2B, 2B-1)

        # Positive pair indices: for i in [0..B-1], positive is at index B-1+i
        # because after removing self, the partner (at original index B+i) is
        # at position B-1+i in the masked row.  Similarly for i in [B..2B-1].
        pos_idx = torch.cat([
            torch.arange(B - 1, 2 * B - 1, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, pos_idx)
        return loss


# ---------------------------------------------------------------------------
# Contrastive encoder (Transformer-based, same backbone as MAE encoder)
# ---------------------------------------------------------------------------
class ContrastiveTransformerEncoder(nn.Module):
    """
    Transformer encoder for contrastive pretraining.  Re-uses the same
    patch-embed + Transformer blocks architecture as the MAE encoder but
    does *not* mask input patches — instead, two augmented views are
    separately encoded and contrasted.

    For finetuning compatibility the encoder still supports private/shared
    splitting and the masking interface (with mask_ratio=0).
    """

    def __init__(self, sig_len: int = 3840, window_len: int = 96,
                 in_chans: int = 1, embed_dim: int = 1024,
                 depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, norm_layer=nn.LayerNorm,
                 private_mask_ratio: float = 0.5):
        super().__init__()
        from .CheapSensorMAE import PatchEmbed1D

        self.patch_embed = PatchEmbed1D(sig_len, window_len, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.private_mask_ratio = private_mask_ratio

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Fixed private mask
        num_private = int(embed_dim * private_mask_ratio)
        noise = torch.rand(embed_dim)
        ids = torch.argsort(noise)
        private_mask = torch.zeros(embed_dim)
        private_mask[ids[:num_private]] = 1.0
        self.register_buffer("private_mask", private_mask)

        self._init_weights()

    def _init_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self.__init_subweights)

    @staticmethod
    def __init_subweights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                mask_ratio: float = 0.0,
                ids_shuffle=None, ids_restore=None, ids_keep=None,
                return_hiddens: bool = False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # Masking (only used at finetuning time with mask_ratio=0 for API compat)
        if mask_ratio > 0.0 and ids_keep is not None:
            N, L, D = x.shape
            len_keep = int(L * (1 - mask_ratio))
            x = torch.gather(x, dim=1,
                             index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
            mask = torch.ones(N, L, device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        hiddens = []
        for blk in self.blocks:
            x = blk(x)
            if return_hiddens:
                hiddens.append(x)
        x = self.norm(x)

        private_embedding = x * self.private_mask
        shared_embedding = x * (1 - self.private_mask)

        if return_hiddens:
            return private_embedding, shared_embedding, mask, hiddens
        return private_embedding, shared_embedding, mask


# ---------------------------------------------------------------------------
# Full contrastive model
# ---------------------------------------------------------------------------
class ContrastiveModel(nn.Module):
    """
    SimCLR-style contrastive model for a single modality.

    Exposes the same public API as CheapSensorMAE for finetuning:
      propose_masking, forward_encoder, num_patches, modality_name, encoder
    """

    def __init__(self, modality_name: str, sig_len: int = 3840,
                 window_len: int = 96, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, norm_layer=nn.LayerNorm,
                 private_mask_ratio: float = 0.5,
                 proj_dim: int = 128, temperature: float = 0.1,
                 # ignored kwargs for API compatibility with CheapSensorMAE
                 decoder_embed_dim=512, decoder_depth=4,
                 decoder_num_heads=16, decoder_mlp_ratio=4.0,
                 norm_pix_loss=True):
        super().__init__()
        print(f"Initializing ContrastiveModel for {modality_name}")
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
        self.projection = ProjectionHead(embed_dim, proj_dim)
        self.criterion = NTXentLoss(temperature)

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

    # ---- Contrastive training interface ----
    def encode_and_project(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal and produce projection for contrastive loss.
        Returns (B, proj_dim) L2-normalized vector.
        """
        private, shared, _ = self.encoder(x)
        # Pool CLS token (index 0) of the combined representation
        cls_repr = (private + shared)[:, 0, :]
        z = self.projection(cls_repr)
        return z

    def contrastive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Create two augmented views and compute NT-Xent loss."""
        view1 = self.augmentor(x)
        view2 = self.augmentor(x)
        z1 = self.encode_and_project(view1)
        z2 = self.encode_and_project(view2)
        return self.criterion(z1, z2)
