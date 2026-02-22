# TCN: Temporal Convolutional Network backbone for physiological time-series
#
# An alternative to both Transformers and plain ResNets.  Uses stacked dilated
# causal 1-D convolutions so that the receptive field grows exponentially with
# depth while preserving temporal ordering.
#
# Produces per-patch token embeddings with private/shared split (same
# interface as CheapSensorMAE) for drop-in use in finetuning.

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CausalConv1d(nn.Module):
    """1-D causal convolution with dilation (no information leakage from future)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation,
                              padding=self.padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future positions introduced by left-padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Single TCN residual block:
      CausalConv → BatchNorm → ReLU → Dropout →
      CausalConv → BatchNorm → ReLU → Dropout + skip
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip = (nn.Conv1d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.skip(x))


class TCNEncoder(nn.Module):
    """
    Multi-layer TCN encoder.

    1. Stem: non-overlapping window convolution (same as PatchEmbed1D) that
       converts the raw signal into ``num_patches`` tokens.
    2. N TCN blocks with exponentially increasing dilation.
    3. 1×1 projection to ``embed_dim``.

    Output: (B, T+1, embed_dim) — includes a prepended CLS token.
    """

    def __init__(self, sig_len: int = 3840, window_len: int = 96,
                 in_chans: int = 1, embed_dim: int = 1024,
                 tcn_channels: int = 256, num_layers: int = 6,
                 kernel_size: int = 3, dropout: float = 0.1,
                 private_mask_ratio: float = 0.5):
        super().__init__()

        self.sig_len = sig_len
        self.window_len = window_len
        self.num_patches = sig_len // window_len
        self.embed_dim = embed_dim
        self.private_mask_ratio = private_mask_ratio

        # --- Stem (patch-embed) ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, tcn_channels, kernel_size=window_len,
                      stride=window_len, bias=False),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU(inplace=True),
        )

        # --- TCN blocks with exponentially growing dilation ---
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(tcn_channels, tcn_channels,
                                   kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

        # --- Projection ---
        self.proj = nn.Conv1d(tcn_channels, embed_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        # --- CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Fixed private mask ---
        num_private = int(embed_dim * private_mask_ratio)
        noise = torch.rand(embed_dim)
        ids = torch.argsort(noise)
        private_mask = torch.zeros(embed_dim)
        private_mask[ids[:num_private]] = 1.0
        self.register_buffer("private_mask", private_mask)

    def forward(self, x: torch.Tensor,
                mask_ratio: float = 0.0,
                ids_shuffle=None, ids_restore=None, ids_keep=None,
                return_hiddens: bool = False):
        B = x.shape[0]

        h = self.stem(x)         # (B, C, num_patches)
        h = self.tcn(h)          # (B, C, num_patches)
        h = self.proj(h)         # (B, embed_dim, num_patches)
        h = h.transpose(1, 2)    # (B, num_patches, embed_dim)
        h = self.norm(h)

        # --- Optional masking ---
        if mask_ratio > 0.0 and ids_keep is not None:
            N, L, D = h.shape
            len_keep = int(L * (1 - mask_ratio))
            h = torch.gather(h, dim=1,
                             index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
            mask = torch.ones(N, L, device=h.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            mask = torch.zeros(B, h.shape[1], device=h.device)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)

        # Private / shared split
        private_embedding = h * self.private_mask
        shared_embedding = h * (1 - self.private_mask)

        if return_hiddens:
            return private_embedding, shared_embedding, mask, [h]
        return private_embedding, shared_embedding, mask


class TCNDecoder(nn.Module):
    """MLP-based decoder (matches ResNet1DDecoder)."""

    def __init__(self, embed_dim: int = 1024, decoder_dim: int = 512,
                 num_patches: int = 40, window_len: int = 96,
                 in_chans: int = 1, num_layers: int = 2):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(decoder_dim, decoder_dim), nn.GELU()]
        layers.append(nn.Linear(decoder_dim, window_len * in_chans))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x[:, 1:, :]
        return self.mlp(x)


class TCNMAE(nn.Module):
    """
    Drop-in replacement for ``CheapSensorMAE`` using a TCN backbone.

    Same public API:
      propose_masking, forward_encoder, forward_decoder,
      reconstruction_loss, patchify, unpatchify, num_patches, modality_name.
    """

    def __init__(self, modality_name: str, sig_len: int = 3840,
                 window_len: int = 96, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 8, num_heads: int = 8,
                 decoder_embed_dim: int = 512, decoder_depth: int = 4,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0, decoder_mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss: bool = True,
                 private_mask_ratio: float = 0.5,
                 # TCN-specific
                 tcn_channels: int = 256, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        print(f"Initializing TCNMAE for {modality_name}")

        self.modality_name = modality_name
        self.window_len = window_len
        self.num_patches = sig_len // window_len
        self.norm_pix_loss = norm_pix_loss

        self.encoder = TCNEncoder(
            sig_len=sig_len, window_len=window_len, in_chans=in_chans,
            embed_dim=embed_dim, tcn_channels=tcn_channels,
            num_layers=depth, kernel_size=kernel_size, dropout=dropout,
            private_mask_ratio=private_mask_ratio,
        )

        self.decoder = TCNDecoder(
            embed_dim=embed_dim, decoder_dim=decoder_embed_dim,
            num_patches=self.num_patches, window_len=window_len,
            in_chans=in_chans, num_layers=decoder_depth,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def patchify(self, sigs):
        p = self.window_len
        l = sigs.shape[-1] // p
        x = sigs.reshape(sigs.shape[0], 1, l, p)
        x = torch.einsum("nchp->nhpc", x)
        return x.reshape(sigs.shape[0], l, p)

    def unpatchify(self, x):
        p = self.window_len
        l = x.shape[1]
        return x.reshape(x.shape[0], 1, l * p)

    def propose_masking(self, batch_size, num_patches, mask_ratio, devic):
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=devic)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_shuffle, ids_restore, ids_keep

    # ------------------------------------------------------------------
    # Forward API
    # ------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio, ids_shuffle, ids_restore,
                        ids_keep):
        return self.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep)

    def forward_decoder(self, private_embedding, shared_embedding,
                        ids_restore):
        x = private_embedding + shared_embedding
        return self.decoder(x, ids_restore)

    def reconstruction_loss(self, sigs, pred, mask):
        target = self.patchify(sigs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_with_visualization(self, sigs, mask_ratio=0.75):
        device = sigs.device
        ids_shuffle, ids_restore, ids_keep = self.propose_masking(
            len(sigs), self.num_patches, mask_ratio, device)
        private, shared, mask = self.forward_encoder(
            sigs, mask_ratio, ids_shuffle, ids_restore, ids_keep)
        pred = self.forward_decoder(private, shared, ids_restore)
        return self.patchify(sigs), pred, mask
