# ResNet1D: 1D Residual Network backbone for physiological time-series
#
# An alternative to the Transformer/MAE architecture, using stacked residual
# blocks with 1D convolutions.  Produces per-patch token embeddings that are
# split into private/shared subspaces (same interface as CheapSensorMAE) so
# it can be used as a drop-in replacement in the finetuning pipeline.

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .model_utils.pos_embed import get_1d_sincos_pos_embed
from timm.models.vision_transformer import Block


class ResidualBlock1D(nn.Module):
    """Basic residual block: two 1-D convolutions with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet1DEncoder(nn.Module):
    """
    1-D ResNet encoder that produces token embeddings at a patch-level
    resolution comparable to the Transformer MAE encoder.

    Architecture:
      1. Stem convolution (maps 1 input channel → base_channels)
      2. N residual stages, each doubling the channel count and halving the
         temporal resolution.
      3. A final 1×1 projection to ``embed_dim``.

    The output sequence length equals ``sig_len // total_stride``, which should
    match ``num_patches = sig_len // window_len`` when the strides are chosen
    appropriately.
    """

    def __init__(self, sig_len: int = 3840, window_len: int = 96,
                 in_chans: int = 1, embed_dim: int = 1024,
                 base_channels: int = 232, num_blocks_per_stage: int = 2,
                 kernel_size: int = 7, private_mask_ratio: float = 0.5):
        super().__init__()

        self.sig_len = sig_len
        self.window_len = window_len
        self.num_patches = sig_len // window_len
        self.embed_dim = embed_dim
        self.private_mask_ratio = private_mask_ratio

        # --- Stem ---
        # Use window_len as the kernel/stride so that each "token" corresponds
        # to exactly one non-overlapping window (same as PatchEmbed1D).
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, base_channels, kernel_size=window_len,
                      stride=window_len, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # --- Residual stages (all stride-1 to preserve token count) ---
        channels = base_channels
        stages = []
        for stage_idx in range(4):
            out_channels = channels * 2 if stage_idx > 0 else channels
            downsample = None
            if out_channels != channels:
                downsample = nn.Sequential(
                    nn.Conv1d(channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            blocks = [ResidualBlock1D(channels, out_channels, kernel_size,
                                      stride=1, downsample=downsample)]
            for _ in range(1, num_blocks_per_stage):
                blocks.append(ResidualBlock1D(out_channels, out_channels,
                                              kernel_size))
            stages.append(nn.Sequential(*blocks))
            channels = out_channels

        self.stages = nn.Sequential(*stages)

        # --- Projection to embed_dim ---
        self.proj = nn.Conv1d(channels, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        # --- CLS token (prepended to sequence) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- Fixed private mask (same scheme as CheapSensorMAE.Encoder) ---
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
        """
        Parameters match the Transformer encoder interface so that the
        finetuning pipeline can call ``forward_encoder`` identically.

        During finetuning ``mask_ratio`` is set to 0 and ids_* are dummies
        produced by ``propose_masking``.

        **Masking strategy for CNNs**: Unlike the Transformer MAE where
        masked tokens are simply removed from the input sequence, a CNN
        processes a contiguous spatial grid and cannot skip positions.
        Instead we **zero-out masked patches at the input level** before
        the signal enters the convolutional layers.  This prevents
        information leakage from masked patches through the convolutions
        and makes the reconstruction task non-trivial.
        """
        B = x.shape[0]

        # --- Apply input-level masking (for pretraining) ---
        if mask_ratio > 0.0 and ids_restore is not None:
            N = x.shape[0]
            L = self.num_patches
            len_keep = int(L * (1 - mask_ratio))
            # Build binary mask: 0 = keep, 1 = remove
            mask = torch.ones(N, L, device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)  # (N, L)
            # Expand mask to raw signal resolution and zero-out masked patches
            # mask shape: (N, L) → (N, 1, L*window_len)
            input_mask = mask.unsqueeze(-1).repeat(1, 1, self.window_len)  # (N, L, W)
            input_mask = input_mask.reshape(N, 1, -1)  # (N, 1, L*W)
            x = x * (1.0 - input_mask)  # zero out masked patches
        else:
            mask = torch.zeros(B, self.num_patches, device=x.device)

        # (B, 1, L_sig) → stem → stages → proj → (B, embed_dim, T)
        h = self.stem(x)
        h = self.stages(h)
        h = self.proj(h)  # (B, embed_dim, T)
        h = h.transpose(1, 2)  # (B, T, embed_dim)
        h = self.norm(h)

        # For masked pretraining: keep only unmasked token embeddings
        # (decoder will reinsert mask tokens at masked positions)
        if mask_ratio > 0.0 and ids_keep is not None:
            N, L, D = h.shape
            h = torch.gather(h, dim=1,
                             index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, T_kept+1, embed_dim)

        # Split into private / shared
        private_embedding = h * self.private_mask
        shared_embedding = h * (1 - self.private_mask)

        if return_hiddens:
            return private_embedding, shared_embedding, mask, [h]
        return private_embedding, shared_embedding, mask


class ResNet1DDecoder(nn.Module):
    """
    Transformer-style decoder for masked patch reconstruction.

    This mirrors CheapSensorMAE's decoder behavior so masked patch predictions
    can attend to visible token content before the final regression head.
    """

    def __init__(self, embed_dim: int = 1024, decoder_dim: int = 512,
                 num_patches: int = 40, window_len: int = 96,
                 in_chans: int = 1, num_layers: int = 4,
                 num_heads: int = 16, mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, window_len * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x


class ResNet1DMAE(nn.Module):
    """
    Drop-in replacement for ``CheapSensorMAE`` using a ResNet1D backbone.

    Exposes the same public API so finetuning and pretraining scripts work
    without modification:
      - ``propose_masking``
      - ``forward_encoder``
      - ``forward_decoder``
      - ``reconstruction_loss``
      - ``patchify`` / ``unpatchify``
      - ``num_patches``, ``modality_name``
    """

    def __init__(self, modality_name: str, sig_len: int = 3840,
                 window_len: int = 96, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 8, num_heads: int = 8,
                 decoder_embed_dim: int = 512, decoder_depth: int = 4,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0, decoder_mlp_ratio: float = 4.0,
                 norm_layer=nn.LayerNorm, norm_pix_loss: bool = True,
                 private_mask_ratio: float = 0.5,
                 # ResNet-specific
                 base_channels: int = 232, num_blocks_per_stage: int = 2,
                 kernel_size: int = 7):
        super().__init__()
        print(f"Initializing ResNet1DMAE for {modality_name}")

        self.modality_name = modality_name
        self.window_len = window_len
        self.num_patches = sig_len // window_len
        self.norm_pix_loss = norm_pix_loss

        self.encoder = ResNet1DEncoder(
            sig_len=sig_len, window_len=window_len, in_chans=in_chans,
            embed_dim=embed_dim, base_channels=base_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            kernel_size=kernel_size,
            private_mask_ratio=private_mask_ratio,
        )

        self.decoder = ResNet1DDecoder(
            embed_dim=embed_dim, decoder_dim=decoder_embed_dim,
            num_patches=self.num_patches, window_len=window_len,
            in_chans=in_chans, num_layers=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=decoder_mlp_ratio,
            norm_layer=norm_layer,
        )

    # ------------------------------------------------------------------
    # Shared helpers (identical to CheapSensorMAE)
    # ------------------------------------------------------------------
    def patchify(self, sigs):
        p = self.window_len
        l = sigs.shape[-1] // p
        x = sigs.reshape(sigs.shape[0], 1, l, p)
        x = torch.einsum("nchp->nhpc", x)
        x = x.reshape(sigs.shape[0], l, p)
        return x

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
    # Forward API (matches CheapSensorMAE)
    # ------------------------------------------------------------------
    def forward_encoder(self, x, mask_ratio, ids_shuffle, ids_restore,
                        ids_keep):
        private, shared, mask = self.encoder(
            x, mask_ratio, ids_shuffle, ids_restore, ids_keep)
        return private, shared, mask

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
