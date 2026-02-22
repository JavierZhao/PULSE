# AutoregressiveTransformer: Causal (GPT-style) next-patch prediction for
# physiological time-series self-supervised pretraining.
#
# Instead of masking random patches and reconstructing them (MAE), this model
# predicts each patch from all *preceding* patches using a causal attention
# mask.  The pretraining objective is analogous to GPT's next-token
# prediction but operates on continuous signal patches rather than discrete
# tokens.
#
# Advantages over MAE:
#   - Naturally captures temporal ordering / causal structure in signals.
#   - No need for a separate decoder — the causal transformer directly
#     predicts the next patch.
#   - Can be used for generation / forecasting tasks.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .model_utils.pos_embed import get_1d_sincos_pos_embed


# ---------------------------------------------------------------------------
# Causal self-attention block
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask."""

    def __init__(self, embed_dim: int, num_heads: int, max_len: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Pre-compute causal mask (lower triangular)
        causal = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", causal.view(1, 1, max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class CausalTransformerBlock(nn.Module):
    """Transformer block with causal attention + MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_len, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Autoregressive encoder
# ---------------------------------------------------------------------------
class AutoregressiveEncoder(nn.Module):
    """
    GPT-style causal transformer that processes the full sequence of patches
    with a causal mask.  At position *t* the model can only attend to
    positions 0..t, so the representation at *t* encodes information about
    the past and present.

    The same private/shared embedding split is applied for compatibility
    with the multi-modal finetuning pipeline.
    """

    def __init__(self, sig_len: int = 3840, window_len: int = 96,
                 in_chans: int = 1, embed_dim: int = 1024,
                 depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 private_mask_ratio: float = 0.5):
        super().__init__()
        from .CheapSensorMAE import PatchEmbed1D

        self.patch_embed = PatchEmbed1D(sig_len, window_len, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.private_mask_ratio = private_mask_ratio

        # Learnable position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        # Causal transformer blocks
        max_len = num_patches + 2  # small buffer
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, mlp_ratio,
                                   max_len, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

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
            self.pos_embed.shape[-1], self.num_patches, cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        self.apply(self.__init_sub)

    @staticmethod
    def __init_sub(m):
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
        """
        Encodes the full sequence causally.
        For finetuning compatibility, mask_ratio is accepted but the causal
        mask is always applied (masking for MAE is not used here).
        """
        x = self.patch_embed(x)          # (B, T, D)
        x = x + self.pos_embed[:, :x.shape[1], :]

        hiddens = []
        for blk in self.blocks:
            x = blk(x)
            if return_hiddens:
                hiddens.append(x)
        x = self.norm(x)

        # For finetuning: prepend a mean-pool "CLS" row so token layout is
        # (B, T+1, D) — same as CheapSensorMAE.
        cls = x.mean(dim=1, keepdim=True)
        x = torch.cat([cls, x], dim=1)

        mask = torch.zeros(x.shape[0], self.num_patches, device=x.device)

        private_embedding = x * self.private_mask
        shared_embedding = x * (1 - self.private_mask)

        if return_hiddens:
            return private_embedding, shared_embedding, mask, hiddens
        return private_embedding, shared_embedding, mask


# ---------------------------------------------------------------------------
# Next-patch prediction head
# ---------------------------------------------------------------------------
class NextPatchPredictor(nn.Module):
    """Linear head that maps each hidden state to the next patch's raw values."""

    def __init__(self, embed_dim: int, window_len: int, in_chans: int = 1):
        super().__init__()
        self.head = nn.Linear(embed_dim, window_len * in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — causal hidden states.  Returns (B, T, window_len)."""
        return self.head(x)


# ---------------------------------------------------------------------------
# Full autoregressive model
# ---------------------------------------------------------------------------
class AutoregressiveModel(nn.Module):
    """
    GPT-style next-patch prediction model for a single modality.

    Pretraining loss: predict patch_{t+1} from the hidden state at position t.

    Exposes the same finetuning API as CheapSensorMAE:
      propose_masking, forward_encoder, num_patches, modality_name, encoder
    """

    def __init__(self, modality_name: str, sig_len: int = 3840,
                 window_len: int = 96, in_chans: int = 1,
                 embed_dim: int = 1024, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, norm_layer=nn.LayerNorm,
                 private_mask_ratio: float = 0.5, dropout: float = 0.0,
                 # Ignored kwargs for API compatibility
                 decoder_embed_dim=512, decoder_depth=4,
                 decoder_num_heads=16, decoder_mlp_ratio=4.0,
                 norm_pix_loss=True):
        super().__init__()
        print(f"Initializing AutoregressiveModel for {modality_name}")
        self.modality_name = modality_name
        self.window_len = window_len
        self.num_patches = sig_len // window_len
        self.norm_pix_loss = norm_pix_loss if norm_pix_loss is not None else True

        self.encoder = AutoregressiveEncoder(
            sig_len=sig_len, window_len=window_len, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
            private_mask_ratio=private_mask_ratio,
        )
        self.predictor = NextPatchPredictor(embed_dim, window_len, in_chans)

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

    # ---- Autoregressive pretraining ----
    def patchify(self, sigs):
        p = self.window_len
        l = sigs.shape[-1] // p
        x = sigs.reshape(sigs.shape[0], 1, l, p)
        x = torch.einsum("nchp->nhpc", x)
        return x.reshape(sigs.shape[0], l, p)

    def next_patch_loss(self, sigs: torch.Tensor) -> torch.Tensor:
        """
        Compute next-patch prediction loss.

        For a sequence of T patches, the model predicts patch_{t+1} from
        hidden state h_t (for t = 0 .. T-2).  Loss is MSE averaged over
        all predicted patches.
        """
        # Encode (without the prepended CLS token)
        x = self.encoder.patch_embed(sigs)
        x = x + self.encoder.pos_embed[:, :x.shape[1], :]
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)  # (B, T, D)

        # Predict next patch from each position
        pred = self.predictor(x)  # (B, T, window_len)

        # Target: the actual next patch
        target = self.patchify(sigs)  # (B, T, window_len)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        # Shift: predict[t] should match target[t+1]
        pred = pred[:, :-1, :]     # (B, T-1, window_len)
        target = target[:, 1:, :]  # (B, T-1, window_len)

        loss = F.mse_loss(pred, target)
        return loss
