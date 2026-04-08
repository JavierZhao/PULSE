# Copyright (c) 2026 Futurewei Technologies, Inc.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class LegacyMultiMAEStudent(nn.Module):
    """
    Adapter model for legacy pretrain_baselines/multimae_wesad checkpoints.

    This model matches legacy per-modality encoder state_dict keys:
      - 0.weight / 0.bias
      - 2.weight / 2.bias
      - 6.weight / 6.bias

    and exposes the KD-compatible API used in train_kd.py.
    """

    supports_reconstruction = False

    def __init__(
        self,
        modality_name: str,
        sig_len: int = 3840,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        conv1_kernel: int = 7,
        conv2_kernel: int = 5,
        target_tokens: int = 480,
        virtual_depth: int = 8,
        private_mask_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.sig_len = sig_len
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.target_tokens = max(1, int(target_tokens))
        self.num_patches = self.target_tokens
        self.virtual_depth = max(1, int(virtual_depth))
        self.private_mask_ratio = private_mask_ratio

        # Names/indices are intentionally aligned to legacy checkpoints.
        self.add_module(
            "0",
            nn.Conv1d(1, self.hidden_dim, kernel_size=conv1_kernel, stride=1, padding=conv1_kernel // 2),
        )
        self.add_module("1", nn.GELU())
        self.add_module(
            "2",
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=conv2_kernel,
                stride=1,
                padding=conv2_kernel // 2,
            ),
        )
        self.add_module("3", nn.GELU())
        self.add_module("4", nn.AdaptiveAvgPool1d(self.target_tokens))
        self.add_module("5", nn.Flatten(start_dim=1))
        self.add_module("6", nn.Linear(self.hidden_dim * self.target_tokens, self.latent_dim))

    def propose_masking(self, batch_size, num_patches, mask_ratio, devic):
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=devic)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_shuffle, ids_restore, ids_keep

    @staticmethod
    def _prepend_cls(tokens: torch.Tensor) -> torch.Tensor:
        cls = torch.zeros(tokens.size(0), 1, tokens.size(-1), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([cls, tokens], dim=1)

    @staticmethod
    def _align_last_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
        d = tokens.size(-1)
        if d == target_dim:
            return tokens
        if d > target_dim:
            return tokens[..., :target_dim]
        pad = torch.zeros(*tokens.shape[:-1], target_dim - d, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=-1)

    def encoder(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep, return_hiddens=False):
        conv1 = self._modules["1"](self._modules["0"](x))
        conv2 = self._modules["3"](self._modules["2"](conv1))

        # Convert conv feature maps into token sequences.
        t1 = self._modules["4"](conv1).transpose(1, 2).contiguous()
        t2 = self._modules["4"](conv2).transpose(1, 2).contiguous()

        pooled = self._modules["4"](conv2)
        flat = self._modules["5"](pooled)
        emb = self._modules["6"](flat)

        # Legacy model outputs one latent vector; broadcast over tokens for KD losses.
        t_emb = emb.unsqueeze(1).expand(-1, self.target_tokens, -1)
        t_emb = self._align_last_dim(t_emb, self.hidden_dim)

        shared = self._prepend_cls(t_emb)
        private = self._prepend_cls(t_emb)
        mask = torch.zeros(x.size(0), self.target_tokens, device=x.device, dtype=x.dtype)

        if not return_hiddens:
            return private, shared, mask

        h1_cls = self._prepend_cls(t1)
        h2_cls = self._prepend_cls(t2)
        h3_cls = self._prepend_cls(t_emb)

        hiddens = []
        b1 = max(1, self.virtual_depth // 3)
        b2 = max(b1 + 1, (2 * self.virtual_depth) // 3)
        for i in range(self.virtual_depth):
            if i < b1:
                hiddens.append(h1_cls)
            elif i < b2:
                hiddens.append(h2_cls)
            else:
                hiddens.append(h3_cls)

        return private, shared, mask, hiddens

    def forward_encoder(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep):
        private, shared, mask = self.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep, return_hiddens=False)
        return private, shared, mask

    def forward_decoder(self, private, shared, ids_restore):
        raise NotImplementedError("LegacyMultiMAEStudent has no decoder; use --recon_weight 0.0.")

    def reconstruction_loss(self, x, pred, mask):
        raise NotImplementedError("LegacyMultiMAEStudent has no reconstruction loss; use --recon_weight 0.0.")
