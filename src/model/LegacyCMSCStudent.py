import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyCMSCStudent(nn.Module):
    """
    Adapter model for legacy pretrain_baselines/cmsc_wesad checkpoints.

    This model matches legacy state_dict keys:
      - net.0.weight / net.2.weight
      - out.1.weight / out.3.weight

    and exposes the KD-compatible API used in train_kd.py.
    """

    supports_reconstruction = False

    def __init__(
        self,
        modality_name: str,
        sig_len: int = 3840,
        hidden_dim: int = 128,
        conv1_kernel: int = 7,
        conv2_kernel: int = 5,
        target_tokens: int = 240,
        virtual_depth: int = 8,
        private_mask_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.sig_len = sig_len
        self.hidden_dim = hidden_dim
        self.target_tokens = max(1, int(target_tokens))
        self.num_patches = self.target_tokens
        self.virtual_depth = max(1, int(virtual_depth))
        self.private_mask_ratio = private_mask_ratio

        # Names/indices are intentionally aligned to legacy checkpoints.
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=conv1_kernel, stride=1, padding=0),  # net.0
            nn.ReLU(inplace=True),                                                     # net.1
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv2_kernel, stride=1, padding=0),  # net.2
            nn.ReLU(inplace=True),                                                     # net.3
        )

        self.out = nn.Sequential(
            nn.Flatten(start_dim=1),                                          # out.0
            nn.Linear(hidden_dim * self.target_tokens, hidden_dim),           # out.1
            nn.ReLU(inplace=True),                                            # out.2
            nn.Linear(hidden_dim, hidden_dim),                                # out.3
        )

    def propose_masking(self, batch_size, num_patches, mask_ratio, devic):
        len_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=devic)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_shuffle, ids_restore, ids_keep

    def _to_tokens(self, x_conv: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> adaptive tokens (B, T, C)
        pooled = F.adaptive_avg_pool1d(x_conv, self.target_tokens)
        return pooled.transpose(1, 2).contiguous()

    @staticmethod
    def _prepend_cls(tokens: torch.Tensor) -> torch.Tensor:
        cls = torch.zeros(tokens.size(0), 1, tokens.size(-1), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([cls, tokens], dim=1)

    def encoder(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep, return_hiddens=False):
        # Two conv stages (legacy "net")
        h1 = self.net[1](self.net[0](x))
        h2 = self.net[3](self.net[2](h1))

        # Convert conv maps into token sequences.
        t1 = self._to_tokens(h1)
        t2 = self._to_tokens(h2)

        # Legacy checkpoints expect out.1 input dim = hidden_dim * target_tokens.
        # Pool first so flattened dim matches loaded linear weights.
        h2_pooled = F.adaptive_avg_pool1d(h2, self.target_tokens)
        # Legacy "out" head gives one embedding vector; broadcast across tokens
        # so downstream KD code (token-level + pooled losses) can reuse it.
        emb = self.out[3](self.out[2](self.out[1](self.out[0](h2_pooled))))
        t_emb = emb.unsqueeze(1).expand(-1, self.target_tokens, -1)

        shared = self._prepend_cls(t_emb)
        private = self._prepend_cls(t_emb)
        mask = torch.zeros(x.size(0), self.target_tokens, device=x.device, dtype=x.dtype)

        if not return_hiddens:
            return private, shared, mask

        h1_cls = self._prepend_cls(t1)
        h2_cls = self._prepend_cls(t2)
        h3_cls = self._prepend_cls(t_emb)

        # Provide a configurable virtual depth so layer indices like 3,5,7 remain valid.
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
        raise NotImplementedError("LegacyCMSCStudent has no decoder; use --recon_weight 0.0.")

    def reconstruction_loss(self, x, pred, mask):
        raise NotImplementedError("LegacyCMSCStudent has no reconstruction loss; use --recon_weight 0.0.")
