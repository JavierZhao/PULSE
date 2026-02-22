"""
Multi-modal autoregressive (GPT-style next-patch prediction) pretraining for
physiological signals.

Pretraining strategy:
  Instead of masking random patches and reconstructing them (MAE), this
  approach uses a causal Transformer that processes patches left-to-right
  and predicts the *next* patch from the current hidden state.  This
  naturally captures the temporal / causal structure of physiological
  signals.

  Cross-modal alignment is achieved with the same hinge loss used in the
  MAE pipeline, applied to the shared embedding subspace.

Usage:
  python -m src.pretrain.train_autoregressive \
      --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.AutoregressiveTransformer import AutoregressiveModel
from src.data.wesad_dataset import WESADDataset
from src.modules.hinge_loss import AllPairsHingeLoss


def setup_logging(run_name, output_path):
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)


def save_checkpoint(epoch, models, optimizers, schedulers, best_val_loss, losses, path):
    state = {'epoch': epoch, 'best_val_loss': best_val_loss, 'losses': losses}
    for name, model in models.items():
        state[f'{name}_model_state_dict'] = model.state_dict()
    for name, opt in optimizers.items():
        state[f'{name}_optimizer_state_dict'] = opt.state_dict()
    for name, sch in schedulers.items():
        state[f'{name}_scheduler_state_dict'] = sch.state_dict()
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")


def train(args):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)

    modality_names = ['ecg', 'bvp', 'acc', 'temp']
    if args.include_eda:
        modality_names.append('eda')
    logging.info(f"Using modalities: {modality_names}")

    # --- Models ---
    models = {}
    for name in modality_names:
        models[name] = AutoregressiveModel(
            modality_name=name,
            sig_len=args.signal_length,
            window_len=args.patch_window_len,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            private_mask_ratio=args.private_mask_ratio,
            dropout=args.dropout,
        ).to(device)

    optimizers = {n: torch.optim.Adam(m.parameters(), lr=args.learning_rate)
                  for n, m in models.items()}
    schedulers = {n: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        o, T_0=args.lr_restart_epochs, T_mult=args.t_mult)
        for n, o in optimizers.items()}

    alignment_loss_fn = AllPairsHingeLoss(alpha=args.hinge_alpha)

    # --- Data ---
    full_ds = WESADDataset(data_path=args.data_path,
                           fold_number=args.fold_number, split='train')
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=gen)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logging.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    logging.info("Configuration:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    best_val_loss = float('inf')
    all_losses = []

    # --- TensorBoard ---
    tb_dir = os.path.join(run_output_path, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info(f"TensorBoard logs: {tb_dir}")

    for epoch in range(args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")

        # --- Train ---
        for m in models.values():
            m.train()
        train_pred_total, train_align_total = 0.0, 0.0

        pbar = tqdm(train_dl, desc=f"Train {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            signals = {n: batch[n].to(device) for n in modality_names}

            for opt in optimizers.values():
                opt.zero_grad()

            # Per-modality next-patch prediction loss
            pred_loss = 0.0
            all_shared_embs = {}
            for name, sig in signals.items():
                npp_loss = models[name].next_patch_loss(sig)
                pred_loss += npp_loss

                # Collect shared embeddings for alignment
                priv, shared, _ = models[name].forward_encoder(
                    sig, 0.0, None, None, None)
                all_shared_embs[name] = shared.mean(dim=1)  # pool over tokens

            pred_loss = pred_loss / len(signals)

            # Cross-modal alignment
            all_shared_norm = {k: nn.functional.normalize(v, p=2, dim=1)
                               for k, v in all_shared_embs.items()}
            if len(all_shared_norm) >= 2:
                align_loss = alignment_loss_fn(all_shared_norm)
            else:
                align_loss = torch.tensor(0.0, device=device)

            loss = pred_loss + args.alignment_loss_weight * align_loss
            loss.backward()

            for opt in optimizers.values():
                opt.step()

            frac_epoch = epoch + (batch_idx + 1) / max(1, len(train_dl))
            for sch in schedulers.values():
                sch.step(frac_epoch)

            train_pred_total += pred_loss.item()
            train_align_total += align_loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_pred = train_pred_total / len(train_dl)
        avg_train_align = train_align_total / len(train_dl)

        # --- Validate ---
        for m in models.values():
            m.eval()
        val_pred_total, val_align_total = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validating"):
                signals = {n: batch[n].to(device) for n in modality_names}
                p_loss = 0.0
                shared_embs = {}
                for name, sig in signals.items():
                    p_loss += models[name].next_patch_loss(sig)
                    _, sh, _ = models[name].forward_encoder(
                        sig, 0.0, None, None, None)
                    shared_embs[name] = sh.mean(dim=1)
                p_loss /= len(signals)
                shared_norm = {k: nn.functional.normalize(v, p=2, dim=1)
                               for k, v in shared_embs.items()}
                if len(shared_norm) >= 2:
                    al = alignment_loss_fn(shared_norm)
                else:
                    al = torch.tensor(0.0, device=device)
                val_pred_total += p_loss.item()
                val_align_total += al.item()

        avg_val_pred = val_pred_total / len(val_dl)
        avg_val_align = val_align_total / len(val_dl)
        avg_val_total = avg_val_pred + args.alignment_loss_weight * avg_val_align

        logging.info(
            f"Epoch {epoch+1} | Train Pred: {avg_train_pred:.4f} Align: {avg_train_align:.4f} | "
            f"Val Pred: {avg_val_pred:.4f} Align: {avg_val_align:.4f}"
        )

        # --- TensorBoard scalars ---
        writer.add_scalar('Loss/train_pred', avg_train_pred, epoch + 1)
        writer.add_scalar('Loss/train_align', avg_train_align, epoch + 1)
        writer.add_scalar('Loss/val_pred', avg_val_pred, epoch + 1)
        writer.add_scalar('Loss/val_align', avg_val_align, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val_total, epoch + 1)
        current_lr = next(iter(schedulers.values())).get_last_lr()[0]
        writer.add_scalar('LR', current_lr, epoch + 1)

        all_losses.append({
            'epoch': epoch + 1,
            'train_pred': avg_train_pred, 'train_align': avg_train_align,
            'val_pred': avg_val_pred, 'val_align': avg_val_align,
        })

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            save_checkpoint(epoch + 1, models, optimizers, schedulers,
                            best_val_loss, all_losses,
                            os.path.join(models_path, "best_ckpt.pt"))

    writer.close()
    logging.info("Autoregressive pretraining complete.")
    losses_df = pd.DataFrame(all_losses)
    np.savez(os.path.join(run_output_path, "losses.npz"),
             **{k: losses_df[k].values for k in losses_df.columns})


def main():
    parser = argparse.ArgumentParser(description='Autoregressive (Next-Patch Prediction) Pretraining')
    parser.add_argument('--run_name', type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S_autoregressive"))
    parser.add_argument('--data_path', type=str,
                        default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid")
    parser.add_argument('--output_path', type=str,
                        default="/j-jepa-vol/PULSE/results/autoregressive")
    parser.add_argument('--fold_number', type=int, default=17)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # Model
    parser.add_argument('--signal_length', type=int, default=3840)
    parser.add_argument('--patch_window_len', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--private_mask_ratio', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout for causal attention/MLP')
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_restart_epochs', type=int, default=20)
    parser.add_argument('--t_mult', type=int, default=1)
    parser.add_argument('--alignment_loss_weight', type=float, default=1.0)
    parser.add_argument('--hinge_alpha', type=float, default=0.2)
    parser.add_argument('--include_eda', action='store_true')
    args = parser.parse_args()

    setup_logging(args.run_name, args.output_path)
    train(args)


if __name__ == '__main__':
    main()
