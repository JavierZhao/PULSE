import argparse
import os
import sys
import math
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset
from src.utils import plot_kd_losses


def setup_logging(run_name, output_path):
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)


def save_checkpoint(epoch, students, kd_heads, optimizer, scheduler, best_val_loss, losses, path, extra_state=None):
    state = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'losses': losses,
        'kd_heads_state_dict': kd_heads.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }
    for name, model in students.items():
        state[f'{name}_model_state_dict'] = model.state_dict()
    if extra_state is not None:
        state.update(extra_state)
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")


def save_students_only(students, path, extra_state=None):
    """
    Save a checkpoint that contains only the student model weights, suitable for direct finetuning.
    Structure: { '<modality>_model_state_dict': state_dict, ... } plus any optional extra_state.
    """
    state = {}
    for name, model in students.items():
        state[f'{name}_model_state_dict'] = model.state_dict()
    if extra_state is not None:
        state.update(extra_state)
    torch.save(state, path)
    logging.info(f"Students-only checkpoint saved to {path}")


def load_checkpoint(path, students, kd_heads, optimizer, scheduler, device):
    if not os.path.isfile(path):
        logging.warning(f"Checkpoint file not found at {path}. Starting from scratch.")
        return 0, float('inf'), []

    checkpoint = torch.load(path, map_location=device)
    for name, model in students.items():
        model.load_state_dict(checkpoint[f'{name}_model_state_dict'])
    # Allow for backward-compatibility if KDHeads structure changed (e.g., removed teacher projector)
    missing, unexpected = kd_heads.load_state_dict(checkpoint['kd_heads_state_dict'], strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        logging.warning(f"KDHeads state_dict loaded with missing keys: {missing} and unexpected keys: {unexpected}")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    losses = checkpoint.get('losses', [])
    logging.info(f"Resumed from epoch {epoch}. Best validation loss was {best_val_loss:.4f}.")
    return epoch, best_val_loss, losses


def linear_time_align(tokens, target_T):
    # tokens: (B, T, C) -> align along T to target_T using linear interpolation
    if tokens.size(1) == target_T:
        return tokens
    return F.interpolate(tokens.transpose(1, 2), size=target_T, mode='linear', align_corners=False).transpose(1, 2)


def hinge_token_loss(a, b, mask=None, alpha: float = 0.2):
    # a,b: (B, T, d)
    # Hinge-style ranking loss across the batch at each token position t
    # Encourages matching student-teacher pairs to be higher than mismatched pairs by margin alpha
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # Optional simple masked positive-only hinge (fallback); not used in current training flow
    if mask is not None:
        pos_sim = (a * b).sum(dim=-1)  # (B, T)
        loss = F.relu(alpha - pos_sim)
        loss = loss * mask  # mask shape (B, T)
        denom = mask.sum() + 1e-6
        return loss.sum() / denom

    # Compute per-token similarity matrices across the batch
    # Shape transform to (T, B, d)
    a_t = a.transpose(0, 1)
    b_t = b.transpose(0, 1)

    # (T, B, B): for each token index t, similarities between all batch pairs
    sim_mats = torch.bmm(a_t, b_t.transpose(1, 2))
    # Diagonal sims for each t and batch index
    diag = sim_mats.diagonal(dim1=1, dim2=2)  # (T, B)

    # Hinge costs in both directions
    cost_1 = F.relu(alpha - sim_mats + diag.unsqueeze(2))  # (T, B, B)
    cost_2 = F.relu(alpha - sim_mats.transpose(1, 2) + diag.unsqueeze(2))  # (T, B, B)

    # Zero out diagonal entries (do not count correct pairs as negatives)
    B = a.size(0)
    I = torch.eye(B, device=a.device, dtype=torch.bool)
    cost_1 = cost_1.masked_fill(I.unsqueeze(0), 0.0)
    cost_2 = cost_2.masked_fill(I.unsqueeze(0), 0.0)

    # Sum over negatives and average by batch size, then average across tokens
    total = (cost_1.sum(dim=(1, 2)) + cost_2.sum(dim=(1, 2))) / float(B)
    return total.mean()


def hinge_batch_loss(a, b, alpha: float = 0.2):
    # a,b: (B, d)
    # Hinge-style ranking loss across the batch
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    # Similarity matrix and diagonal of positives
    sim = torch.matmul(a, b.t())  # (B, B)
    diag = sim.diag()  # (B,)

    # Hinge costs in both directions
    cost_1 = F.relu(alpha - sim + diag.view(-1, 1))
    cost_2 = F.relu(alpha - sim.t() + diag.view(-1, 1))

    # Zero out diagonal entries
    B = a.size(0)
    I = torch.eye(B, device=a.device, dtype=torch.bool)
    cost_1 = cost_1.masked_fill(I, 0.0)
    cost_2 = cost_2.masked_fill(I, 0.0)

    # Sum over negatives and average by batch size
    return (cost_1.sum() + cost_2.sum()) / float(B)


def cosine_token_loss(a, b, mask=None):
    # Simple cosine similarity loss: 1 - mean cosine between matching pairs
    # a,b: (B, T, d)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    pos_sim = (a * b).sum(dim=-1)  # (B, T)
    if mask is not None:
        denom = mask.sum() + 1e-6
        mean_sim = (pos_sim * mask).sum() / denom
    else:
        mean_sim = pos_sim.mean()
    return 1.0 - mean_sim


def cosine_batch_loss(a, b):
    # Simple cosine similarity loss for pooled embeddings: 1 - mean cosine
    # a,b: (B, d)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    pos_sim = (a * b).sum(dim=-1)  # (B,)
    return 1.0 - pos_sim.mean()


def cross_covariance_loss(shared, private):
    # shared: (B, T, d_s), private: (B, T, d_p)
    B, T, d_s = shared.shape
    d_p = private.size(-1)
    s = shared.reshape(B * T, d_s)
    p = private.reshape(B * T, d_p)
    s = s - s.mean(dim=0, keepdim=True)
    p = p - p.mean(dim=0, keepdim=True)
    cov = (s.t() @ p) / (s.size(0) - 1.0)  # (d_s, d_p)
    return (cov.pow(2).sum())


class KDHeads(nn.Module):
    def __init__(self, student_dim, teacher_dim, shared_dim, private_dim, modalities):
        super().__init__()
        self.shared_projector = nn.ModuleDict({m: nn.Linear(student_dim, shared_dim, bias=False) for m in modalities})
        self.private_projector = nn.ModuleDict({m: nn.Linear(student_dim, private_dim, bias=False) for m in modalities})
        self.fuse_projector = nn.Linear(len(modalities) * shared_dim, shared_dim, bias=False)
        self.pre_ln_student = nn.LayerNorm(student_dim)
        self.pre_ln_teacher = nn.LayerNorm(teacher_dim)

        # Initialize fuse_projector as averaging across modalities: concat([I, I, ..., I]) / M
        num_modalities = max(1, len(modalities))
        with torch.no_grad():
            eye = torch.eye(shared_dim)
            weight = torch.cat([eye for _ in range(num_modalities)], dim=1) / float(num_modalities)
            self.fuse_projector.weight.copy_(weight)

    def project_student(self, hidden_tokens_by_mod):
        # hidden_tokens_by_mod[m] = (B, T, D)
        S = {}
        P = {}
        for m, H in hidden_tokens_by_mod.items():
            Hn = self.pre_ln_student(H)
            S[m] = self.shared_projector[m](Hn)
            P[m] = self.private_projector[m](Hn)
        return S, P

    def project_teacher(self, teacher_tokens):
        # teacher_tokens: (B, T_t, D_t)
        # No projector: assume teacher_dim == shared_dim; only apply LayerNorm
        return self.pre_ln_teacher(teacher_tokens)

    def fuse_shared(self, S_dict):
        # Concat along last dim then linear fuse
        cat = torch.cat([S_dict[m] for m in sorted(S_dict.keys())], dim=-1)
        return self.fuse_projector(cat)


def get_hidden_and_final_tokens(model, x, mask_ratio, device):
    # Single pass: returns (hidden tokens per block w/o CLS, final shared tokens w/o CLS, final private tokens w/o CLS)
    ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(x), model.num_patches, mask_ratio, device)
    private, shared, mask, hiddens = model.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep, return_hiddens=True)
    tokens_list = [h[:, 1:, :] for h in hiddens]
    final_shared = shared[:, 1:, :]
    final_private = private[:, 1:, :]
    return tokens_list, final_shared, final_private


def map_layers(student_layers, teacher_depth):
    # Map student layer indices to teacher indices proportionally if depths differ
    # Assumes student depth = max(student_layers)+1 approximately
    if len(student_layers) == 0:
        return {}
    student_depth = max(student_layers) + 1
    mapping = {}
    for l in student_layers:
        t = int(round((l + 1) * teacher_depth / student_depth)) - 1
        t = max(0, min(teacher_depth - 1, t))
        mapping[l] = t
    return mapping


@torch.no_grad()
def validate(batch_iter, students, teacher, kd_heads, layers_to_match, layer_map, device, args):
    for model in students.values():
        model.eval()
    teacher.eval()

    total_kd_hid = 0.0
    total_kd_emb = 0.0
    total_recon = 0.0
    total_perp = 0.0
    count = 0

    for batch in batch_iter:
        ecg = batch['ecg'].to(device)
        bvp = batch['bvp'].to(device)
        acc = batch['acc'].to(device)
        temp = batch['temp'].to(device)
        eda = batch['eda'].to(device)

        # Forward teacher and students once for hidden and final tokens
        Ht_list, _, T_final_private = get_hidden_and_final_tokens(teacher, eda, mask_ratio=0.0, device=device)
        # Build per-batch teacher->student layer mapping if not provided
        teacher_depth = len(Ht_list)
        if layer_map is None:
            if len(layers_to_match) > 0:
                student_depth = max(layers_to_match) + 1
                layer_map = {}
                for l in layers_to_match:
                    t = int(round((l + 1) * teacher_depth / student_depth)) - 1
                    t = max(0, min(teacher_depth - 1, t))
                    layer_map[l] = t
            else:
                layer_map = {}
        Hs_hidden = {}
        Hs_final_shared = {}
        Hs_final_private = {}
        for name, x in [('ecg', ecg), ('bvp', bvp), ('acc', acc), ('temp', temp)]:
            hs, hf_shared, hf_private = get_hidden_and_final_tokens(students[name], x, mask_ratio=0.0, device=device)
            Hs_hidden[name] = hs
            Hs_final_shared[name] = hf_shared
            Hs_final_private[name] = hf_private

        L_hid_fuse = torch.tensor(0.0, device=device)
        L_emb_fuse = 0.0
        L_perp_total = 0.0

        # Select KD loss functions based on args
        if getattr(args, 'kd_loss', 'cosine') == 'hinge':
            token_kd_loss_fn = hinge_token_loss
            batch_kd_loss_fn = hinge_batch_loss
        else:
            token_kd_loss_fn = cosine_token_loss
            batch_kd_loss_fn = cosine_batch_loss

        for l in layers_to_match:
            # Collect student tokens at layer l
            student_tokens = {m: Hs_hidden[m][l] for m in Hs_hidden}
            S_dict, _ = kd_heads.project_student(student_tokens)

            # Teacher tokens for mapped layer
            t_idx = layer_map[l]
            T_tokens = kd_heads.project_teacher(Ht_list[t_idx])

            # Align time length
            T_tokens = linear_time_align(T_tokens, target_T=S_dict['ecg'].size(1))

            # Fuse
            S_fuse = kd_heads.fuse_shared(S_dict)

            # Token-level KD
            L_hid_fuse = L_hid_fuse + token_kd_loss_fn(S_fuse, T_tokens)

        # Normalize hidden KD over number of matched layers
        L_hid_fuse = L_hid_fuse / max(1, len(layers_to_match))

        # Final-layer pooled embedding KD
        S_final_dict, _ = kd_heads.project_student(Hs_final_shared)
        S_final_fuse = kd_heads.fuse_shared(S_final_dict)
        T_final_proj = kd_heads.project_teacher(T_final_private)
        s = S_final_fuse.mean(dim=1)
        t = T_final_proj.mean(dim=1)
        L_emb_fuse = batch_kd_loss_fn(s, t)

        # Optional decorrelation on last matched hidden layer (validation)
        if args.perp_weight > 0.0 and len(layers_to_match) > 0:
            last_l = layers_to_match[-1]
            student_tokens_last = {m: Hs_hidden[m][last_l] for m in Hs_hidden}  # (B,T,D)
            S_last, P_last = kd_heads.project_student(student_tokens_last)       # (B,T,d_s)/(B,T,d_p)
            for m in S_last.keys():
                L_perp_total = L_perp_total + cross_covariance_loss(S_last[m], P_last[m])

        # Optional reconstruction
        if args.recon_weight > 0.0:
            recon_loss = 0.0
            for name, x in [('ecg', ecg), ('bvp', bvp), ('acc', acc), ('temp', temp)]:
                ids_shuffle, ids_restore, ids_keep = students[name].propose_masking(len(x), students[name].num_patches, args.mask_ratio, device)
                private, shared, mask = students[name].forward_encoder(x, args.mask_ratio, ids_shuffle, ids_restore, ids_keep)
                rec = students[name].forward_decoder(private, shared, ids_restore)
                recon_loss = recon_loss + students[name].reconstruction_loss(x, rec, mask)
        else:
            recon_loss = torch.tensor(0.0, device=device)

        total_kd_hid += L_hid_fuse.item()
        total_kd_emb += L_emb_fuse.item()
        total_recon += recon_loss.item()
        total_perp += (L_perp_total.item() if isinstance(L_perp_total, torch.Tensor) else L_perp_total)
        count += 1

    return {
        'kd_hid': total_kd_hid / max(1, count),
        'kd_emb': total_kd_emb / max(1, count),
        'recon': total_recon / max(1, count),
        'perp': total_perp / max(1, count),
        'total': args.lambda_hid * (total_kd_hid / max(1, count)) + args.lambda_emb * (total_kd_emb / max(1, count)) + args.recon_weight * (total_recon / max(1, count)) + args.perp_weight * (total_perp / max(1, count)),
    }


def train(args):
    device = torch.device(args.device)

    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)

    # Instantiate teacher and students
    teacher = CheapSensorMAE(
        modality_name='eda',
        sig_len=args.signal_length,
        window_len=args.patch_window_len,
        private_mask_ratio=1.0,
    ).to(device)
    # freeze the teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    students = {
        'ecg': CheapSensorMAE(modality_name='ecg', sig_len=args.signal_length, window_len=args.patch_window_len,
                              embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                              decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                              mlp_ratio=args.mlp_ratio, decoder_mlp_ratio=args.decoder_mlp_ratio, private_mask_ratio=args.private_mask_ratio).to(device),
        'bvp': CheapSensorMAE(modality_name='bvp', sig_len=args.signal_length, window_len=args.patch_window_len,
                              embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                              decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                              mlp_ratio=args.mlp_ratio, decoder_mlp_ratio=args.decoder_mlp_ratio, private_mask_ratio=args.private_mask_ratio).to(device),
        'acc': CheapSensorMAE(modality_name='acc', sig_len=args.signal_length, window_len=args.patch_window_len,
                              embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                              decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                              mlp_ratio=args.mlp_ratio, decoder_mlp_ratio=args.decoder_mlp_ratio, private_mask_ratio=args.private_mask_ratio).to(device),
        'temp': CheapSensorMAE(modality_name='temp', sig_len=args.signal_length, window_len=args.patch_window_len,
                               embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                               decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                               mlp_ratio=args.mlp_ratio, decoder_mlp_ratio=args.decoder_mlp_ratio, private_mask_ratio=args.private_mask_ratio).to(device),
    }

    # Load checkpoints
    if args.teacher_ckpt_path is not None and os.path.isfile(args.teacher_ckpt_path):
        ckpt = torch.load(args.teacher_ckpt_path, map_location=device)
        key = 'model_state_dict' if 'model_state_dict' in ckpt else next((k for k in ckpt.keys() if k.endswith('model_state_dict')), None)
        if key is None:
            logging.warning("Could not find teacher model_state_dict in checkpoint. Skipping load.")
        else:
            teacher.load_state_dict(ckpt[key], strict=False)
            logging.info(f"Loaded teacher weights from {args.teacher_ckpt_path}")
    else:
        logging.warning("No valid teacher_ckpt_path provided. Using randomly initialized teacher (frozen).")

    if args.students_ckpt_path is not None and os.path.isfile(args.students_ckpt_path):
        ckpt = torch.load(args.students_ckpt_path, map_location=device)
        for name in students.keys():
            key = f'{name}_model_state_dict'
            if key in ckpt:
                students[name].load_state_dict(ckpt[key], strict=False)
                logging.info(f"Loaded student {name} weights from {args.students_ckpt_path}")
            else:
                logging.warning(f"Key {key} not found in students checkpoint.")
    else:
        logging.warning("No valid students_ckpt_path provided. Students start from init.")

    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.signal_length, device=device)
        Ht_hidden, _, _ = get_hidden_and_final_tokens(teacher, dummy, mask_ratio=0.0, device=device)
    teacher_dim = Ht_hidden[0].size(-1)
    # Resolve shared_dim to align with teacher when unset/non-positive
    resolved_shared_dim = args.shared_dim if getattr(args, 'shared_dim', -1) and args.shared_dim > 0 else int(teacher_dim)

    # KD heads
    # kd_heads = KDHeads(student_dim=args.embed_dim, teacher_dim=args.embed_dim, shared_dim=args.shared_dim, private_dim=args.private_dim, modalities=list(students.keys())).to(device)
    kd_heads = KDHeads(
        student_dim=args.embed_dim,
        teacher_dim=teacher_dim,
        shared_dim=resolved_shared_dim,
        private_dim=args.private_dim,
        modalities=list(students.keys())
    ).to(device)

    # Layers to match (student indices)
    layers_to_match = sorted(args.layers_to_match)
    

    # Optimizer/scheduler
    # Freeze fusion projector initially unless unfreeze epoch is 0
    if getattr(args, 'unfreeze_fuse_projector_epoch', -1) != 0:
        for p in kd_heads.fuse_projector.parameters():
            p.requires_grad_(False)

    # Always include KD head parameters in optimizer so they can be unfrozen later
    optim_params = list(kd_heads.parameters())
    for model in students.values():
        optim_params += list(model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs) if args.use_cosine else None

    # Data
    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train')
    if args.dataset_percentage < 1.0:
        original_size = len(full_train_dataset)
        subset_size = int(original_size * args.dataset_percentage)
        generator = torch.Generator().manual_seed(args.seed)
        full_train_dataset, _ = random_split(full_train_dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {args.dataset_percentage*100:.0f}% of the dataset: {len(full_train_dataset)} out of {original_size} windows.")

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Logging configuration
    logging.info("--- Starting KD Training Run ---")
    logging.info(f"Run Name: {args.run_name}")
    logging.info("Configuration:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")
    logging.info("\nModel Architecture:")
    logging.info(str(kd_heads))
    logging.info(f"Total parameters: {sum(p.numel() for p in kd_heads.parameters()):,}")
    logging.info("-" * 50)

    start_epoch, best_val_loss, losses = 0, float('inf'), []
    if args.resume_from:
        start_epoch, best_val_loss, losses = load_checkpoint(args.resume_from, students, kd_heads, optimizer, scheduler, device)

    for epoch in range(start_epoch, args.num_epochs):
        # Optionally unfreeze fusion projector at the specified epoch (1-based)
        if getattr(args, 'unfreeze_fuse_projector_epoch', -1) == (epoch + 1):
            for p in kd_heads.fuse_projector.parameters():
                p.requires_grad_(True)
            logging.info(f"Unfroze fusion projector at epoch {epoch+1}.")

        for model in students.values():
            model.train()
        teacher.eval()

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        epoch_kd_hid = 0.0
        epoch_kd_emb = 0.0
        epoch_recon = 0.0
        num_steps = 0
        logged_variance = False

        for batch in pbar:
            ecg = batch['ecg'].to(device)
            bvp = batch['bvp'].to(device)
            acc = batch['acc'].to(device)
            temp = batch['temp'].to(device)
            eda = batch['eda'].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                Ht_list, _, T_final_private = get_hidden_and_final_tokens(teacher, eda, mask_ratio=0.0, device=device)
                # Build per-batch teacher->student layer mapping based on actual teacher depth
                teacher_depth = len(Ht_list)
                if len(layers_to_match) > 0:
                    student_depth = max(layers_to_match) + 1
                    layer_map = {}
                    for l in layers_to_match:
                        t = int(round((l + 1) * teacher_depth / student_depth)) - 1
                        t = max(0, min(teacher_depth - 1, t))
                        layer_map[l] = t
                else:
                    layer_map = {}

            Hs_hidden = {}
            Hs_final_shared = {}
            Hs_final_private = {}
            for name, x in [('ecg', ecg), ('bvp', bvp), ('acc', acc), ('temp', temp)]:
                hs, hf_shared, hf_private = get_hidden_and_final_tokens(students[name], x, mask_ratio=0.0, device=device)
                Hs_hidden[name] = hs
                Hs_final_shared[name] = hf_shared
                Hs_final_private[name] = hf_private

            L_hid_fuse = 0.0
            L_perp_total = 0.0

            # Select KD loss functions based on args
            if getattr(args, 'kd_loss', 'cosine') == 'hinge':
                token_kd_loss_fn = hinge_token_loss
                batch_kd_loss_fn = hinge_batch_loss
            else:
                token_kd_loss_fn = cosine_token_loss
                batch_kd_loss_fn = cosine_batch_loss

            for l in layers_to_match:
                student_tokens = {m: Hs_hidden[m][l] for m in Hs_hidden}  # (B, T, D)
                S_dict, _ = kd_heads.project_student(student_tokens)

                
                t_idx = layer_map[l]
                T_tokens = kd_heads.project_teacher(Ht_list[t_idx])
                T_tokens = linear_time_align(T_tokens, target_T=S_dict['ecg'].size(1))

                # Fusion
                S_fuse = kd_heads.fuse_shared(S_dict)

                # KD losses
                L_hid_fuse = L_hid_fuse + token_kd_loss_fn(S_fuse, T_tokens)

                # Quick diagnostic variance checks (log once per epoch)
                if not logged_variance:
                    with torch.no_grad():
                        t_std = T_tokens.reshape(-1, T_tokens.size(-1)).std(dim=0).mean().item()
                        s_std = S_fuse.reshape(-1, S_fuse.size(-1)).std(dim=0).mean().item()
                    logging.info(f"Teacher token mean-std: {t_std:.4f} | Student fused token mean-std: {s_std:.4f}")
                    logged_variance = True
            
            # Normalize hidden KD over number of matched layers
            L_hid_fuse = L_hid_fuse / max(1, len(layers_to_match))

            # Optional decorrelation on last matched hidden layer
            if args.perp_weight > 0.0:
                 last_l = layers_to_match[-1]
                 student_tokens_last = {m: Hs_hidden[m][last_l] for m in Hs_hidden}   # (B,T,D)
                 S_last, P_last = kd_heads.project_student(student_tokens_last)        # (B,T,d_s)/(B,T,d_p)
                 for m in S_last.keys():
                     L_perp_total += cross_covariance_loss(S_last[m], P_last[m])

            # Optional reconstruction
            if args.recon_weight > 0.0:
                recon_loss = 0.0
                for name, x in [('ecg', ecg), ('bvp', bvp), ('acc', acc), ('temp', temp)]:
                    ids_shuffle, ids_restore, ids_keep = students[name].propose_masking(len(x), students[name].num_patches, args.mask_ratio, device)
                    private, shared, mask = students[name].forward_encoder(x, args.mask_ratio, ids_shuffle, ids_restore, ids_keep)
                    rec = students[name].forward_decoder(private, shared, ids_restore)
                    recon_loss = recon_loss + students[name].reconstruction_loss(x, rec, mask)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            # Final pooled KD using previously computed final shared tokens (no extra forward)
            S_final_dict, _ = kd_heads.project_student(Hs_final_shared)
            S_final_fuse = kd_heads.fuse_shared(S_final_dict)
            T_final_proj = kd_heads.project_teacher(T_final_private)
            s = S_final_fuse.mean(dim=1)
            t = T_final_proj.mean(dim=1)
            L_emb_final = batch_kd_loss_fn(s, t)

            # If no hidden layers were matched, still log variance diagnostics once per epoch using final tokens
            if not logged_variance and len(layers_to_match) == 0:
                with torch.no_grad():
                    t_std = T_final_proj.reshape(-1, T_final_proj.size(-1)).std(dim=0).mean().item()
                    s_std = S_final_fuse.reshape(-1, S_final_fuse.size(-1)).std(dim=0).mean().item()
                logging.info(f"Teacher token mean-std: {t_std:.4f} | Student fused token mean-std: {s_std:.4f}")
                logged_variance = True

            loss = args.lambda_hid * L_hid_fuse + args.lambda_emb * L_emb_final + args.recon_weight * recon_loss + args.perp_weight * L_perp_total

            loss.backward()
            optimizer.step()

            epoch_kd_hid += L_hid_fuse.item()
            epoch_kd_emb += L_emb_final.item()
            epoch_recon += recon_loss.item()
            num_steps += 1

            pbar.set_postfix({
                'KD_hid': f"{L_hid_fuse.item():.4f}",
                'KD_emb': f"{L_emb_final.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
            })

        # Validation (recompute mapping inside validate per batch as well)
        val_metrics = validate(val_loader, students, teacher, kd_heads, layers_to_match, None, device, args)

        if scheduler is not None:
            scheduler.step()

        avg_kd_hid = epoch_kd_hid / max(1, num_steps)
        avg_kd_emb = epoch_kd_emb / max(1, num_steps)
        avg_recon = epoch_recon / max(1, num_steps)
        # Note: train_perp is reflected in the batch loss and optimizer step via L_perp_total, but not tracked separately per-step.
        # To report it, we can recompute it during validation (already reported as val_perp) or track it here similarly.
        train_total = args.lambda_hid * avg_kd_hid + args.lambda_emb * avg_kd_emb + args.recon_weight * avg_recon

        logging.info(
            f"Epoch {epoch+1}/{args.num_epochs} | Train: total={train_total:.4f} (hid={avg_kd_hid:.4f}, emb={avg_kd_emb:.4f}, recon={avg_recon:.4f}) | "
            f"Val: total={val_metrics['total']:.4f} (hid={val_metrics['kd_hid']:.4f}, emb={val_metrics['kd_emb']:.4f}, recon={val_metrics['recon']:.4f}, perp={val_metrics.get('perp', 0.0):.4f})"
        )

        # Save loss history
        losses.append({
            'epoch': epoch + 1,
            'train_total': train_total,
            'train_kd_hid': avg_kd_hid,
            'train_kd_emb': avg_kd_emb,
            'train_recon': avg_recon,
            'val_total': val_metrics['total'],
            'val_kd_hid': val_metrics['kd_hid'],
            'val_kd_emb': val_metrics['kd_emb'],
            'val_recon': val_metrics['recon'],
            'val_perp': val_metrics.get('perp', 0.0),
            'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        })

        # Checkpointing
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            # Save full KD checkpoint under a distinct name
            best_kd_path = os.path.join(models_path, "best_ckpt_full_kd.pt")
            save_checkpoint(
                epoch + 1,
                students,
                kd_heads,
                optimizer,
                scheduler or torch.optim.lr_scheduler.LinearLR(optimizer),
                best_val_loss,
                losses,
                best_kd_path,
                extra_state={'teacher_ckpt_used': args.teacher_ckpt_path, 'students_ckpt_used': args.students_ckpt_path}
            )
            # Save students-only checkpoint for finetuning with the standard name
            students_only_path = os.path.join(models_path, "best_ckpt.pt")
            save_students_only(students, students_only_path, extra_state={'epoch': epoch + 1})

        # Plot losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_path = os.path.join(run_output_path, f"kd_loss_curves.png")
            df = pd.DataFrame(losses)
            plot_kd_losses(df, plot_path)
            logging.info(f"Saved KD loss plot to {plot_path}")

    logging.info("Training complete.")


def parse_layers(s):
    # e.g., "3,5,7" -> [3,5,7]
    if s is None or len(s.strip()) == 0:
        return [3, 5, 7]
    return [int(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation: EDA teacher -> 4 CheapSensorMAE students')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--data_path', type=str, default="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s")
    parser.add_argument('--output_path', type=str, default="/fd24T/zzhao3/EDA/results/kd")
    parser.add_argument('--fold_number', type=int, default=17)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_percentage', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model sizes
    parser.add_argument('--signal_length', type=int, default=3840)
    parser.add_argument('--patch_window_len', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--decoder_mlp_ratio', type=float, default=4.0)
    parser.add_argument('--private_mask_ratio', type=float, default=0.5)

    # KD head dims
    # shared_dim: default 0 means "match teacher dim"; we will resolve at runtime
    parser.add_argument('--shared_dim', type=int, default=0)
    parser.add_argument('--private_dim', type=int, default=256)

    # Loss weights
    parser.add_argument('--lambda_hid', type=float, default=1.0)
    parser.add_argument('--lambda_emb', type=float, default=1.0)
    parser.add_argument('--perp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    # KD loss selection
    parser.add_argument('--kd_loss', type=str, choices=['cosine', 'hinge'], default='cosine', help='KD loss type: cosine similarity (1 - cos) or hinge ranking')

    # Layers
    parser.add_argument('--layers_to_match', type=parse_layers, default=[3, 5, 7])

    # Optim
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--use_cosine', action='store_true')

    # Training behavior
    parser.add_argument('--unfreeze_fuse_projector_epoch', type=int, default=-1, help='Epoch at which to unfreeze the fusion projector (1-based). -1 means never unfreeze; 0 means start unfrozen.')

    # Checkpoints
    parser.add_argument('--teacher_ckpt_path', type=str, default="/fd24T/zzhao3/EDA/results/eda_mae/300p/models/best_ckpt.pt", help='Path to teacher best_ckpt.pt from EDA MAE training')
    parser.add_argument('--students_ckpt_path', type=str, default="/fd24T/zzhao3/EDA/results/cheap_maes/hinge_loss/default/models/best_ckpt.pt", help='Path to students best_ckpt.pt from multi-modal MAE pretraining')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    setup_logging(args.run_name, args.output_path)
    train(args)


if __name__ == '__main__':
    main()


