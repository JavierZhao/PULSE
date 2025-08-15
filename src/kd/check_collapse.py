import argparse
import os
import sys
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root on path so we can import src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset


def infer_student_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Infer key architecture hyperparameters from a CheapSensorMAE student state_dict.
    Returns a dict with keys: embed_dim, depth, decoder_embed_dim, decoder_depth, decoder_num_heads, window_len, num_heads.
    Note: num_heads cannot be perfectly inferred from weights; default to 8 if ambiguous.
    """
    # Depth: count encoder blocks
    depth = 0
    for k in state_dict.keys():
        if k.startswith('encoder.blocks.'):
            try:
                idx = int(k.split('.')[2])
                depth = max(depth, idx + 1)
            except Exception:
                continue

    # embed_dim from encoder.norm.weight, else try a few sensible fallbacks
    if 'encoder.norm.weight' in state_dict:
        embed_dim = state_dict['encoder.norm.weight'].shape[0]
    elif 'encoder.blocks.0.norm1.weight' in state_dict:
        embed_dim = state_dict['encoder.blocks.0.norm1.weight'].shape[0]
    elif 'encoder.blocks.0.attn.proj.weight' in state_dict:  # [embed_dim, embed_dim]
        embed_dim = state_dict['encoder.blocks.0.attn.proj.weight'].shape[0]
    else:
        embed_dim = 1024  # safe default matching training script

    # decoder_embed_dim from decoder_norm or decoder_pred
    if 'decoder_norm.weight' in state_dict:
        decoder_embed_dim = state_dict['decoder_norm.weight'].shape[0]
    elif 'decoder_pred.weight' in state_dict:
        w = state_dict['decoder_pred.weight']
        decoder_embed_dim = w.shape[1] if w.dim() == 2 else 512
    else:
        decoder_embed_dim = 512  # fallback

    # decoder_depth: count decoder blocks
    decoder_depth = 0
    for k in state_dict.keys():
        if k.startswith('decoder_blocks.'):
            try:
                idx = int(k.split('.')[2])
                decoder_depth = max(decoder_depth, idx + 1)
            except Exception:
                continue

    # decoder_num_heads not directly inferable; default 8
    decoder_num_heads = 8

    # window_len: prefer Conv1d kernel size, else decoder_pred.out_features
    if 'encoder.patch_embed.proj.weight' in state_dict:
        # Conv1d weight: [embed_dim, in_chans, window_len]
        window_len = state_dict['encoder.patch_embed.proj.weight'].shape[-1]
    elif 'decoder_pred.weight' in state_dict:
        window_len = state_dict['decoder_pred.weight'].shape[0]
    else:
        window_len = 96  # fallback to training default

    # num_heads not encoded in weights; assume 8 (matches training defaults)
    num_heads = 8

    return {
        'embed_dim': int(embed_dim),
        'depth': int(depth),
        'decoder_embed_dim': int(decoder_embed_dim),
        'decoder_depth': int(decoder_depth),
        'decoder_num_heads': int(decoder_num_heads),
        'window_len': int(window_len),
        'num_heads': int(num_heads),
    }


@torch.no_grad()
def extract_final_tokens(model: CheapSensorMAE, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_shared_tokens_wo_cls, final_private_tokens_wo_cls) with shape (B, T, D).
    Uses mask_ratio=0 to avoid temporal masking during the probe.
    """
    device = x.device
    ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(x), model.num_patches, 0.0, device)
    private, shared, mask = model.encoder(x, 0.0, ids_shuffle, ids_restore, ids_keep, return_hiddens=False)
    return shared[:, 1:, :], private[:, 1:, :]


def compute_collapse_metrics(tokens: torch.Tensor) -> Dict[str, float]:
    """
    tokens: (N, T, D). Computes several statistics indicative of collapse.
    """
    N, T, D = tokens.shape
    X = tokens.reshape(N * T, D)

    # Center
    Xc = X - X.mean(dim=0, keepdim=True)

    # Per-feature variance
    feat_var = Xc.var(dim=0, unbiased=False)
    mean_feat_var = feat_var.mean().item()
    min_feat_var = feat_var.min().item()
    max_feat_var = feat_var.max().item()

    # Covariance eigen spectrum (D x D)
    cov = (Xc.t() @ Xc) / max(1, Xc.shape[0] - 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
        eigvals_sorted, _ = torch.sort(eigvals, descending=True)
        total_var = eigvals_sorted.sum().item() + 1e-12
        top1_exp_var = (eigvals_sorted[0].item() / total_var) if eigvals_sorted.numel() > 0 else 0.0
        rank_thresh = 1e-6 * eigvals_sorted.max().item() if eigvals_sorted.numel() > 0 else 0.0
        numerical_rank = int((eigvals_sorted > rank_thresh).sum().item()) if eigvals_sorted.numel() > 0 else 0
    except RuntimeError:
        top1_exp_var = float('nan')
        numerical_rank = 0

    # Sample-level pooled embeddings cosine stats
    pooled = X.mean(dim=0, keepdim=True) if T == 0 else tokens.mean(dim=1)  # (N, D)
    pooled = nn.functional.normalize(pooled, dim=-1)
    sim_mat = pooled @ pooled.t()
    # Exclude diagonal
    off_diag = sim_mat[~torch.eye(sim_mat.shape[0], dtype=torch.bool, device=sim_mat.device)]
    mean_pairwise_cos = off_diag.mean().item() if off_diag.numel() > 0 else float('nan')
    std_pairwise_cos = off_diag.std().item() if off_diag.numel() > 0 else float('nan')

    # Magnitude spread
    min_val = X.min().item()
    max_val = X.max().item()
    spread = max_val - min_val

    return {
        'mean_feature_variance': float(mean_feat_var),
        'min_feature_variance': float(min_feat_var),
        'max_feature_variance': float(max_feat_var),
        'top1_explained_variance': float(top1_exp_var),
        'numerical_rank': float(numerical_rank),
        'mean_pairwise_cosine': float(mean_pairwise_cos),
        'std_pairwise_cosine': float(std_pairwise_cos),
        'activation_spread': float(spread),
    }


def decide_collapse(metrics: Dict[str, float]) -> bool:
    """
    Heuristic collapse decision.
    Flags collapse if representations are nearly constant or rank-deficient.
    """
    mean_var = metrics['mean_feature_variance']
    top1 = metrics['top1_explained_variance']
    mean_cos = metrics['mean_pairwise_cosine']
    spread = metrics['activation_spread']
    rank_est = metrics['numerical_rank']

    rules = [
        mean_var < 1e-6,
        spread < 1e-5,
        top1 > 0.995,
        mean_cos > 0.99,
        rank_est <= 1,
    ]
    return any(rules)


def build_students_from_ckpt(ckpt: Dict[str, Dict[str, torch.Tensor]], sig_len: int, device: torch.device) -> Dict[str, CheapSensorMAE]:
    """
    Build a dict of modality -> model, using inferred config from one of the student state_dicts.
    """
    # Prefer ecg if present for config inference
    modality_key = next((k for k in ['ecg_model_state_dict', 'bvp_model_state_dict', 'acc_model_state_dict', 'temp_model_state_dict'] if k in ckpt), None)
    if modality_key is None:
        raise ValueError('No *_model_state_dict found in checkpoint. Expected students-only checkpoint.')

    inferred = infer_student_config(ckpt[modality_key])

    students = {}
    for name in ['ecg', 'bvp', 'acc', 'temp']:
        key = f'{name}_model_state_dict'
        if key not in ckpt:
            continue
        model = CheapSensorMAE(
            modality_name=name,
            sig_len=sig_len,
            window_len=inferred['window_len'],
            embed_dim=inferred['embed_dim'],
            depth=inferred['depth'],
            num_heads=inferred['num_heads'],
            decoder_embed_dim=inferred['decoder_embed_dim'],
            decoder_depth=inferred['decoder_depth'],
            decoder_num_heads=inferred['decoder_num_heads'],
            mlp_ratio=4.0,
            decoder_mlp_ratio=4.0,
            private_mask_ratio=0.5,
        ).to(device)
        # Load weights
        model.load_state_dict(ckpt[key], strict=False)
        model.eval()
        students[name] = model

    return students


def sample_signal_length(dataset: WESADDataset) -> int:
    item = dataset[0]
    # Any modality has same length; pick ecg
    return int(item['ecg'].shape[-1])


def run_checks(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)

    # Load students-only checkpoint
    ckpt = torch.load(args.students_ckpt_path, map_location=device)

    # Data
    ds = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split=args.split)
    sig_len = sample_signal_length(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Models
    students = build_students_from_ckpt(ckpt, sig_len=sig_len, device=device)

    results = {}
    batches_run = 0
    accum = {m: [] for m in students.keys()}

    with torch.no_grad():
        for batch in loader:
            if batches_run >= args.num_batches:
                break
            for name, model in students.items():
                x = batch[name].to(device)
                shared, private = extract_final_tokens(model, x)
                accum[name].append(shared.cpu())
            batches_run += 1

    # Compute metrics per modality
    for name, chunks in accum.items():
        if len(chunks) == 0:
            continue
        tokens = torch.cat(chunks, dim=0)
        metrics = compute_collapse_metrics(tokens)
        collapsed = decide_collapse(metrics)
        results[name] = {
            'collapsed': bool(collapsed),
            'metrics': metrics,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Check if KD student models have collapsed.')
    parser.add_argument('--students_ckpt_path', type=str, required=True, help='Path to students-only checkpoint (best_ckpt.pt)')
    parser.add_argument('--data_path', type=str, default='/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s')
    parser.add_argument('--fold_number', type=int, default=17)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    results = run_checks(args)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()


