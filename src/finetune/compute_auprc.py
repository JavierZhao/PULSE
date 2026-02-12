import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score


# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.wesad_dataset import WESADDataset
from src.model.CheapSensorMAE import CheapSensorMAE
from src.finetune.finetune import StressClassifier


PREFERRED_MODALITY_ORDER: List[str] = ['ecg', 'bvp', 'acc', 'temp']


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "true", "yes", "y"}


def parse_logged_args(log_path: str) -> Dict[str, Any]:
    """Parse the initial command-line arguments logged in finetune.log.

    The logger prefixes each line with a timestamp and level; extract the
    'key: value' pairs within the marked block by reading the message part
    after " - LEVEL: ".
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    start_marker = "Command Line Arguments"
    end_marker = "------------------------------"
    in_block = False
    parsed: Dict[str, Any] = {}

    # Capture the message after " - LEVEL: " (e.g., " - INFO: key: value")
    msg_regex = re.compile(r"\s-\s[A-Z]+:\s+(?P<msg>.+)$")

    with open(log_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not in_block and start_marker in line:
                in_block = True
                continue
            if in_block and end_marker in line:
                break
            if in_block:
                m = msg_regex.search(line)
                msg = m.group('msg') if m else line
                if ':' not in msg:
                    continue
                key, value = msg.split(':', 1)
                key = key.strip()
                value = value.strip()
                low = value.lower()
                if low in {"true", "false"}:
                    parsed[key] = (low == "true")
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except ValueError:
                    pass
                parsed[key] = value
    return parsed


def extract_modalities_from_state(state_dict: Dict[str, Any]) -> List[str]:
    mods = []
    seen = set()
    for k in state_dict.keys():
        if k.startswith('models.'):
            parts = k.split('.')
            if len(parts) > 2:
                mod = parts[1]
                if mod not in seen:
                    seen.add(mod)
                    mods.append(mod)
    return [m for m in PREFERRED_MODALITY_ORDER if m in seen] or mods


def build_model_from_args(
    args_dict: Dict[str, Any],
    device: torch.device,
    only_shared_override: Optional[bool] = None,
    fuse_override: Optional[bool] = None,
    linear_override: Optional[bool] = None,
    modalities: Optional[List[str]] = None,
) -> StressClassifier:
    model_args = {
        'sig_len': int(args_dict.get('signal_length', 3840)),
        'window_len': int(args_dict.get('patch_window_len', 96)),
        'private_mask_ratio': float(args_dict.get('private_mask_ratio', 0.5)),
        'embed_dim': int(args_dict.get('embed_dim', 1024)),
        'depth': int(args_dict.get('depth', 8)),
        'num_heads': int(args_dict.get('num_heads', 4)),
        'decoder_embed_dim': int(args_dict.get('decoder_embed_dim', 512)),
        'decoder_depth': int(args_dict.get('decoder_depth', 4)),
        'decoder_num_heads': int(args_dict.get('decoder_num_heads', 16)),
        'mlp_ratio': float(args_dict.get('mlp_ratio', 4.0)),
        'decoder_mlp_ratio': float(args_dict.get('decoder_mlp_ratio', 4.0)),
    }
    selected_modalities = modalities
    if selected_modalities is None:
        mod_from_log = str(args_dict.get('modality', 'all'))
        if mod_from_log == 'all':
            selected_modalities = PREFERRED_MODALITY_ORDER.copy()
        else:
            selected_modalities = [mod_from_log]

    base_models = {
        name: CheapSensorMAE(modality_name=name, **model_args).to(device)
        for name in selected_modalities
    }
    only_shared = only_shared_override if only_shared_override is not None else _to_bool(args_dict.get('only_shared', False))
    fuse_embeddings = fuse_override if fuse_override is not None else _to_bool(args_dict.get('fuse_embeddings', False))
    linear_classifier = linear_override if linear_override is not None else _to_bool(args_dict.get('linear_classifier', False))
    model = StressClassifier(
        base_models,
        embed_dim=int(args_dict.get('embed_dim', 1024)),
        freeze_backbone=False,
        linear_classifier=linear_classifier,
        only_shared=only_shared,
        fuse_embeddings=fuse_embeddings,
        modalities=selected_modalities,
    ).to(device)
    return model


def collect_labels_and_scores(model: StressClassifier, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: List[int] = []
    scores: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            ecg = batch['ecg'].to(device)
            bvp = batch['bvp'].to(device)
            acc = batch['acc'].to(device)
            temp = batch['temp'].to(device)
            eda = batch['eda'].to(device) if 'eda' in model.modalities else None
            y = batch['label'].to(device)
            logits = model(ecg, bvp, acc, temp, eda).squeeze()
            probs = torch.sigmoid(logits)
            labels.extend(y.cpu().numpy().tolist())
            scores.extend(probs.cpu().numpy().astype(float).tolist())
    return np.array(labels), np.array(scores)


def evaluate_run_dir_auprc(
    run_dir: str,
    device: torch.device,
    batch_size: int,
    data_path: Optional[str],
    fold_number: Optional[int],
    include_eda: bool = False,
) -> Optional[Tuple[str, float]]:
    log_path = os.path.join(run_dir, 'finetune.log')
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if not (os.path.isfile(log_path) and os.path.isfile(ckpt_path)):
        return None

    logged_args = parse_logged_args(log_path)

    dp = data_path or logged_args.get('data_path')
    fn = fold_number or int(logged_args.get('fold_number', 17))
    if dp is None:
        return None

    state = torch.load(ckpt_path, map_location=device)
    state_keys = list(state.keys())
    is_mlp = any(k.startswith('classifier.3.') for k in state_keys)
    first_w = state.get('classifier.0.weight', None)
    embed_dim = int(logged_args.get('embed_dim', 1024))

    ckpt_modalities = extract_modalities_from_state(state)
    if ckpt_modalities:
        selected_modalities = ckpt_modalities
    else:
        mod_from_log = str(logged_args.get('modality', 'all'))
        selected_modalities = PREFERRED_MODALITY_ORDER.copy() if mod_from_log == 'all' else [mod_from_log]

    if include_eda and 'eda' not in selected_modalities:
        selected_modalities = selected_modalities + ['eda']

    only_shared_override = None
    fuse_override = None
    linear_override = None

    if first_w is not None and first_w.ndim == 2:
        in_features = int(first_w.shape[1])
        linear_override = not is_mlp
        m = len(selected_modalities)
        if in_features == embed_dim:
            only_shared_override = True
            fuse_override = True
        elif in_features == 2 * embed_dim:
            only_shared_override = False
            fuse_override = True
        elif in_features == m * embed_dim:
            only_shared_override = True
            fuse_override = False
        elif in_features == 2 * m * embed_dim:
            only_shared_override = False
            fuse_override = False

    model = build_model_from_args(
        logged_args,
        device,
        only_shared_override=only_shared_override,
        fuse_override=fuse_override,
        linear_override=linear_override,
        modalities=selected_modalities,
    )
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()

    test_dataset = WESADDataset(data_path=dp, fold_number=fn, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    y_true, y_score = collect_labels_and_scores(model, test_loader, device)

    try:
        auprc_val = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc_val = float('nan')

    out_path = os.path.join(run_dir, 'auprc.json')
    with open(out_path, 'w') as f:
        json.dump({'auprc': auprc_val}, f, indent=2)

    # Derive an id for fold
    base = os.path.basename(run_dir.rstrip('/'))
    m = re.search(r"f_(\d+)$", base)
    fold_id = m.group(1) if m else base

    print(f"[{base}] AUPRC={auprc_val:.6f}")
    return fold_id, auprc_val


def summarize_auprc(results: List[Tuple[str, float]], run_dir: str, output_prefix: str = 'auprc') -> None:
    if not results:
        return
    # Per-fold CSV
    rows = [{'fold': fid, 'auprc': float(val)} for fid, val in results]
    df = pd.DataFrame(rows).sort_values(by='fold')
    per_fold_csv = os.path.join(run_dir, f"{output_prefix}_per_fold.csv")
    df.to_csv(per_fold_csv, index=False)

    # Summary stats
    vals = np.array([float(v) for _, v in results], dtype=float)
    summary = {
        'auprc': {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals, ddof=0)),
            'min': float(np.nanmin(vals)),
            'max': float(np.nanmax(vals)),
            'count': int(np.sum(~np.isnan(vals))),
        }
    }
    summary_json = os.path.join(run_dir, f"{output_prefix}_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # Compact CSV
    summary_df = pd.DataFrame([
        {'metric': 'auprc', **summary['auprc']}
    ], columns=['metric', 'mean', 'std', 'min', 'max', 'count'])
    summary_csv = os.path.join(run_dir, f"{output_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote per-fold AUPRC to: {per_fold_csv}")
    print(f"Wrote AUPRC summary (JSON) to: {summary_json}")
    print(f"Wrote AUPRC summary (CSV) to: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description='Compute AUPRC for each fold under a run directory and summarize results')
    parser.add_argument('--run_dir', type=str, required=True, help='Run dir with checkpoint/log, or a parent dir containing multiple run subdirs')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid")
    parser.add_argument('--fold_number', type=int, default=None)
    parser.add_argument('--include_eda', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)

    results: List[Tuple[str, float]] = []

    # If the provided run_dir itself looks like a run, evaluate it
    has_top = os.path.isfile(os.path.join(args.run_dir, 'finetune.log')) and os.path.isfile(os.path.join(args.run_dir, 'best_model.pt'))
    if has_top:
        try:
            r = evaluate_run_dir_auprc(args.run_dir, device, args.batch_size, args.data_path, args.fold_number, include_eda=args.include_eda)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"Error evaluating {args.run_dir}: {e}")

    # Evaluate all immediate subdirectories
    try:
        subdirs = [os.path.join(args.run_dir, d) for d in os.listdir(args.run_dir) if os.path.isdir(os.path.join(args.run_dir, d))]
    except FileNotFoundError:
        print(f"run_dir not found: {args.run_dir}")
        return

    for sub in sorted(subdirs):
        try:
            r = evaluate_run_dir_auprc(sub, device, args.batch_size, args.data_path, args.fold_number, include_eda=args.include_eda)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"Error evaluating {sub}: {e}")

    summarize_auprc(results, args.run_dir, output_prefix='fold_test_auprc')


if __name__ == '__main__':
    main()


