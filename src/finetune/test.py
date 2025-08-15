import argparse
import os
import sys
import json
import logging
import re
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)
import pandas as pd

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.wesad_dataset import WESADDataset
from src.model.CheapSensorMAE import CheapSensorMAE
from src.finetune.finetune import StressClassifier
from src.finetune.summarize_folds import find_fold_dirs, load_metrics, parse_fold_id, aggregate_metrics

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
                # Convert value to appropriate type when possible
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
    # Keep a stable, preferred order
    return [m for m in PREFERRED_MODALITY_ORDER if m in seen] or mods


def build_model_from_args(
    args_dict: Dict[str, Any],
    device: torch.device,
    only_shared_override: Optional[bool] = None,
    fuse_override: Optional[bool] = None,
    linear_override: Optional[bool] = None,
    modalities: Optional[List[str]] = None,
) -> StressClassifier:
    """Construct StressClassifier with base CheapSensorMAE models using args from log and selected modalities."""
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


def evaluate(model: StressClassifier, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            ecg, bvp, acc, temp = [batch[k].to(device) for k in ('ecg', 'bvp', 'acc', 'temp')]
            labels = batch['label'].to(device)
            logits = model(ecg, bvp, acc, temp).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_scores.extend(probs.cpu().numpy().astype(float).tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_scores)

    metrics: Dict[str, Any] = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='binary'))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))

    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics['roc_auc'] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    metrics['classification_report'] = classification_report(y_true, y_pred, digits=4)

    return metrics, y_true, y_score, y_pred, cm


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Non-stress', 'Stress']
    tick_marks = np.arange(len(classes))
    ax.set(xticks=tick_marks, yticks=tick_marks, xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray, save_dir: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_roc_curve.png'))
        plt.close()
    except Exception:
        pass

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'test_pr_curve.png'))
        plt.close()
    except Exception:
        pass


def write_summary_files(run_dir: str, output_prefix: str = 'fold_test'):
    # Build the same summaries as summarize_folds
    fold_dirs = find_fold_dirs(run_dir)
    if not fold_dirs:
        return
    rows: List[Dict[str, Any]] = []
    for fd in fold_dirs:
        metrics_path = os.path.join(fd, 'test_metrics.json')
        try:
            metrics = load_metrics(metrics_path)
        except Exception:
            continue
        fold_id = parse_fold_id(fd)
        row = {'fold': fold_id}
        for key in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            row[key] = metrics.get(key, None)
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(by='fold')
    per_fold_csv = os.path.join(run_dir, f"{output_prefix}_per_fold.csv")
    df.to_csv(per_fold_csv, index=False)
    summary = aggregate_metrics(rows)
    summary_json_path = os.path.join(run_dir, f"{output_prefix}_summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    summary_rows = []
    for metric, stats in summary.items():
        sr = {'metric': metric}
        sr.update(stats)
        summary_rows.append(sr)
    summary_df = pd.DataFrame(summary_rows, columns=['metric', 'mean', 'std', 'min', 'max', 'count'])
    summary_csv = os.path.join(run_dir, f"{output_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)


def evaluate_run_dir(run_dir: str, device: torch.device, batch_size: int, data_path: Optional[str], fold_number: Optional[int]):
    log_path = os.path.join(run_dir, 'finetune.log')
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if not (os.path.isfile(log_path) and os.path.isfile(ckpt_path)):
        print(f"Skip: missing finetune.log or best_model.pt in {run_dir}")
        return

    # Parse training args from log to reconstruct model and dataset
    logged_args = parse_logged_args(log_path)

    dp = data_path or logged_args.get('data_path')
    fn = fold_number or int(logged_args.get('fold_number', 17))
    print(f"using fold {fn}")
    if dp is None:
        print(f"Skip: data_path not found in log and not provided for {run_dir}")
        return

    # Inspect checkpoint to infer classifier architecture and modalities
    state = torch.load(ckpt_path, map_location=device)
    state_keys = list(state.keys())
    is_mlp = any(k.startswith('classifier.3.') for k in state_keys)
    first_w = state.get('classifier.0.weight', None)
    embed_dim = int(logged_args.get('embed_dim', 1024))

    # Infer modalities from checkpoint, fallback to log
    ckpt_modalities = extract_modalities_from_state(state)
    if ckpt_modalities:
        selected_modalities = ckpt_modalities
    else:
        mod_from_log = str(logged_args.get('modality', 'all'))
        selected_modalities = PREFERRED_MODALITY_ORDER.copy() if mod_from_log == 'all' else [mod_from_log]

    only_shared_override = None
    fuse_override = None
    linear_override = None

    if first_w is not None and first_w.ndim == 2:
        in_features = int(first_w.shape[1])
        linear_override = not is_mlp
        # Infer only_shared and fuse from in_features vs embed_dim and num modalities
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

    device_t = device
    model = build_model_from_args(
        logged_args,
        device_t,
        only_shared_override=only_shared_override,
        fuse_override=fuse_override,
        linear_override=linear_override,
        modalities=selected_modalities,
    )

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[{os.path.basename(run_dir)}] strict load failed: {e}. Retrying with strict=False...")
        model.load_state_dict(state, strict=False)
    model.eval()

    # DataLoader for test set
    test_dataset = WESADDataset(data_path=dp, fold_number=fn, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Evaluate
    metrics, y_true, y_score, y_pred, cm = evaluate(model, test_loader, device_t)

    # Save outputs
    metrics_path = os.path.join(run_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, 'test_classification_report.txt'), 'w') as f:
        f.write(metrics['classification_report'])

    cm_path = os.path.join(run_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)

    plot_roc_pr_curves(y_true, y_score, run_dir)

    print(f"[{os.path.basename(run_dir)}] mods={selected_modalities} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a finetuned model on the WESAD test set')
    parser.add_argument('--run_dir', type=str, required=True, help='Run dir with checkpoint/log, or a parent dir containing multiple run subdirs')
    parser.add_argument('--ckpt_name', type=str, default='best_model.pt', help='(unused now) expected checkpoint filename inside run_dir')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_path', type=str, default="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid", help='Optional override of data_path from log')
    parser.add_argument('--fold_number', type=int, default=None, help='Optional override of fold_number from log')
    args = parser.parse_args()

    device = torch.device(args.device)

    # If the provided run_dir itself looks like a run (has finetune.log and best_model.pt), evaluate it.
    has_top = os.path.isfile(os.path.join(args.run_dir, 'finetune.log')) and os.path.isfile(os.path.join(args.run_dir, 'best_model.pt'))

    if has_top:
        try:
            evaluate_run_dir(args.run_dir, device, args.batch_size, args.data_path, args.fold_number)
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
            evaluate_run_dir(sub, device, args.batch_size, args.data_path, args.fold_number)
        except Exception as e:
            print(f"Error evaluating {sub}: {e}")

    # After evaluations, write aggregate summary across fold subdirectories
    write_summary_files(args.run_dir, output_prefix='fold_test')


if __name__ == '__main__':
    main()
