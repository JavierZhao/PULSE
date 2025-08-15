import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


def find_fold_dirs(run_dir: str) -> List[str]:
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    subdirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    # Prefer directories that look like folds (e.g., f_1, f_17), but include all with test_metrics.json
    fold_like = []
    for sd in subdirs:
        if os.path.isfile(os.path.join(sd, 'test_metrics.json')):
            fold_like.append(sd)
    return sorted(fold_like)


def parse_fold_id(dirname: str) -> str:
    base = os.path.basename(dirname.rstrip('/'))
    m = re.search(r"f_(\d+)$", base)
    if m:
        return m.group(1)
    # Fallback: return the directory base name
    return base


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    with open(metrics_path, 'r') as f:
        return json.load(f)


def aggregate_metrics(per_fold: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    # Collect numeric metrics; ignore non-numeric or None
    keys = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        values = [float(m[k]) for m in per_fold if (k in m and m[k] is not None)]
        if len(values) == 0:
            summary[k] = {
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'q1': float('nan'),
                'median': float('nan'),
                'q3': float('nan'),
                'max': float('nan'),
                'iqr': float('nan'),
                'count': 0,
            }
        else:
            arr = np.array(values, dtype=float)
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            summary[k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=0)),
                'min': float(np.min(arr)),
                'q1': q1,
                'median': float(np.median(arr)),
                'q3': q3,
                'max': float(np.max(arr)),
                'iqr': float(q3 - q1),
                'count': int(arr.size),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize test metrics across fold subdirectories of a run directory')
    parser.add_argument('--run_dir', type=str, required=True, help='Parent run directory containing fold subdirectories (each with test_metrics.json)')
    parser.add_argument('--output_prefix', type=str, default='fold_test', help='Prefix for output files written to run_dir')
    args = parser.parse_args()

    run_dir = args.run_dir
    fold_dirs = find_fold_dirs(run_dir)
    if not fold_dirs:
        print(f"No fold subdirectories with test_metrics.json found under: {run_dir}")
        return

    rows: List[Dict[str, Any]] = []
    for fd in fold_dirs:
        metrics_path = os.path.join(fd, 'test_metrics.json')
        try:
            metrics = load_metrics(metrics_path)
        except Exception as e:
            print(f"Skipping {fd}: failed to load metrics: {e}")
            continue
        fold_id = parse_fold_id(fd)
        row = {'fold': fold_id}
        # Pull selected metrics if available
        for key in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            row[key] = metrics.get(key, None)
        rows.append(row)

    if not rows:
        print("No valid test metrics found to summarize.")
        return

    df = pd.DataFrame(rows).sort_values(by='fold')
    per_fold_csv = os.path.join(run_dir, f"{args.output_prefix}_per_fold.csv")
    df.to_csv(per_fold_csv, index=False)

    summary = aggregate_metrics(rows)
    summary_json_path = os.path.join(run_dir, f"{args.output_prefix}_summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Also write a compact CSV for summary
    summary_rows = []
    for metric, stats in summary.items():
        sr = {'metric': metric}
        sr.update(stats)
        summary_rows.append(sr)
    summary_df = pd.DataFrame(summary_rows, columns=['metric', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'iqr', 'count'])
    summary_csv = os.path.join(run_dir, f"{args.output_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote per-fold metrics to: {per_fold_csv}")
    print(f"Wrote summary (JSON) to: {summary_json_path}")
    print(f"Wrote summary (CSV) to: {summary_csv}")


if __name__ == '__main__':
    main()
