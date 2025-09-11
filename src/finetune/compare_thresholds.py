import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _find_fold_dirs(parent: str) -> List[str]:
    if not os.path.isdir(parent):
        return []
    subdirs = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    # Keep those with test_metrics.json
    return sorted([d for d in subdirs if os.path.isfile(os.path.join(d, 'test_metrics.json'))])


def _parse_fold_id(dirname: str) -> str:
    base = os.path.basename(dirname.rstrip('/'))
    # Accept names like f_1, f_17; otherwise return base
    import re
    m = re.search(r"f_(\d+)$", base)
    return m.group(1) if m else base


def _extract_thr_metrics(metrics: Dict[str, Any], which: str) -> Dict[str, Any]:
    # which in {"0.5", "best"}
    if which == "0.5":
        src = metrics.get('metrics_at_0_5', {})
    else:
        src = metrics.get('metrics_at_best_threshold', {})
    cm = src.get('confusion_matrix')
    tn = fp = fn = tp = None
    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
        tn, fp = cm[0]
        fn, tp = cm[1]
    return {
        'threshold': src.get('threshold'),
        'accuracy': src.get('accuracy'),
        'f1': src.get('f1'),
        'precision': src.get('precision'),
        'recall': src.get('recall'),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def build_comparison_rows(run_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fd in _find_fold_dirs(run_dir):
        tm_path = os.path.join(fd, 'test_metrics.json')
        tm = _read_json(tm_path)
        if not tm:
            continue
        fold = _parse_fold_id(fd)
        base = {
            'fold': fold,
            'auroc': tm.get('roc_auc'),
            'auprc': tm.get('auprc'),
        }
        m05 = _extract_thr_metrics(tm, '0.5')
        mbest = _extract_thr_metrics(tm, 'best') if 'metrics_at_best_threshold' in tm else None

        row = dict(base)
        # thr=0.5 columns
        row.update({
            'thr_0_5': m05.get('threshold'),
            'acc_0_5': m05.get('accuracy'),
            'f1_0_5': m05.get('f1'),
            'precision_0_5': m05.get('precision'),
            'recall_0_5': m05.get('recall'),
            'tn_0_5': m05.get('tn'),
            'fp_0_5': m05.get('fp'),
            'fn_0_5': m05.get('fn'),
            'tp_0_5': m05.get('tp'),
        })
        # thr=best columns (may be None)
        row.update({
            'thr_best': None,
            'acc_best': None,
            'f1_best': None,
            'precision_best': None,
            'recall_best': None,
            'tn_best': None,
            'fp_best': None,
            'fn_best': None,
            'tp_best': None,
        })
        if mbest is not None:
            row.update({
                'thr_best': mbest.get('threshold'),
                'acc_best': mbest.get('accuracy'),
                'f1_best': mbest.get('f1'),
                'precision_best': mbest.get('precision'),
                'recall_best': mbest.get('recall'),
                'tn_best': mbest.get('tn'),
                'fp_best': mbest.get('fp'),
                'fn_best': mbest.get('fn'),
                'tp_best': mbest.get('tp'),
            })

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Compare threshold-dependent test metrics at thr=0.5 vs best threshold across folds')
    parser.add_argument('--run_dir', type=str, required=True, help='Parent run directory containing fold subdirectories')
    parser.add_argument('--output_name', type=str, default='fold_threshold_comparison.csv', help='Output CSV filename written under run_dir')
    args = parser.parse_args()

    rows = build_comparison_rows(args.run_dir)
    if not rows:
        print(f"No rows to write. Ensure test_metrics.json exists under subdirectories of: {args.run_dir}")
        return

    df = pd.DataFrame(rows)
    # Order columns for readability
    preferred_cols = [
        'fold', 'auroc', 'auprc',
        'thr_0_5', 'acc_0_5', 'f1_0_5', 'precision_0_5', 'recall_0_5', 'tn_0_5', 'fp_0_5', 'fn_0_5', 'tp_0_5',
        'thr_best', 'acc_best', 'f1_best', 'precision_best', 'recall_best', 'tn_best', 'fp_best', 'fn_best', 'tp_best',
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols].sort_values(by='fold')

    out_path = os.path.join(args.run_dir, args.output_name)
    df.to_csv(out_path, index=False)
    print(f"Wrote threshold comparison CSV to: {out_path}")


if __name__ == '__main__':
    main()


