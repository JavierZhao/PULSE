import argparse
import os
import numpy as np
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_header(header_path):
    if os.path.isfile(header_path):
        try:
            with open(header_path, 'r') as f:
                header_line = f.read().strip()
            # Expect comma-separated keys
            header = [h.strip() for h in header_line.split(',') if len(h.strip()) > 0]
            # Basic sanity check
            if len(header) >= 5:
                return header
        except Exception:
            pass
    # Fallback to known ordering used in finetune.py
    return [
        'epoch',
        'train_loss', 'train_acc', 'train_f1',
        'val_loss', 'val_acc', 'val_f1', 'val_auroc', 'val_auprc',
        'val_f1_tuned', 'val_tuned_thr', 'val_precision_tuned', 'val_recall_tuned',
    ]


def plot_curves(npy_path: str, output_path: str | None = None):
    npy_path = os.path.abspath(npy_path)
    npy_dir = os.path.dirname(npy_path)
    header_path = os.path.join(npy_dir, 'metrics_header.txt')
    header = load_header(header_path)
    header_to_idx = {name: idx for idx, name in enumerate(header)}

    data = np.load(npy_path)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Unexpected metrics array shape: {data.shape}. Expected (N, >=5)")

    # X-axis as epoch if present, else 1..N
    if 'epoch' in header_to_idx:
        x = data[:, header_to_idx['epoch']]
    else:
        x = np.arange(1, data.shape[0] + 1)

    def get_col(name: str, default=None):
        if name in header_to_idx:
            return data[:, header_to_idx[name]]
        return default

    train_loss = get_col('train_loss')
    val_loss = get_col('val_loss')
    train_acc = get_col('train_acc')
    val_acc = get_col('val_acc')
    val_auprc = get_col('val_auprc')

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

    # Losses
    ax = axes[0]
    if train_loss is not None:
        ax.plot(x, train_loss, label='Train Loss')
    if val_loss is not None:
        ax.plot(x, val_loss, label='Val Loss')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[1]
    if train_acc is not None:
        ax.plot(x, train_acc, label='Train Acc')
    if val_acc is not None:
        ax.plot(x, val_acc, label='Val Acc')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # AUPRC (validation only)
    ax = axes[2]
    if val_auprc is not None:
        ax.plot(x, val_auprc, label='Val AUPRC')
    ax.set_title('AUPRC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUPRC')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save
    if output_path is None:
        output_path = os.path.join(npy_dir, 'training_curves.png')
    fig.suptitle(os.path.basename(os.path.dirname(npy_dir)) + ' / ' + os.path.basename(npy_dir))
    fig.savefig(output_path, dpi=150)
    print(f"Saved plots to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training/validation curves from training_metrics.npy')
    parser.add_argument('--npy', type=str, required=True, help='Path to training_metrics.npy')
    parser.add_argument('--out', type=str, default=None, help='Optional path to save the output image')
    args = parser.parse_args()
    plot_curves(args.npy, args.out)


if __name__ == '__main__':
    main()


