import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(log_path):
    """Parses the finetune.log file to extract metrics."""
    epoch_pattern = re.compile(
        r"Epoch (\d+) \| "
        r"Train Loss: ([\d.]+) \| "
        r"Train Acc: ([\d.]+) \| "
        r"Train F1: ([\d.]+) \| "
        r"Val Loss: ([\d.]+) \| "
        r"Val F1: ([\d.]+) \| "
        r"Val Acc: ([\d.]+)"
    )
    
    metrics = []
    with open(log_path, 'r') as f:
        for line in f:
            match = epoch_pattern.search(line)
            if match:
                metrics.append({
                    'epoch': int(match.group(1)),
                    'train_loss': float(match.group(2)),
                    'train_acc': float(match.group(3)),
                    'train_f1': float(match.group(4)),
                    'val_loss': float(match.group(5)),
                    'val_f1': float(match.group(6)),
                    'val_acc': float(match.group(7)),
                })
    
    if not metrics:
        raise ValueError("No metrics found in log file. Check the log format.")
        
    return pd.DataFrame(metrics)

def plot_metrics(df, output_dir, title):
    """Plots and saves the training and validation metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)

    # Plot Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    axes[0].plot(df['epoch'], df['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs. Epoch')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    axes[1].plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs. Epoch')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)

    # Plot F1 Score
    axes[2].plot(df['epoch'], df['train_f1'], label='Train F1 Score')
    axes[2].plot(df['epoch'], df['val_f1'], label='Validation F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score vs. Epoch')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = os.path.join(output_dir, 'finetune_metrics.png')
    plt.savefig(save_path)
    print(f"Saved plots to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from finetune.log file.')
    parser.add_argument('input_dir', type=str, help='Directory containing the finetune.log file.')
    parser.add_argument('--tag', type=str, default='Finetuning Metrics', help='Title for the plot.')
    args = parser.parse_args()

    log_file = os.path.join(args.input_dir, 'finetune.log')

    if not os.path.isfile(log_file):
        print(f"Error: finetune.log not found in {args.input_dir}")
        return

    try:
        metrics_df = parse_log_file(log_file)
        plot_metrics(metrics_df, args.input_dir, args.tag)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

