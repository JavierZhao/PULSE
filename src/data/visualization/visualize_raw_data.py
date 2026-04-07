# Copyright (c) 2026 PULSE contributors
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.secure_io import load_wesad_pickle

# --- Configuration ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
WESAD_PKL_DIR = os.path.join(DATA_DIR, 'WESAD')
RAW_CSV_DIR = DATA_DIR 
OUTPUT_DIR = 'figures/raw_plots'

# Label mapping for the plot
LABEL_DICT = {
    0: ('Transient', 'gray'),
    1: ('Baseline', 'green'),
    2: ('Stress', 'red'),
    3: ('Amusement', 'blue'),
    4: ('Meditation', 'purple'),
    5: ('', 'white'), # Ignore
    6: ('', 'white'), # Ignore
    7: ('', 'white')  # Ignore
}

# Original sampling rates for signals we will plot
NATIVE_FS = {
    'bvp': 64, 
    'eda': 4, 
    'temp': 4,
    'label': 700,
    'ecg': 700,
    'net_acc_wrist': 32,
    'net_acc_chest': 700
}

def plot_raw_subject_data(subject_id):
    """
    Loads raw BVP, EDA, and TEMP signals along with labels for a subject
    and plots them over the entire experiment duration.
    """
    subject_name = f'S{subject_id}'
    print(f"--- Plotting raw data for {subject_name} ---")

    # --- 1. Load Data ---
    signals = {}
    
    # Load signals from their respective CSV files
    csv_base_path = os.path.join(RAW_CSV_DIR, f'{subject_name}_raw_data')
    if not os.path.isdir(csv_base_path):
        print(f"Error: Raw data directory not found at {csv_base_path}")
        return

    # Load single-column signals
    for signal_name in ['bvp', 'eda', 'temp', 'ecg']:
        file_path = os.path.join(csv_base_path, f'{signal_name}.csv')
        if os.path.exists(file_path):
            print(f"  - Loading {signal_name} from {file_path}")
            signals[signal_name] = pd.read_csv(file_path, index_col=0).values.ravel()
        else:
            print(f"  - Warning: {signal_name}.csv not found. Skipping.")

    # Load multi-column accelerometer signals and compute net acceleration
    for acc_type in ['acc_wrist', 'acc_chest']:
        file_path = os.path.join(csv_base_path, f'{acc_type}.csv')
        if os.path.exists(file_path):
            print(f"  - Loading and processing {acc_type} from {file_path}")
            acc_data = pd.read_csv(file_path, index_col=0).values
            net_acc = np.linalg.norm(acc_data, axis=1)
            signals[f'net_{acc_type}'] = net_acc
        else:
            print(f"  - Warning: {acc_type}.csv not found. Skipping.")

    # Load labels from the pkl file
    pkl_path = os.path.join(WESAD_PKL_DIR, subject_name, f'{subject_name}.pkl')
    if not os.path.exists(pkl_path):
        print(f"Error: Label .pkl file not found at {pkl_path}")
        return
        
    print(f"  - Loading labels from {pkl_path}")
    pkl_data = load_wesad_pickle(pkl_path)
    labels = pkl_data['label'].ravel()

    # print the label distribution
    from collections import Counter
    print(f"Label distribution: {Counter(labels)}")
    
    if not signals:
        print("Error: No signals were loaded. Aborting plot.")
        return

    # --- 2. Create Plots ---
    num_signals = len(signals)
    fig, axes = plt.subplots(
        num_signals + 1, 1, 
        figsize=(18, 4 * (num_signals + 1)), 
        sharex=True
    )
    
    fig.suptitle(f'Raw Data for Subject {subject_id}', fontsize=16)

    # Plot each signal
    for i, (name, data) in enumerate(signals.items()):
        ax = axes[i]
        fs = NATIVE_FS.get(name, 1)
        time_axis = np.arange(len(data)) / fs
        ax.plot(time_axis, data, label=name.upper())
        ax.set_ylabel(f'{name.upper()} Signal')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Plot the labels on the last subplot
    ax_labels = axes[-1]
    label_fs = NATIVE_FS['label']
    label_time_axis = np.arange(len(labels)) / label_fs

    # Create a color-mapped visualization for labels
    for label_int, (label_name, color) in LABEL_DICT.items():
        if label_int > 4: continue # Don't plot "ignore" labels
        # Find where the label occurs
        label_indices = np.where(labels == label_int)[0]
        # Create a boolean array to plot segments
        segments = np.zeros_like(labels, dtype=bool)
        segments[label_indices] = True
        ax_labels.fill_between(label_time_axis, 0, 1, where=segments, color=color, alpha=0.5, step='post')

    # Create a custom legend for the labels
    patches = [mpatches.Patch(color=color, label=name) for name, color in LABEL_DICT.values() if name]
    ax_labels.legend(handles=patches, loc='upper right', title='Labels')
    ax_labels.set_yticks([])
    ax_labels.set_ylabel('Condition')
    ax_labels.set_xlabel('Time (seconds)')
    ax_labels.grid(False)

    # --- 3. Save Figure ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'S{subject_id}_raw_plot.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


def main():
    """Main function to parse arguments and trigger plotting."""
    parser = argparse.ArgumentParser(
        description="Visualize raw BVP, EDA, TEMP, ECG, and Net Acceleration signals and labels for a given WESAD subject."
    )
    parser.add_argument(
        '--subject_id', type=int, required=True,
        help="The subject ID to load and visualize (e.g., 2)."
    )
    args = parser.parse_args()

    plot_raw_subject_data(args.subject_id)


if __name__ == '__main__':
    main() 
