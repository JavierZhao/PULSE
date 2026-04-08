# Copyright (c) 2026 Futurewei Technologies, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.secure_io import load_npz_archive

# --- Configuration ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PREPROCESSED_DIR = os.path.join(REPO_ROOT, 'preprocessed_data')
NUM_SAMPLES_TO_PLOT = 4
FS = 64 # The target sample rate from the preprocessing script
OUTPUT_DIR = 'figures/temp_only_plots'

# --- Label Mapping ---
LABEL_DICT = {
    1: 'Baseline',
    2: 'Stress',
    3: 'Amusement'
}

def plot_temp_for_fold(fold_number):
    """
    Loads a preprocessed data fold and plots only the temperature signal
    for a few random samples from the test set.
    """
    # 1. Construct the file path and load the data
    file_path = os.path.join(PREPROCESSED_DIR, f'fold_{fold_number}.npz')
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    print(f"Loading data from {file_path}...")
    try:
        with load_npz_archive(file_path) as data:
            X_test = data['X_test']
            L_test = data['L_test']
            feature_names = np.asarray(data['feature_names']).astype(str, copy=False)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Find the index for the 'temp' signal
    try:
        temp_idx = np.where(feature_names == 'temp')[0][0]
        print(f"Found 'temp' signal at feature index {temp_idx}.")
    except IndexError:
        print("Error: 'temp' signal not found in the feature names for this fold.")
        return

    if X_test.size == 0:
        print("No test data available in this fold to plot.")
        return

    # Create base output directory for the fold
    output_dir_fold = os.path.join(OUTPUT_DIR, f'fold_{fold_number}')
    os.makedirs(output_dir_fold, exist_ok=True)

    time_axis = np.linspace(0, X_test.shape[1] / FS, num=X_test.shape[1])

    # 3. Iterate through each label to create a plot
    for label_int, label_name in LABEL_DICT.items():
        indices_for_label = np.where(L_test == label_int)[0]

        if len(indices_for_label) == 0:
            print(f"\nInfo: No test samples for label '{label_name}' ({label_int}). Skipping plot.")
            continue

        num_to_select = min(NUM_SAMPLES_TO_PLOT, len(indices_for_label))
        plot_indices = np.random.choice(
            indices_for_label, size=num_to_select, replace=False
        )

        print(f"\nPlotting {len(plot_indices)} temperature samples for label '{label_name}'...")

        # 4. Create the plot for the current label
        fig, axes = plt.subplots(
            len(plot_indices), 1,
            figsize=(12, 3 * len(plot_indices)),
            sharex=True, sharey=True
        )
        if len(plot_indices) == 1:
            axes = [axes]

        fig.suptitle(f'Temperature Signals for Label: {label_name}', fontsize=16)

        for i, sample_idx in enumerate(plot_indices):
            ax = axes[i]
            temp_signal = X_test[sample_idx, :, temp_idx]
            print(temp_signal.shape)
            print(temp_signal)
            
            ax.plot(time_axis, temp_signal, 'r-', label='Temperature')
            ax.set_ylabel('Temp (Z-scored)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'Test Sample #{sample_idx}')
            ax.legend()

        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        save_path = os.path.join(output_dir_fold, f'{label_name}_temp_samples.png')
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close(fig)


def main():
    """Main function to parse arguments and trigger plotting."""
    parser = argparse.ArgumentParser(
        description="Visualize ONLY the preprocessed temperature signal from a specific WESAD fold."
    )
    parser.add_argument(
        'fold_number', type=int,
        help="The fold number to load and visualize (e.g., 2 for 'fold_2.npz')."
    )
    args = parser.parse_args()

    plot_temp_for_fold(args.fold_number)


if __name__ == '__main__':
    main() 
