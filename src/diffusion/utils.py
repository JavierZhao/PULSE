import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import logging
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# --- Configuration ---
def get_config():
    """Returns a SimpleNamespace containing the base configuration."""
    return SimpleNamespace(
        run_name="EDA_Diffusion_Training",
        epochs=100,
        batch_size=16,
        seed=42,
        device="cuda:1" if torch.cuda.is_available() else "cpu", # Default to GPU 1
        lr=1e-4,
        noise_steps=1000,
        sequence_length=3840, # 60s * 64Hz
        num_channels=1, # EDA only
        dataset_path="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s",
        output_path="/fd24T/zzhao3/EDA/results/eda_diffusion",
        fold_for_training=17,
        spec_loss_weight=0.5,
        dataset_percentage=1.0,
        resume_from=None
    )

# --- Data Loading ---
class WESADDataset(Dataset):
    """Loads the 'Y_train' EDA signals from a single preprocessed WESAD .npz fold."""
    def __init__(self, data_path, fold_number):
        super().__init__()
        file_path = os.path.join(data_path, f"fold_{fold_number}.npz")
        logging.info(f"Loading training data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with np.load(file_path) as data:
            self.signals = data['Y_train'][:, np.newaxis, :]
        logging.info(f"Loaded {self.signals.shape[0]} windows from fold {fold_number}.")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.from_numpy(self.signals[idx]).float()

# --- Plotting Functions ---
def plot_reconstructions(ground_truth: torch.Tensor, reconstructions: torch.Tensor, path: str, num_samples: int = 3):
    """Saves a plot comparing ground truth signals and their reconstructions by overlaying them."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples), sharex=True, sharey=True)
    if num_samples == 1:
        axes = [axes]
    
    gt_to_plot = ground_truth.cpu().numpy()
    recon_to_plot = reconstructions.cpu().numpy()

    ax_list = axes if isinstance(axes, list) else axes.flatten()
    for i in range(num_samples):
        ax: Axes = ax_list[i]
        ax.plot(gt_to_plot[i, 0, :], 'g', label='Ground Truth')
        ax.plot(recon_to_plot[i, 0, :], 'b', label='Reconstruction', alpha=0.7)
        ax.set_title(f"Reconstruction vs. Ground Truth (Sample {i+1})")
        ax.set_ylabel("Signal Value")
        ax.legend()
        ax.grid(True)
    
    ax_list[-1].set_xlabel("Time Step")
    fig.suptitle("One-to-One Signal Reconstruction", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(path)
    plt.close()

def plot_loss_subplots(path: str, total_train, total_val, mse_train, mse_val, spec_train, spec_val):
    """Saves a plot of loss curves for all components in separate subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    ax_list = axes.flatten()

    # Total Loss
    ax_list[0].plot(total_train, label="Training Loss")
    ax_list[0].plot(total_val, label="Validation Loss")
    ax_list[0].set_title("Total Combined Loss")
    ax_list[0].set_ylabel("Loss")
    ax_list[0].legend()
    ax_list[0].grid(True)
    
    # MSE Loss
    ax_list[1].plot(mse_train, label="Training MSE Loss")
    ax_list[1].plot(mse_val, label="Validation MSE Loss")
    ax_list[1].set_title("MSE Loss Component")
    ax_list[1].set_ylabel("Loss")
    ax_list[1].legend()
    ax_list[1].grid(True)
    
    # Spectral Loss
    ax_list[2].plot(spec_train, label="Training Spectral Loss")
    ax_list[2].plot(spec_val, label="Validation Spectral Loss")
    ax_list[2].set_title("Spectral Loss Component")
    ax_list[2].set_ylabel("Loss")
    ax_list[2].legend()
    ax_list[2].grid(True)
    
    plt.xlabel("Epoch")
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_all_loss_curves(path: str, total_train, total_val, mse_train, mse_val, spec_train, spec_val):
    """Saves a plot of all 6 loss curves on a single canvas."""
    plt.figure(figsize=(12, 8))
    
    plt.plot(total_train, label='Train Total', color='blue', linestyle='-')
    plt.plot(total_val, label='Val Total', color='blue', linestyle='--')
    
    plt.plot(mse_train, label='Train MSE', color='green', linestyle='-')
    plt.plot(mse_val, label='Val MSE', color='green', linestyle='--')
    
    plt.plot(spec_train, label='Train Spectral', color='red', linestyle='-')
    plt.plot(spec_val, label='Val Spectral', color='red', linestyle='--')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("All Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close() 