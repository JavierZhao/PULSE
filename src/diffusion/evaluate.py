import os
import torch
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

# Add src to path to allow project-level imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.modules import UNet_1D as UNet
from model.eda_diffusion import EDADiffusion
from utils import get_config, WESADDataset, plot_reconstructions
from torch.utils.data import DataLoader

def calculate_metrics(ground_truth_batch, reconstructions_batch):
    """Calculates a set of quality metrics for a batch of signals."""
    metrics = {'mae': [], 'pearson_r': [], 'max_xcorr': [], 'max_xcorr_lag': []}
    
    # Ensure batches are on CPU and numpy format for calculations
    gt_np = ground_truth_batch.cpu().numpy().squeeze(axis=1)
    recon_np = reconstructions_batch.cpu().numpy().squeeze(axis=1)

    for i in range(gt_np.shape[0]):
        gt_signal = gt_np[i]
        recon_signal = recon_np[i]
        
        # MAE
        metrics['mae'].append(np.mean(np.abs(gt_signal - recon_signal)))
        
        # Pearson Correlation
        r, _ = pearsonr(gt_signal, recon_signal)
        metrics['pearson_r'].append(r)
        
        # Cross-Correlation
        xcorr = np.correlate(gt_signal - np.mean(gt_signal), recon_signal - np.mean(recon_signal), mode='full')
        max_corr_idx = np.argmax(xcorr)
        max_corr_lag = max_corr_idx - (len(gt_signal) - 1)
        
        # Normalize the max cross-correlation to be between -1 and 1
        norm_factor = np.sqrt(np.sum(gt_signal**2) * np.sum(recon_signal**2))
        max_xcorr_val = xcorr[max_corr_idx] / norm_factor if norm_factor > 0 else 0
        
        metrics['max_xcorr'].append(max_xcorr_val)
        metrics['max_xcorr_lag'].append(max_corr_lag)
        
    return metrics

def evaluate(args):
    """Loads a trained model and evaluates it quantitatively and qualitatively."""
    config = get_config()
    device = args.device
    
    # --- Setup ---
    run_dir = os.path.dirname(os.path.dirname(args.model_path))
    output_dir = os.path.join(run_dir, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file and console
    log_path = os.path.join(output_dir, "evaluation.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s: %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info(f"--- Starting Evaluation for {args.model_path} ---")

    # --- Data Loading ---
    dataset = WESADDataset(config.dataset_path, config.fold_for_training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Model Loading ---
    model = UNet(in_channels=config.num_channels, out_channels=config.num_channels).to(device)
    logging.info(f"Loading model checkpoint from {args.model_path}")
    # When resuming, the checkpoint is a dictionary
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # For older, model-only checkpoints
    model.eval()

    diffusion = EDADiffusion(
        noise_steps=config.noise_steps,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        device=device
    )

    # --- Evaluation Loop (Single-Step Denoising) ---
    all_metrics = {'mae': [], 'pearson_r': [], 'max_xcorr': [], 'max_xcorr_lag': []}
    first_batch_for_plotting = None

    logging.info(f"Evaluating on {len(val_dataset)} validation samples using efficient single-step denoising...")
    pbar = tqdm(val_dataloader, desc="Evaluating Batches")
    with torch.no_grad():
        for i, ground_truth_batch in enumerate(pbar):
            ground_truth_batch = ground_truth_batch.to(device)
            
            # Pick a random timestep t
            t = diffusion.sample_timesteps(ground_truth_batch.shape[0]).to(device)
            
            # Add noise to get x_t, and get the original noise
            x_t, noise = diffusion.q_sample(ground_truth_batch, t)
            
            # Predict the noise with the model
            predicted_noise = model(x_t, t)
            
            # Denoise x_t to get the reconstructed signal x0_hat
            alpha_bar = diffusion.alpha_hat[t].view(-1, 1, 1)
            reconstructions = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

            # Calculate and store metrics between ground truth and one-step reconstruction
            batch_metrics = calculate_metrics(ground_truth_batch, reconstructions)
            for key in all_metrics:
                all_metrics[key].extend(batch_metrics[key])

            # Save the first batch for generating a qualitative plot later
            if i == 0:
                first_batch_for_plotting = (ground_truth_batch.cpu(), reconstructions.cpu())
    
    # --- Reporting Metrics ---
    logging.info("--- Quantitative Evaluation Results ---")
    results_summary = ""
    for key, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        log_line = f"- {key.replace('_', ' ').title():<18}: Mean={mean_val:.4f}, Std={std_val:.4f}"
        logging.info(log_line)
        results_summary += log_line + "\n"

    # Save summary to a text file
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("--- Quantitative Evaluation Results ---\n")
        f.write(f"Evaluated Model: {args.model_path}\n\n")
        f.write(results_summary)
    logging.info(f"Saved metrics summary to {summary_path}")
        
    # --- Qualitative Plotting ---
    if first_batch_for_plotting:
        logging.info("Generating reconstruction plot for a sample batch...")
        num_samples_to_plot = min(8, first_batch_for_plotting[0].shape[0])
        plot_path = os.path.join(output_dir, "reconstruction_comparison_plot.png")
        plot_reconstructions(
            ground_truth=first_batch_for_plotting[0][:num_samples_to_plot], 
            reconstructions=first_batch_for_plotting[1][:num_samples_to_plot], 
            path=plot_path, 
            num_samples=num_samples_to_plot
        )
        logging.info(f"Saved sample reconstruction plot to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained diffusion model by reconstructing signals.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file). e.g., results/eda_diffusion/RUN_NAME/models/best_ckpt.pt')
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to use ('cuda:0', 'cpu', etc.)")
    
    config = get_config()
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help="Batch size for evaluation.")

    args = parser.parse_args()
    config.batch_size = args.batch_size # Update config with eval batch size
    evaluate(args) 