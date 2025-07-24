import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch import optim
import numpy as np
import argparse
import logging
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Assuming a 1D U-Net is available in modules.modules
from modules.modules import UNet_1D as UNet
from model.eda_diffusion import EDADiffusion
from losses.spectral import spectral_l1
from utils import get_config, WESADDataset, plot_reconstructions, plot_loss_subplots, plot_all_loss_curves

# --- Setup Logging ---
def setup_logging(run_name, output_path):
    """Configures logging to file and console."""
    log_dir = os.path.join(output_path, run_name)
    log_path = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler to save logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # Console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

# --- Training and Sampling ---
def train(config):
    device = config.device
    
    # Data
    dataset = WESADDataset(config.dataset_path, config.fold_for_training)

    # Reduce dataset size based on percentage argument
    if config.dataset_percentage < 1.0:
        original_size = len(dataset)
        subset_size = int(original_size * config.dataset_percentage)
        generator = torch.Generator().manual_seed(config.seed)
        dataset, _ = torch.utils.data.random_split(dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {config.dataset_percentage*100:.0f}% of the dataset: {len(dataset)} out of {original_size} windows.")

    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Model and Diffusion
    model = UNet(in_channels=config.num_channels, out_channels=config.num_channels).to(device)
    diffusion = EDADiffusion(
        noise_steps=config.noise_steps,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        device=device
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_mse_losses, val_mse_losses = [], []
    train_spec_losses, val_spec_losses = [], []
    
    # --- Resume from Checkpoint ---
    if config.resume_from and os.path.isfile(config.resume_from):
        logging.info(f"Resuming training from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load loss histories
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_mse_losses = checkpoint.get('train_mse_losses', [])
        val_mse_losses = checkpoint.get('val_mse_losses', [])
        train_spec_losses = checkpoint.get('train_spec_losses', [])
        val_spec_losses = checkpoint.get('val_spec_losses', [])

        logging.info(f"Resumed from epoch {start_epoch}. Best validation loss was {best_val_loss:.4f}.")
    else:
        logging.info("--- Starting New Training Run ---")
        logging.info(f"Run Name: {config.run_name}")
        logging.info("Configuration:")
        for key, value in vars(config).items():
            logging.info(f"  {key}: {value}")
        logging.info("\nModel Architecture:")
        logging.info(model)
        logging.info(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
        logging.info(f"\nTraining set size: {len(train_dataset)}")
        logging.info(f"Validation set size: {len(val_dataset)}\n")

    run_output_path = os.path.join(config.output_path, config.run_name)
    
    for epoch in range(start_epoch, config.epochs):
        logging.info(f"--- Epoch {epoch+1}/{config.epochs} ---")
        
        # --- Training Loop ---
        model.train()
        train_loss, train_mse_loss, train_spec_loss = 0, 0, 0
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for signals in pbar_train:
            signals = signals.to(device)
            t = diffusion.sample_timesteps(signals.shape[0]).to(device)
            
            x_t, noise = diffusion.q_sample(signals, t)
            predicted_noise = model(x_t, t)
            mse_loss = mse(noise, predicted_noise)

            # -------- denoise to x0_hat ---------------------------------------
            alpha_bar = diffusion.alpha_hat[t].view(-1, 1, 1)   # shape [B,1,1]
            x0_hat = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) \
                    / torch.sqrt(alpha_bar)                     # [B, C, T]  C=1
            x0_hat = x0_hat.squeeze(1)                           # [B, T]
            # -----------------------------------------------------------------

            spec_loss = spectral_l1(signals.squeeze(1), x0_hat)
            loss = mse_loss + config.spec_loss_weight * spec_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_spec_loss += spec_loss.item()
            pbar_train.set_postfix(MSE=mse_loss.item(), Spectral=spec_loss.item())

        # Append average losses for the epoch
        train_losses.append(train_loss / len(train_dataloader))
        train_mse_losses.append(train_mse_loss / len(train_dataloader))
        train_spec_losses.append(train_spec_loss / len(train_dataloader))
        logging.info(f"Epoch {epoch+1} average training loss: {train_losses[-1]:.4f}")
        
        # --- Validation Loop ---
        model.eval()
        val_loss, val_mse_loss, val_spec_loss = 0, 0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}")
            for signals in pbar_val:
                signals = signals.to(device)
                t = diffusion.sample_timesteps(signals.shape[0]).to(device)
                x_t, noise = diffusion.q_sample(signals, t)
                predicted_noise = model(x_t, t)
                mse_loss = mse(noise, predicted_noise)
                # -------- denoise to x0_hat ---------------------------------------
                alpha_bar = diffusion.alpha_hat[t].view(-1, 1, 1)   # shape [B,1,1]
                x0_hat = (x_t - torch.sqrt(1 - alpha_bar) * predicted_noise) \
                        / torch.sqrt(alpha_bar)                     # [B, C, T]  C=1
                x0_hat = x0_hat.squeeze(1)                           # [B, T]
                # -----------------------------------------------------------------

                spec_loss = spectral_l1(signals.squeeze(1), x0_hat)
                
                val_loss += mse_loss.item() + config.spec_loss_weight * spec_loss.item()
                val_mse_loss += mse_loss.item()
                val_spec_loss += spec_loss.item()
                pbar_val.set_postfix(MSE=mse_loss.item(), Spectral=spec_loss.item())
        
        # Append average losses for the epoch
        val_losses.append(val_loss / len(val_dataloader))
        val_mse_losses.append(val_mse_loss / len(val_dataloader))
        val_spec_losses.append(val_spec_loss / len(val_dataloader))
        logging.info(f"Epoch {epoch+1} average validation loss: {val_losses[-1]:.4f}")

        # --- Save Losses Every Epoch ---
        loss_path = os.path.join(run_output_path, "losses.npz")
        np.savez(loss_path, 
                 train_losses=np.array(train_losses), val_losses=np.array(val_losses),
                 train_mse_losses=np.array(train_mse_losses), val_mse_losses=np.array(val_mse_losses),
                 train_spec_losses=np.array(train_spec_losses), val_spec_losses=np.array(val_spec_losses)
        )

        # --- Save Best Model ---
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_path = os.path.join(run_output_path, "models", "best_ckpt.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses, 'val_losses': val_losses,
                'train_mse_losses': train_mse_losses, 'val_mse_losses': val_mse_losses,
                'train_spec_losses': train_spec_losses, 'val_spec_losses': val_spec_losses
            }, best_model_path)
            logging.info(f"New best validation loss. Saved model to {best_model_path}")
        
        # --- Sampling and Checkpoint Saving ---
        if (epoch + 1) % 10 == 0:
            logging.info("Sampling new signals for comparison...")
            
            # Create a directory for this epoch's samples
            epoch_sample_dir = os.path.join(run_output_path, "samples", f"epoch_{epoch+1}")
            os.makedirs(epoch_sample_dir, exist_ok=True)
            
            # Get random ground truth samples from the validation set to compare against
            val_indices = np.random.choice(len(val_dataset), size=3 * 3, replace=False) # 3 plots * 3 samples
            ground_truth_batch = torch.stack([val_dataset.dataset[val_dataset.indices[i]] for i in val_indices]).to(device)

            for i in range(3): # Generate 3 comparison plots
                # Generate a batch of new samples
                sampled_signals = diffusion.sample(model, n=3)
                
                # Select a slice of ground truth samples for this specific plot
                start_idx = i * 3
                end_idx = start_idx + 3
                gt_samples_for_plot = ground_truth_batch[start_idx:end_idx]

                # Save the comparison plot
                plot_path = os.path.join(epoch_sample_dir, f"comparison_{i+1}.png")
                # Note the change in function name and argument order
                plot_reconstructions(ground_truth=gt_samples_for_plot, reconstructions=sampled_signals, path=plot_path, num_samples=3)
            
            logging.info(f"Saved 3 sample comparison plots to {epoch_sample_dir}")
            
            # Plot the loss curve subplots up to this point
            loss_subplots_path = os.path.join(epoch_sample_dir, "loss_subplots.png")
            plot_loss_subplots(loss_subplots_path,
                               total_train=train_losses, total_val=val_losses,
                               mse_train=train_mse_losses, mse_val=val_mse_losses,
                               spec_train=train_spec_losses, spec_val=val_spec_losses)
            logging.info(f"Saved intermediate loss curve subplot to {loss_subplots_path}")

            # Plot the comprehensive loss curve up to this point
            all_loss_path = os.path.join(epoch_sample_dir, "all_loss_curves.png")
            plot_all_loss_curves(all_loss_path,
                                 total_train=train_losses, total_val=val_losses,
                                 mse_train=train_mse_losses, mse_val=val_mse_losses,
                                 spec_train=train_spec_losses, spec_val=val_spec_losses)
            logging.info(f"Saved intermediate comprehensive loss curve to {all_loss_path}")

            # Save periodic checkpoint
            model_path = os.path.join(run_output_path, "models", f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses, 'val_losses': val_losses,
                'train_mse_losses': train_mse_losses, 'val_mse_losses': val_mse_losses,
                'train_spec_losses': train_spec_losses, 'val_spec_losses': val_spec_losses
            }, model_path)
            logging.info(f"Saved periodic checkpoint to {model_path}")

    # --- Save Final Plots ---
    logging.info("Training complete. Plotting final loss curves.")
    
    # Save subplot plot
    subplots_path = os.path.join(run_output_path, "loss_subplots.png")
    plot_loss_subplots(subplots_path,
                     total_train=train_losses, total_val=val_losses,
                     mse_train=train_mse_losses, mse_val=val_mse_losses,
                     spec_train=train_spec_losses, spec_val=val_spec_losses)
    logging.info(f"Saved loss curves subplot to {subplots_path}")

    # Save comprehensive plot
    all_plot_path = os.path.join(run_output_path, "all_loss_curves.png")
    plot_all_loss_curves(all_plot_path,
                         total_train=train_losses, total_val=val_losses,
                         mse_train=train_mse_losses, mse_val=val_mse_losses,
                         spec_train=train_spec_losses, spec_val=val_spec_losses)
    logging.info(f"Saved comprehensive loss curves plot to {all_plot_path}")

# --- Main Execution ---
def main():
    config = get_config()
    parser = argparse.ArgumentParser(description='Train an EDA Diffusion Model')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--device', type=str, default=config.device, help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path)
    parser.add_argument('--fold_for_training', type=int, default=config.fold_for_training, help="The fold number to train on (e.g., 17 loads fold_17.npz).")
    parser.add_argument('--spec_loss_weight', type=float, default=config.spec_loss_weight, help="Weight for the spectral loss component.")
    parser.add_argument('--dataset_percentage', type=float, default=config.dataset_percentage, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')

    args = parser.parse_args()
    
    # Update config with parsed args
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    # Setup output directories and logging
    run_output_dir = os.path.join(config.output_path, args.run_name)
    os.makedirs(os.path.join(run_output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, "samples"), exist_ok=True)
    setup_logging(config.run_name, config.output_path)


        
    train(config)

if __name__ == '__main__':
    main() 