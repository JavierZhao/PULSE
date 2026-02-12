import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset
from src.utils import plot_single_loss_curve, plot_single_reconstruction

def setup_logging(run_name, output_path):
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, losses, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'losses': losses
    }, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer, scheduler, device):
    if not os.path.isfile(path):
        logging.warning(f"Checkpoint file not found at {path}. Starting from scratch.")
        return 0, float('inf'), []
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    losses = checkpoint.get('losses', [])
    logging.info(f"Resumed from epoch {epoch}. Best validation loss was {best_val_loss:.4f}.")
    return epoch, best_val_loss, losses

def validate(model, dataloader, device, mask_ratio):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            sig = batch['eda'].to(device)
            ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, mask_ratio, device)
            private, shared, mask = model.forward_encoder(sig, mask_ratio, ids_shuffle, ids_restore, ids_keep)
            rec = model.forward_decoder(private, shared, ids_restore)
            loss = model.reconstruction_loss(sig, rec, mask)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def run_periodic_evaluation(run_output_path, models_path, args, epoch_plus_one):
    """Invoke evaluate_eda_mae.py to generate a reconstruction plot and rename it per-epoch."""
    ckpt_path = os.path.join(models_path, "best_ckpt.pt")
    if not os.path.isfile(ckpt_path):
        logging.info("Periodic evaluation skipped (no best_ckpt.pt yet).")
        return
    eval_script = os.path.join(os.path.dirname(__file__), 'evaluate_eda_mae.py')
    cmd = [
        sys.executable, eval_script,
        '--checkpoint_path', ckpt_path,
        '--data_path', args.data_path,
        '--output_path', run_output_path,
        '--fold_number', str(args.fold_number),
        '--device', args.device,
        '--signal_length', str(args.signal_length),
        '--patch_window_len', str(args.patch_window_len),
    ]
    try:
        logging.info(f"Running periodic evaluation at epoch {epoch_plus_one}...")
        subprocess.run(cmd, check=True)
        # Rename output plot with epoch suffix if present
        out_plot = os.path.join(run_output_path, 'eda_reconstruction_example.png')
        if os.path.isfile(out_plot):
            new_name = os.path.join(run_output_path, f'eda_reconstruction_example_epoch_{epoch_plus_one}.png')
            try:
                if os.path.isfile(new_name):
                    os.remove(new_name)
                os.rename(out_plot, new_name)
                logging.info(f"Saved periodic reconstruction plot to {new_name}")
            except Exception as re:
                logging.warning(f"Could not rename reconstruction plot: {re}")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Periodic evaluation failed: {e}")


def train(args):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)
    
    if not (0.0 <= args.mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0.0, 1.0], got {args.mask_ratio}")
    
    model = CheapSensorMAE(
        modality_name='eda',
        sig_len=args.signal_length,
        window_len=args.patch_window_len,
        private_mask_ratio=1.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    start_epoch, best_val_loss, losses = 0, float('inf'), []
    if args.resume_from:
        start_epoch, best_val_loss, losses = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
    else:
        logging.info("--- Starting New EDA MAE Training Run ---")
        logging.info(f"Run Name: {args.run_name}")
        logging.info("Configuration:")
        for key, value in vars(args).items():
            logging.info(f"  {key}: {value}")
        logging.info("\nModel Architecture:")
        logging.info(str(model))
        logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info("-" * 50)

    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train')

    if args.dataset_percentage < 1.0:
        original_size = len(full_train_dataset)
        subset_size = int(original_size * args.dataset_percentage)
        generator = torch.Generator().manual_seed(args.seed)
        full_train_dataset, _ = random_split(full_train_dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {args.dataset_percentage*100:.0f}% of the dataset: {len(full_train_dataset)} out of {original_size} windows.")

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get a fixed batch from the validation set for visualization
    vis_batch = next(iter(val_dataloader))
    vis_sig = vis_batch['eda'][0].unsqueeze(0).to(device) # Get the first sample and add batch dim

    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in pbar_train:
            sig = batch['eda'].to(device)
            optimizer.zero_grad()
            
            ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, args.mask_ratio, device)
            private, shared, mask = model.forward_encoder(sig, args.mask_ratio, ids_shuffle, ids_restore, ids_keep)
            rec = model.forward_decoder(private, shared, ids_restore)
            loss = model.reconstruction_loss(sig, rec, mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix(Loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = validate(model, val_dataloader, device, args.mask_ratio)
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        losses.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        losses_df = pd.DataFrame(losses)
        
        loss_path = os.path.join(run_output_path, "losses.npz")
        np.savez(loss_path, epoch=losses_df['epoch'].values, train_loss=losses_df['train_loss'].values, val_loss=losses_df['val_loss'].values)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(models_path, "best_ckpt.pt")
            save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss, losses_df.to_dict('records'), best_model_path)
        
        if (epoch + 1) % 5 == 0:
            plot_path = os.path.join(run_output_path, "loss_curves.png")
            plot_single_loss_curve(losses_df, plot_path)
            logging.info(f"Saved loss plot to {plot_path}")

            # Plot reconstructions
            recon_plot_path = os.path.join(run_output_path, f"reconstruction_epoch_{epoch+1}.png")
            plot_single_reconstruction(model, vis_sig, recon_plot_path, title=f"EDA Reconstruction Epoch {epoch+1}")
            logging.info(f"Saved reconstruction plot to {recon_plot_path}")

        # Periodic external evaluation using evaluate_eda_mae
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            run_periodic_evaluation(run_output_path, models_path, args, epoch + 1)

    logging.info("Training complete.")
    final_plot_path = os.path.join(run_output_path, "loss_curves.png")
    plot_single_loss_curve(pd.DataFrame(losses), final_plot_path)
    logging.info(f"Saved final loss plot to {final_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a Teacher MAE for EDA')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/j-jepa-vol/PULSE/results/eda_mae", help='Directory to save logs and models')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for training/validation')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_percentage', type=float, default=1.0, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    
    # Model specific arguments
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Fraction of patches to mask during MAE training/validation (0.0 to 1.0).')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--eval_every', type=int, default=10, help='Run evaluate_eda_mae every N epochs (0 to disable)')
    args = parser.parse_args()

    setup_logging(args.run_name, args.output_path)
    train(args)

if __name__ == '__main__':
    main() 