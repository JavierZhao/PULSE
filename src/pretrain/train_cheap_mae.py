import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset
from src.utils import plot_mae_losses, plot_reconstructions

def setup_logging(run_name, output_path):
    """Configures logging to file and console."""
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

def save_checkpoint(epoch, models, optimizers, schedulers, best_val_loss, losses, path):
    """Saves the training state to a checkpoint file."""
    state = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'losses': losses
    }
    for name, model in models.items():
        state[f'{name}_model_state_dict'] = model.state_dict()
    for name, optimizer in optimizers.items():
        state[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
    for name, scheduler in schedulers.items():
        state[f'{name}_scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)
    logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(path, models, optimizers, schedulers, device):
    """Loads the training state from a checkpoint file."""
    if not os.path.isfile(path):
        logging.warning(f"Checkpoint file not found at {path}. Starting from scratch.")
        return 0, float('inf'), []
        
    checkpoint = torch.load(path, map_location=device)
    for name, model in models.items():
        model.load_state_dict(checkpoint[f'{name}_model_state_dict'])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint[f'{name}_optimizer_state_dict'])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(checkpoint[f'{name}_scheduler_state_dict'])
        
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    losses = checkpoint.get('losses', [])
    logging.info(f"Resumed from epoch {epoch}. Best validation loss was {best_val_loss:.4f}.")
    return epoch, best_val_loss, losses

def validate(models, dataloader, device, alignment_loss_weight):
    """Runs a validation loop and returns the average loss."""
    for model in models.values():
        model.eval()
    total_loss, total_recon_loss, total_align_loss = 0, 0, 0
    alignment_loss_fn = nn.CosineSimilarity(dim=1)

    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            ecg_sig, bvp_sig, acc_sig, temp_sig = [batch[k].to(device) for k in ('ecg', 'bvp', 'acc', 'temp')]
            
            # Step 1: Get all embeddings from the encoders
            all_private_embs, all_shared_embs, all_masks, all_ids_restore = {}, [], {}, {}
            for name, sig in [('ecg', ecg_sig), ('bvp', bvp_sig), ('acc', acc_sig), ('temp', temp_sig)]:
                model = models[name]
                ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, 0.75, device)
                private, shared, mask = model.forward_encoder(sig, 0.75, ids_shuffle, ids_restore, ids_keep)
                all_private_embs[name] = private
                all_shared_embs.append(shared)
                all_masks[name] = mask
                all_ids_restore[name] = ids_restore

            # Step 2: Create the aligned shared embedding by averaging
            aligned_shared_emb = torch.stack(all_shared_embs).mean(dim=0)
            
            # Step 3: Calculate reconstruction loss using the aligned shared embedding
            reconstruction_loss = 0
            for name, sig in [('ecg', ecg_sig), ('bvp', bvp_sig), ('acc', acc_sig), ('temp', temp_sig)]:
                rec = models[name].forward_decoder(all_private_embs[name], aligned_shared_emb, all_ids_restore[name])
                reconstruction_loss += models[name].reconstruction_loss(sig, rec, all_masks[name])

            # Step 4: Calculate alignment loss on the original (pre-average) shared embeddings
            loss_align = 1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[1].mean(dim=1)).mean() + \
                         1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[2].mean(dim=1)).mean() + \
                         1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[3].mean(dim=1)).mean()
            
            total_loss += (reconstruction_loss + alignment_loss_weight * loss_align).item()
            total_recon_loss += reconstruction_loss.item()
            total_align_loss += loss_align.item()

    avg_total_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_align_loss = total_align_loss / len(dataloader)
    
    return {'total': avg_total_loss, 'recon': avg_recon_loss, 'align': avg_align_loss}


def train(args):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)
    
    # --- Models, Optimizers, Schedulers ---
    models = {
        'ecg': CheapSensorMAE(modality_name='ecg', sig_len=args.signal_length, window_len=args.patch_window_len, private_mask_ratio=args.private_mask_ratio).to(device),
        'bvp': CheapSensorMAE(modality_name='bvp', sig_len=args.signal_length, window_len=args.patch_window_len, private_mask_ratio=args.private_mask_ratio).to(device),
        'acc': CheapSensorMAE(modality_name='acc', sig_len=args.signal_length, window_len=args.patch_window_len, private_mask_ratio=args.private_mask_ratio).to(device),
        'temp': CheapSensorMAE(modality_name='temp', sig_len=args.signal_length, window_len=args.patch_window_len, private_mask_ratio=args.private_mask_ratio).to(device)
    }
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=args.learning_rate) for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs) for name, opt in optimizers.items()}

    # --- Load Checkpoint ---
    start_epoch, best_val_loss, losses = 0, float('inf'), []
    if args.resume_from:
        start_epoch, best_val_loss, losses = load_checkpoint(args.resume_from, models, optimizers, schedulers, device)
    else:
        logging.info("--- Starting New Training Run ---")
        logging.info(f"Run Name: {args.run_name}")
        logging.info("Configuration:")
        for key, value in vars(args).items():
            logging.info(f"  {key}: {value}")
        
        # Log model architecture (they are all the same)
        ecg_model = models['ecg']
        logging.info("\nModel Architecture:")
        logging.info(str(ecg_model))
        logging.info(f"Total parameters: {sum(p.numel() for p in ecg_model.parameters()):,}")
        logging.info("-" * 50)

    # --- Data ---
    # Load the full training dataset for the fold
    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train')

    # Reduce dataset size based on percentage argument
    if args.dataset_percentage < 1.0:
        original_size = len(full_train_dataset)
        subset_size = int(original_size * args.dataset_percentage)
        generator = torch.Generator().manual_seed(args.seed)
        full_train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {args.dataset_percentage*100:.0f}% of the dataset: {len(full_train_dataset)} out of {original_size} windows.")

    # Split training data into train and validation sets (90/10)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get a fixed batch from the validation set for visualization
    vis_batch = next(iter(val_dataloader))
    
    alignment_loss_fn = nn.CosineSimilarity(dim=1)
    
    for epoch in range(start_epoch, args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        # --- Training Loop ---
        for model in models.values():
            model.train()
        train_total_loss, train_recon_loss, train_align_loss = 0, 0, 0
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in pbar_train:
            ecg_sig, bvp_sig, acc_sig, temp_sig = [batch[k].to(device) for k in ('ecg', 'bvp', 'acc', 'temp')]
            
            # Zero gradients
            for opt in optimizers.values():
                opt.zero_grad()
                
            # Step 1: Get all embeddings from the encoders
            all_private_embs, all_shared_embs, all_masks, all_ids_restore = {}, [], {}, {}
            for name, sig in [('ecg', ecg_sig), ('bvp', bvp_sig), ('acc', acc_sig), ('temp', temp_sig)]:
                model = models[name]
                ids_shuffle, ids_restore, ids_keep = model.propose_masking(len(sig), model.num_patches, 0.75, device)
                private, shared, mask = model.forward_encoder(sig, 0.75, ids_shuffle, ids_restore, ids_keep)
                all_private_embs[name] = private
                all_shared_embs.append(shared)
                all_masks[name] = mask
                all_ids_restore[name] = ids_restore

            # Step 2: Create the aligned shared embedding by averaging
            aligned_shared_emb = torch.stack(all_shared_embs).mean(dim=0)

            # Step 3: Calculate reconstruction loss using the aligned shared embedding
            reconstruction_loss = 0
            for name, sig in [('ecg', ecg_sig), ('bvp', bvp_sig), ('acc', acc_sig), ('temp', temp_sig)]:
                rec = models[name].forward_decoder(all_private_embs[name], aligned_shared_emb, all_ids_restore[name])
                reconstruction_loss += models[name].reconstruction_loss(sig, rec, all_masks[name])

            # Step 4: Calculate alignment loss on the original (pre-average) shared embeddings
            loss_align = 1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[1].mean(dim=1)).mean() + \
                         1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[2].mean(dim=1)).mean() + \
                         1 - alignment_loss_fn(all_shared_embs[0].mean(dim=1), all_shared_embs[3].mean(dim=1)).mean()
            
            # Step 5: Combine losses
            loss = reconstruction_loss + args.alignment_loss_weight * loss_align

            loss.backward()
            for opt in optimizers.values():
                opt.step()
                
            train_total_loss += loss.item()
            train_recon_loss += reconstruction_loss.item()
            train_align_loss += loss_align.item()
            pbar_train.set_postfix(Loss=loss.item())

        avg_train_total_loss = train_total_loss / len(train_dataloader)
        avg_train_recon_loss = train_recon_loss / len(train_dataloader)
        avg_train_align_loss = train_align_loss / len(train_dataloader)
        
        # --- Validation Loop ---
        val_losses = validate(models, val_dataloader, device, args.alignment_loss_weight)
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_total_loss:.4f} | Val Loss: {val_losses['total']:.4f}")

        # Update learning rate
        for sch in schedulers.values():
            sch.step()

        # --- Save Losses and Checkpoints ---
        losses.append({
            'epoch': epoch + 1,
            'train_total_loss': avg_train_total_loss, 'train_recon_loss': avg_train_recon_loss, 'train_align_loss': avg_train_align_loss,
            'val_total_loss': val_losses['total'], 'val_recon_loss': val_losses['recon'], 'val_align_loss': val_losses['align']
        })
        losses_df = pd.DataFrame(losses)
        
        # Save losses as a .npz file
        loss_path = os.path.join(run_output_path, "losses.npz")
        np.savez(loss_path,
                 epoch=losses_df['epoch'].values,
                 train_total_loss=losses_df['train_total_loss'].values,
                 train_recon_loss=losses_df['train_recon_loss'].values,
                 train_align_loss=losses_df['train_align_loss'].values,
                 val_total_loss=losses_df['val_total_loss'].values,
                 val_recon_loss=losses_df['val_recon_loss'].values,
                 val_align_loss=losses_df['val_align_loss'].values)

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(models_path, "best_ckpt.pt")
            save_checkpoint(epoch + 1, models, optimizers, schedulers, best_val_loss, losses_df.to_dict('records'), best_model_path)
        
        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_path = os.path.join(run_output_path, f"loss_curves.png")
            plot_mae_losses(losses_df, plot_path)
            logging.info(f"Saved loss plot to {plot_path}")
            
            # Plot reconstructions
            recon_plot_path = os.path.join(run_output_path, f"reconstructions_epoch_{epoch+1}.png")
            plot_reconstructions(models, vis_batch, device, recon_plot_path)
            logging.info(f"Saved reconstruction plot to {recon_plot_path}")

    logging.info("Training complete.")
    # --- Final Plot ---
    final_plot_path = os.path.join(run_output_path, "loss_curves.png")
    plot_mae_losses(pd.DataFrame(losses), final_plot_path)
    logging.info(f"Saved final loss plot to {final_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a Multi-Modal Cheap Sensor MAE')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--data_path', type=str, default="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/fd24T/zzhao3/EDA/results/cheap_maes", help='Directory to save logs and models')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for training/validation')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_percentage', type=float, default=1.0, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")
    parser.add_argument('--alignment_loss_weight', type=float, default=1.0, help="Weight for the alignment loss component.")
    parser.add_argument('--device', type=str, default='cuda:15' if torch.cuda.is_available() else 'cpu', help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    
    # Model specific arguments
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--private_mask_ratio', type=float, default=0.5, help='Ratio of private to shared embeddings')
    args = parser.parse_args()

    setup_logging(args.run_name, args.output_path)
    train(args)

if __name__ == '__main__':
    main() 