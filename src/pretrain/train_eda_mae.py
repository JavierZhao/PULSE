import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.backbone_registry import MAE_BACKBONE_REGISTRY
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

def resolve_teacher_setup(args):
    aliases = {
        'csms': 'cmsc',
    }
    teacher_for = aliases.get(args.teacher_for.lower(), args.teacher_for.lower())
    supported_targets = {'mae', 'multimae', 'clip', 'cmsc', 'contrastive', 'vicreg', 'autoregressive'}
    if teacher_for not in supported_targets:
        raise ValueError(
            f"Unknown teacher_for '{args.teacher_for}'. "
            f"Choose from: ['mae', 'multimae', 'clip', 'cmsc', 'csms', 'contrastive', 'vicreg', 'autoregressive']"
        )

    backbone = args.backbone
    if backbone not in MAE_BACKBONE_REGISTRY:
        raise ValueError(
            f"Masked-reconstruction teacher training requires a MAE backbone. "
            f"Got backbone='{backbone}', choose from {sorted(MAE_BACKBONE_REGISTRY.keys())}"
        )

    return teacher_for, backbone


def build_eda_teacher_model(args, device, backbone):
    backbone_cls = MAE_BACKBONE_REGISTRY.get(backbone)
    if backbone_cls is None:
        raise ValueError(
            f"Unknown MAE backbone '{backbone}'. Choose from: {sorted(MAE_BACKBONE_REGISTRY.keys())}"
        )

    model_kwargs = {
        'modality_name': 'eda',
        'sig_len': args.signal_length,
        'window_len': args.patch_window_len,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'decoder_embed_dim': args.decoder_embed_dim,
        'decoder_depth': args.decoder_depth,
        'decoder_num_heads': args.decoder_num_heads,
        'mlp_ratio': args.mlp_ratio,
        'decoder_mlp_ratio': args.decoder_mlp_ratio,
        'private_mask_ratio': 1.0,
    }

    if backbone == 'resnet1d':
        model_kwargs.update({
            'base_channels': args.resnet_base_channels,
            'num_blocks_per_stage': args.resnet_num_blocks_per_stage,
            'kernel_size': args.resnet_kernel_size,
        })
    elif backbone == 'tcn':
        model_kwargs.update({
            'tcn_channels': args.tcn_channels,
            'kernel_size': args.tcn_kernel_size,
            'dropout': args.tcn_dropout,
        })

    return backbone_cls(**model_kwargs).to(device)


def compute_teacher_loss(model, sig, mask_ratio, device):
    ids_shuffle, ids_restore, ids_keep = model.propose_masking(
        len(sig), model.num_patches, mask_ratio, device
    )
    private, shared, mask = model.forward_encoder(sig, mask_ratio, ids_shuffle, ids_restore, ids_keep)
    rec = model.forward_decoder(private, shared, ids_restore)
    return model.reconstruction_loss(sig, rec, mask)


def validate_teacher(model, dataloader, device, mask_ratio):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            sig = batch['eda'].to(device)
            loss = compute_teacher_loss(model, sig, mask_ratio, device)
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
        '--backbone', args.backbone,
        '--signal_length', str(args.signal_length),
        '--patch_window_len', str(args.patch_window_len),
        '--embed_dim', str(args.embed_dim),
        '--depth', str(args.depth),
        '--num_heads', str(args.num_heads),
        '--decoder_embed_dim', str(args.decoder_embed_dim),
        '--decoder_depth', str(args.decoder_depth),
        '--decoder_num_heads', str(args.decoder_num_heads),
        '--mlp_ratio', str(args.mlp_ratio),
        '--decoder_mlp_ratio', str(args.decoder_mlp_ratio),
    ]
    if args.backbone == 'resnet1d':
        cmd.extend([
            '--resnet_base_channels', str(args.resnet_base_channels),
            '--resnet_num_blocks_per_stage', str(args.resnet_num_blocks_per_stage),
            '--resnet_kernel_size', str(args.resnet_kernel_size),
        ])
    elif args.backbone == 'tcn':
        cmd.extend([
            '--tcn_channels', str(args.tcn_channels),
            '--tcn_kernel_size', str(args.tcn_kernel_size),
            '--tcn_dropout', str(args.tcn_dropout),
        ])
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
    # Flash Attention 2 and Memory-Efficient Attention can trigger
    # "CUDA error: invalid argument" on RTX 3090/4090 for this workload.
    # Force the math backend for stability, matching other pretraining scripts.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    models_path = os.path.join(run_output_path, "models")
    os.makedirs(models_path, exist_ok=True)

    teacher_for, resolved_backbone = resolve_teacher_setup(args)
    args.teacher_for = teacher_for
    args.backbone = resolved_backbone

    if not (0.0 <= args.mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0.0, 1.0], got {args.mask_ratio}")

    model = build_eda_teacher_model(args, device, resolved_backbone)
    
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
        logging.info(f"  resolved_backbone: {resolved_backbone}")
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

            loss = compute_teacher_loss(model, sig, args.mask_ratio, device)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix(Loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = validate_teacher(model, val_dataloader, device, args.mask_ratio)
        
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
    parser = argparse.ArgumentParser(description='Train an EDA teacher with masked reconstruction pretraining for KD.')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/j-jepa-vol/PULSE/results/eda_mae", help='Directory to save logs and models')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for training/validation')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--teacher_for', type=str, default='mae',
                        choices=['mae', 'multimae', 'clip', 'cmsc', 'csms', 'contrastive', 'vicreg', 'autoregressive'],
                        help='Student pretraining family this teacher is intended for (metadata only). EDA teacher objective remains masked reconstruction.')
    parser.add_argument('--backbone', type=str, default='transformer', choices=sorted(MAE_BACKBONE_REGISTRY.keys()),
                        help='EDA teacher MAE backbone: transformer, resnet1d, or tcn.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset_percentage', type=float, default=1.0, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    
    # Model specific arguments
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension of the encoder.')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the encoder.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the encoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='Embedding dimension of the decoder.')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Depth of the decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=16, help='Number of heads in the decoder.')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio for the encoder.')
    parser.add_argument('--decoder_mlp_ratio', type=float, default=4.0, help='MLP ratio for the decoder.')

    # ResNet1D-specific arguments
    parser.add_argument('--resnet_base_channels', type=int, default=232, help='Base channels for ResNet1D encoder.')
    parser.add_argument('--resnet_num_blocks_per_stage', type=int, default=2, help='Residual blocks per ResNet1D stage.')
    parser.add_argument('--resnet_kernel_size', type=int, default=7, help='Kernel size for ResNet1D convolutions.')

    # TCN-specific arguments
    parser.add_argument('--tcn_channels', type=int, default=1520, help='Hidden channels for TCN encoder.')
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help='Kernel size for TCN convolutions.')
    parser.add_argument('--tcn_dropout', type=float, default=0.1, help='Dropout rate for TCN blocks.')
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
