import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.backbone_registry import MAE_BACKBONE_REGISTRY
from src.data.wesad_dataset import WESADDataset
from src.utils import plot_single_reconstruction

def setup_logging(output_path):
    """Configures basic logging for the evaluation script."""
    log_path = os.path.join(output_path, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def build_eda_teacher_model(args, device):
    backbone_cls = MAE_BACKBONE_REGISTRY.get(args.backbone)
    if backbone_cls is None:
        raise ValueError(
            f"Unknown backbone '{args.backbone}'. Choose from: {sorted(MAE_BACKBONE_REGISTRY.keys())}"
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
    if args.backbone == 'resnet1d':
        model_kwargs.update({
            'base_channels': args.resnet_base_channels,
            'num_blocks_per_stage': args.resnet_num_blocks_per_stage,
            'kernel_size': args.resnet_kernel_size,
        })
    elif args.backbone == 'tcn':
        model_kwargs.update({
            'tcn_channels': args.tcn_channels,
            'kernel_size': args.tcn_kernel_size,
            'dropout': args.tcn_dropout,
        })

    return backbone_cls(**model_kwargs).to(device)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained EDA MAE model and plot reconstruction.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory.')
    parser.add_argument('--output_path', type=str, default=".", help='Directory to save the reconstruction plot and log file.')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for the test set.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Specify the device for evaluation.")
    parser.add_argument('--backbone', type=str, default='transformer', choices=sorted(MAE_BACKBONE_REGISTRY.keys()),
                        help='EDA teacher MAE backbone used for training.')
    
    # Model specific arguments (should match the trained model)
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

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    setup_logging(args.output_path)

    device = torch.device(args.device)

    # --- Load Model ---
    logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    # Initialize model architecture
    model = build_eda_teacher_model(args, device)
    
    # Load the state dict
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info("Model loaded successfully.")

    # --- Load Data ---
    logging.info(f"Loading test data for fold {args.fold_number}...")
    test_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True) # Shuffle to get a random sample each time

    if not test_dataset:
        logging.error("Test dataset is empty. Cannot perform evaluation.")
        return

    # --- Get a Sample and Plot ---
    logging.info("Fetching a sample from the test set for visualization...")
    sample_batch = next(iter(test_dataloader))
    signal = sample_batch['eda'].to(device)

    output_plot_path = os.path.join(args.output_path, "eda_reconstruction_example.png")
    
    logging.info(f"Generating reconstruction plot and saving to {output_plot_path}...")
    plot_single_reconstruction(
        model, 
        signal, 
        output_plot_path,
        title=f"EDA Reconstruction from Checkpoint"
    )
    logging.info("Plot saved successfully.")
    logging.info("Evaluation complete.")

if __name__ == '__main__':
    main() 
