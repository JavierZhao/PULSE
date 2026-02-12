import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained EDA MAE model and plot reconstruction.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory.')
    parser.add_argument('--output_path', type=str, default=".", help='Directory to save the reconstruction plot and log file.')
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for the test set.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Specify the device for evaluation.")
    
    # Model specific arguments (should match the trained model)
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    setup_logging(args.output_path)

    device = torch.device(args.device)

    # --- Load Model ---
    logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    # Initialize model architecture
    model = CheapSensorMAE(
        modality_name='eda',
        sig_len=args.signal_length,
        window_len=args.patch_window_len,
    ).to(device)
    
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