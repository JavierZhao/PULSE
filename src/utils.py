import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_reconstruction(model, signal, title="MAE Reconstruction"):
    """
    Visualizes the original time series and the MAE's reconstruction.

    Args:
        model (CheapSensorMAE): The trained MAE model.
        signal (torch.Tensor): A single time series signal tensor (1, 1, L).
        title (str): The title for the plot.
    """
    model.eval()
    with torch.no_grad():
        original_patched, pred_patched, mask = model.forward_with_visualization(signal)

    # Unpatchify to get the full time series
    original_signal = model.unpatchify(original_patched).squeeze().cpu().numpy()
    pred_signal = model.unpatchify(pred_patched).squeeze().cpu().numpy()
    
    # The mask is on patches, so we need to repeat it for each value in the patch
    mask = mask.repeat_interleave(model.window_len).cpu().numpy()

    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.plot(original_signal, label="Original Signal")
    plt.plot(pred_signal, label="Reconstructed Signal", linestyle='--')
    
    # Create a shaded region for the masked parts
    plt.fill_between(np.arange(len(mask)), 
                     plt.ylim()[0], 
                     plt.ylim()[1], 
                     where=mask==1, 
                     color='red', 
                     alpha=0.2, 
                     label='Masked Area')

    plt.xlabel("Time Steps")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mae_losses(losses_df, path):
    """
    Plots and saves the training and validation loss curves for the MAE model.
    The plot will contain three subplots for total, reconstruction, and alignment losses.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('MAE Loss Curves', fontsize=16)

    # Plot Total Loss
    axes[0].plot(losses_df['epoch'], losses_df['train_total_loss'], label='Train Total Loss')
    axes[0].plot(losses_df['epoch'], losses_df['val_total_loss'], label='Validation Total Loss', linestyle='--')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Reconstruction Loss
    axes[1].plot(losses_df['epoch'], losses_df['train_recon_loss'], label='Train Reconstruction Loss')
    axes[1].plot(losses_df['epoch'], losses_df['val_recon_loss'], label='Validation Reconstruction Loss', linestyle='--')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Alignment Loss
    axes[2].plot(losses_df['epoch'], losses_df['train_align_loss'], label='Train Alignment Loss')
    axes[2].plot(losses_df['epoch'], losses_df['val_align_loss'], label='Validation Alignment Loss', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Alignment Loss')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.close(fig)

def plot_single_loss_curve(losses_df, path):
    """
    Plots and saves a single training and validation loss curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses_df['epoch'], losses_df['train_loss'], label='Train Loss')
    plt.plot(losses_df['epoch'], losses_df['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('MAE Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_reconstructions(models, data_batch, device, output_path):
    """
    Plots and saves a comparison of original vs. reconstructed signals for each modality.
    """
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 12))
    fig.suptitle('Signal Reconstructions', fontsize=16)
    
    # Take the first sample from the batch for visualization
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        ax = axes[i]
        signal = data_batch[name][0].unsqueeze(0).to(device) # Shape: (1, 1, L)
        
        with torch.no_grad():
            original_patched, pred_patched, mask = model.forward_with_visualization(signal)
            
        original_signal = model.unpatchify(original_patched).squeeze().cpu().numpy()
        pred_signal = model.unpatchify(pred_patched).squeeze().cpu().numpy()
        mask = mask.repeat_interleave(model.window_len).cpu().numpy()
        
        ax.plot(original_signal, label="Original Signal")
        ax.plot(pred_signal, label="Reconstructed Signal", linestyle='--')
        
        # Get current y-limits to draw the shaded region
        bottom, top = ax.get_ylim()
        ax.fill_between(np.arange(len(mask)), bottom, top, where=mask==1, color='red', alpha=0.2, label='Masked Area')
        
        ax.set_title(f"Modality: {name.upper()}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig) 