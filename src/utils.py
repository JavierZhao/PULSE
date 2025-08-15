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
    The plot will contain four subplots for total, reconstruction, and alignment losses, and learning rate.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    fig.suptitle('MAE Training Curves', fontsize=16)

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

    # Plot Learning Rate
    axes[3].plot(losses_df['epoch'], losses_df['lr'], label='Learning Rate')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Learning Rate')
    axes[3].set_title('Learning Rate Schedule')
    axes[3].legend()
    axes[3].grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path)
    plt.close(fig)

def plot_kd_losses(losses_df, path):
    """
    Plots and saves KD training curves given a DataFrame with keys:
      - epoch
      - train_total, val_total
      - train_kd_hid, val_kd_hid
      - train_kd_emb, val_kd_emb
      - train_recon,   val_recon
      - val_perp (optional)
      - lr (optional)
    """
    has_perp = 'val_perp' in losses_df.columns
    has_lr = 'lr' in losses_df.columns

    num_rows = 5 if has_lr else 4
    if has_perp:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), sharex=True)
    axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    fig.suptitle('KD Training Curves', fontsize=16)

    row = 0
    # Total Loss
    axes[row].plot(losses_df['epoch'], losses_df['train_total'], label='Train Total')
    axes[row].plot(losses_df['epoch'], losses_df['val_total'], label='Val Total', linestyle='--')
    axes[row].set_ylabel('Loss')
    axes[row].set_title('Total Loss')
    axes[row].legend(); axes[row].grid(True)
    row += 1

    # Hidden KD Loss
    if 'train_kd_hid' in losses_df.columns and 'val_kd_hid' in losses_df.columns:
        axes[row].plot(losses_df['epoch'], losses_df['train_kd_hid'], label='Train KD Hidden')
        axes[row].plot(losses_df['epoch'], losses_df['val_kd_hid'], label='Val KD Hidden', linestyle='--')
        axes[row].set_ylabel('Loss')
        axes[row].set_title('Token-level KD (Hidden Layers)')
        axes[row].legend(); axes[row].grid(True)
        row += 1

    # Embedding KD Loss
    if 'train_kd_emb' in losses_df.columns and 'val_kd_emb' in losses_df.columns:
        axes[row].plot(losses_df['epoch'], losses_df['train_kd_emb'], label='Train KD Embedding')
        axes[row].plot(losses_df['epoch'], losses_df['val_kd_emb'], label='Val KD Embedding', linestyle='--')
        axes[row].set_ylabel('Loss')
        axes[row].set_title('Pooled Embedding KD (Final Layer)')
        axes[row].legend(); axes[row].grid(True)
        row += 1

    # Reconstruction Loss (optional)
    if 'train_recon' in losses_df.columns and 'val_recon' in losses_df.columns:
        axes[row].plot(losses_df['epoch'], losses_df['train_recon'], label='Train Recon')
        axes[row].plot(losses_df['epoch'], losses_df['val_recon'], label='Val Recon', linestyle='--')
        axes[row].set_ylabel('Loss')
        axes[row].set_title('Reconstruction Loss')
        axes[row].legend(); axes[row].grid(True)
        row += 1

    # Perp Loss (validation only)
    if has_perp:
        axes[row].plot(losses_df['epoch'], losses_df['val_perp'], label='Val Perp', color='tab:purple')
        axes[row].set_ylabel('Loss')
        axes[row].set_title('Decor Perp (Validation)')
        axes[row].legend(); axes[row].grid(True)
        row += 1

    # Learning Rate (optional)
    if has_lr:
        axes[row].plot(losses_df['epoch'], losses_df['lr'], label='Learning Rate')
        axes[row].set_xlabel('Epoch')
        axes[row].set_ylabel('LR')
        axes[row].set_title('Learning Rate Schedule')
        axes[row].legend(); axes[row].grid(True)

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

def plot_single_reconstruction(model, signal, output_path, title="Signal Reconstruction"):
    """
    Plots and saves a single signal reconstruction.
    
    Args:
        model (nn.Module): The trained model.
        signal (torch.Tensor): A single time series signal tensor, e.g., (1, 1, L).
        output_path (str): Path to save the plot image.
        title (str): The title for the plot.
    """
    model.eval()
    with torch.no_grad():
        original_patched, pred_patched, mask = model.forward_with_visualization(signal)

    original_signal = model.unpatchify(original_patched).squeeze().cpu().numpy()
    pred_signal = model.unpatchify(pred_patched).squeeze().cpu().numpy()
    mask = mask.repeat_interleave(model.window_len).cpu().numpy()

    fig = plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.plot(original_signal, label="Original Signal")
    plt.plot(pred_signal, label="Reconstructed Signal", linestyle='--')
    
    bottom, top = plt.ylim()
    plt.fill_between(np.arange(len(mask)), bottom, top, where=mask==1, color='red', alpha=0.2, label='Masked Area')

    plt.xlabel("Time Steps")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close(fig)

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