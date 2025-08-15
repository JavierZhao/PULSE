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
from sklearn.metrics import f1_score, accuracy_score

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.CheapSensorMAE import CheapSensorMAE
from src.data.wesad_dataset import WESADDataset

class StressClassifier(nn.Module):
    def __init__(self, models, embed_dim, freeze_backbone=False, linear_classifier=False, only_shared=False, fuse_embeddings=False, modalities=None):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.modalities = modalities if modalities is not None else ['ecg', 'bvp', 'acc', 'temp']
        if freeze_backbone:
            for param in self.models.parameters():
                param.requires_grad = False
        
        num_modalities = len(self.modalities)
        if fuse_embeddings and only_shared:
            fusion_dim = embed_dim
        elif fuse_embeddings and not only_shared:
            fusion_dim = 2 * embed_dim
        elif not fuse_embeddings and only_shared:
            fusion_dim = num_modalities * embed_dim
        else:
            fusion_dim = 2 * num_modalities * embed_dim
        self.fusion_dim = fusion_dim
        self.only_shared = only_shared
        self.fuse_embeddings = fuse_embeddings
        if linear_classifier:
            self.classifier = nn.Sequential(nn.Linear(self.fusion_dim, 1))
        else:
            self.classifier = nn.Sequential( 
                nn.Linear(self.fusion_dim, 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(4, 1)  # 1 class (binary)
            )

    def forward(self, ecg_sig, bvp_sig, acc_sig, temp_sig):
        # Map input tensors by modality name
        input_map = {
            'ecg': ecg_sig,
            'bvp': bvp_sig,
            'acc': acc_sig,
            'temp': temp_sig,
        }
        all_private_embs, all_shared_embs = {}, {}
        
        # 1. Get tokens from each selected MAE
        for name in self.modalities:
            sig = input_map[name]
            model = self.models[name]
            # In finetuning, we process the whole signal, but we still need to generate the indices
            # for the forward_encoder method, so we use a mask_ratio of 0.0.
            ids_shuffle, ids_restore, ids_keep = model.propose_masking(
                batch_size=sig.shape[0], 
                num_patches=model.num_patches, 
                mask_ratio=0.0, 
                devic=sig.device
            )
            private_tokens, shared_tokens, _ = model.forward_encoder(sig, 0.0, ids_shuffle, ids_restore, ids_keep)
            all_private_embs[name] = private_tokens
            all_shared_embs[name] = shared_tokens

        # 2. Pooling
        priv_pool = {name: tokens.mean(dim=1) for name, tokens in all_private_embs.items()}
        sh_pool = {name: tokens.mean(dim=1) for name, tokens in all_shared_embs.items()}
        
        # 3. Fusion 
        if self.fuse_embeddings and self.only_shared:
            # first stack the shared embeddings for selected modalities, then take the mean
            fusion_vector = torch.stack(
                [sh_pool[name] for name in self.modalities], dim=1
            ).mean(dim=1)
        elif self.fuse_embeddings and not self.only_shared:
            fusion_shared = torch.stack(
                [sh_pool[name] for name in self.modalities], dim=1
            ).mean(dim=1)
            fusion_private = torch.stack(
                [priv_pool[name] for name in self.modalities], dim=1
            ).mean(dim=1)
            fusion_vector = torch.cat([fusion_shared, fusion_private], dim=-1)
        elif not self.fuse_embeddings and self.only_shared:
            fusion_vector = torch.cat([sh_pool[name] for name in self.modalities], dim=-1)
        else:
            fusion_vector = torch.cat(
                [torch.cat([priv_pool[name], sh_pool[name]], dim=-1) for name in self.modalities],
                dim=-1
            )
        
        # 4. Classification
        logits = self.classifier(fusion_vector)
        return logits

def setup_logging(run_name, output_path):
    log_dir = os.path.join(output_path, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "finetune.log")
    
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

def load_pretrained_models(path, device, model_args, modalities=None):
    """Loads the pretrained MAE models from a checkpoint file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    
    selected = modalities if modalities is not None else ['ecg', 'bvp', 'acc', 'temp']
    models = {}
    for name in selected:
        model = CheapSensorMAE(modality_name=name, **model_args).to(device)
        model_state_dict = checkpoint[f'{name}_model_state_dict']
        model.load_state_dict(model_state_dict)
        models[name] = model
        
    logging.info(f"Loaded pretrained models from {path} for modalities={selected}")
    return models

def validate(model, dataloader, criterion, device):
    """Runs a validation loop and returns metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            ecg, bvp, acc, temp = [batch[k].to(device) for k in ('ecg', 'bvp', 'acc', 'temp')]
            labels = batch['label'].to(device)
            
            logits = model(ecg, bvp, acc, temp).squeeze()
            loss = criterion(logits, labels.float())
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, accuracy


def train(args, pretrained_run_name=None):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, args.run_name)
    os.makedirs(run_output_path, exist_ok=True)

    # Determine modalities list
    modalities = ['ecg', 'bvp', 'acc', 'temp'] if args.modality == 'all' else [args.modality]
    
    # --- Log arguments ---
    logging.info("--- Command Line Arguments ---")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"{arg}: {value}")
    logging.info("------------------------------")
    logging.info(f"Using modalities: {modalities}")
    
    # --- Load Pretrained Models ---
    model_args = {
        'sig_len': args.signal_length, 'window_len': args.patch_window_len, 
        'private_mask_ratio': args.private_mask_ratio, 'embed_dim': args.embed_dim, 
        'depth': args.depth, 'num_heads': args.num_heads,
        'decoder_embed_dim': args.decoder_embed_dim, 'decoder_depth': args.decoder_depth, 
        'decoder_num_heads': args.decoder_num_heads, 'mlp_ratio': args.mlp_ratio, 
        'decoder_mlp_ratio': args.decoder_mlp_ratio
    }
    
    if args.from_scratch:
        logging.info("--- Training from Scratch ---")
        base_models = {
            name: CheapSensorMAE(modality_name=name, **model_args).to(device)
            for name in modalities
        }
    else:
        logging.info("--- Loading Pretrained Models for Fine-tuning ---")
        load_run_name = pretrained_run_name if pretrained_run_name else args.run_name
        args.pretrained_ckpt_path = f"/fd24T/zzhao3/EDA/results/cheap_maes/{load_run_name}/models/best_ckpt.pt"
        base_models = load_pretrained_models(args.pretrained_ckpt_path, device, model_args, modalities)

    # --- Create Classifier ---
    model = StressClassifier(
        base_models,
        args.embed_dim,
        args.freeze_backbone,
        args.linear_classifier,
        args.only_shared,
        args.fuse_embeddings,
        modalities=modalities,
    ).to(device)
    
    # --- Optimizer ---
    if args.freeze_backbone:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.head_lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.models.parameters(), 'lr': args.backbone_lr},
            {'params': model.classifier.parameters(), 'lr': args.head_lr}
        ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_restart_epochs, T_mult=args.t_mult)

    # --- Data ---
    # Load the full training dataset for the fold
    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train')

    # Ensure the validation subject is not the same as the test subject
    if args.val_subject_id == args.fold_number:
        raise ValueError("Validation subject ID cannot be the same as the test subject ID (fold_number).")

    # Split dataset into training and validation based on subject ID
    val_indices = [i for i, sid in enumerate(full_train_dataset.S) if sid == args.val_subject_id]
    train_indices = [i for i, sid in enumerate(full_train_dataset.S) if sid != args.val_subject_id]
    
    # Apply downsampling to the training set if a rate is specified
    if args.finetune_sample_rate > 1:
        train_indices = train_indices[::args.finetune_sample_rate]
        logging.info(f"Downsampling training data with rate 1/{args.finetune_sample_rate}.")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    # Print label distribution
    logging.info(f"Full training data has {len(full_train_dataset)} windows.")
    unique_labels, counts = np.unique(full_train_dataset.labels, return_counts=True)
    logging.info(f"Full training dataset label distribution: {dict(zip(unique_labels, counts))}")
    
    logging.info(f"Training set size: {len(train_dataset)} (subjects other than {args.val_subject_id})")
    logging.info(f"Validation set size: {len(val_dataset)} (subject {args.val_subject_id})\n")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    metrics_history = []

    for epoch in range(args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        # --- Training Loop ---
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in pbar_train:
            ecg, bvp, acc, temp = [batch[k].to(device) for k in ('ecg', 'bvp', 'acc', 'temp')]
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(ecg, bvp, acc, temp).squeeze()
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            pbar_train.set_postfix(Train_Acc=batch_acc, Train_Loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        
        # --- Validation Loop ---
        val_loss, val_f1, val_acc = validate(model, val_dataloader, criterion, device)
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })

        scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(run_output_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved new best model to {best_model_path} with F1 score: {val_f1:.4f}")

    logging.info("Fine-tuning complete.")
    
    # --- Save Metrics ---
    metrics_array = np.array([list(d.values()) for d in metrics_history])
    metrics_path = os.path.join(run_output_path, "training_metrics.npy")
    header = ",".join(metrics_history[0].keys())
    np.save(metrics_path, metrics_array)
    
    header_path = os.path.join(run_output_path, "metrics_header.txt")
    with open(header_path, 'w') as f:
        f.write(header)
    logging.info(f"Saved training metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Finetune Cheap Sensor MAEs for Stress Prediction')
    # Paths
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S_finetune"))
    parser.add_argument('--tag', type=str, default="", help='Optional tag to append to the run name.')
    parser.add_argument('--data_path', type=str, default="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/fd24T/zzhao3/EDA/results/finetuned_models", help='Directory to save logs and models')    
    # Data and training settings
    parser.add_argument('--fold_number', type=int, default=17, help='The fold number to use for training/validation (this is the test subject).')
    parser.add_argument('--val_subject_id', type=int, default=16, help='The subject ID to use for the validation set.')
    parser.add_argument('--finetune_sample_rate', type=int, default=1, help='Rate for downsampling the finetuning training data (1 in r samples).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:15' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help="Learning rate for the MAE backbones")
    parser.add_argument('--head_lr', type=float, default=1e-3, help="Learning rate for the classification head")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze the MAE backbones and only train the classifier head")
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch without loading pretrained weights.')
    parser.add_argument('--lr_restart_epochs', type=int, default=10, help='Number of epochs to restart the learning rate')
    parser.add_argument('--t_mult', type=int, default=2, help='Multiplier for the learning rate scheduler')
    parser.add_argument('--linear_classifier', action='store_true', help='Use a linear classifier instead of a MLP')
    parser.add_argument('--only_shared', action='store_true', help='Use only the shared embeddings for classification')
    parser.add_argument('--fuse_embeddings', action='store_true', help='Fuse the embeddings of the private and shared modalities')
    parser.add_argument('--modality', type=str, default='all', choices=['all', 'ecg', 'bvp', 'acc', 'temp'], help='Select a single modality or all')
    # Model specific arguments (should match pre-training)
    parser.add_argument('--signal_length', type=int, default=3840, help='Length of the input signal windows.')
    parser.add_argument('--patch_window_len', type=int, default=96, help='Length of the patch window for the MAE.')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension of the encoder.')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the encoder.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in the encoder.')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='Embedding dimension of the decoder.')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Depth of the decoder.')
    parser.add_argument('--decoder_num_heads', type=int, default=16, help='Number of heads in the decoder.')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio for the encoder.')
    parser.add_argument('--decoder_mlp_ratio', type=float, default=4.0, help='MLP ratio for the decoder.')
    parser.add_argument('--private_mask_ratio', type=float, default=0.5, help='Ratio of private to shared embeddings')

    args = parser.parse_args()
    
    # Store original run name for loading pretrained model and append tag if provided
    original_run_name = args.run_name
    if args.tag:
        args.run_name = f"{args.run_name}_{args.tag}"

    setup_logging(args.run_name, args.output_path)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    train(args, original_run_name)

if __name__ == '__main__':
    main()
