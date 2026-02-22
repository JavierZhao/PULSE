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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.backbone_registry import BACKBONE_REGISTRY, get_backbone_class
from src.data.wesad_dataset import WESADDataset

class StressClassifier(nn.Module):
    def __init__(self, models, embed_dim, freeze_backbone=False, linear_classifier=False, only_shared=False, fuse_embeddings=False, modalities=None, num_classes=1, adaptive_fusion=False):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.modalities = modalities if modalities is not None else ['ecg', 'bvp', 'acc', 'temp']
        self.num_classes = num_classes
        self.adaptive_fusion = adaptive_fusion
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
        # Optional learnable fusion weights when averaging embeddings
        if self.adaptive_fusion and self.fuse_embeddings:
            if self.only_shared:
                self.fusion_logits_shared = nn.Parameter(torch.zeros(num_modalities))
            else:
                self.fusion_logits_shared = nn.Parameter(torch.zeros(num_modalities))
                self.fusion_logits_private = nn.Parameter(torch.zeros(num_modalities))
        if linear_classifier:
            self.classifier = nn.Sequential(nn.Linear(self.fusion_dim, self.num_classes))
        else:
            self.classifier = nn.Sequential( 
                nn.Linear(self.fusion_dim, 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(4, self.num_classes)
            )

    def forward(self, ecg_sig, bvp_sig, acc_sig, temp_sig, eda_sig=None):
        # Map input tensors by modality name
        input_map = {
            'ecg': ecg_sig,
            'bvp': bvp_sig,
            'acc': acc_sig,
            'temp': temp_sig,
            'eda': eda_sig,
        }
        if 'eda' in self.modalities and eda_sig is None:
            raise ValueError("EDA modality selected but no eda signal provided to forward().")
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
            if self.adaptive_fusion:
                stacked = torch.stack([sh_pool[name] for name in self.modalities], dim=1)
                weights = torch.softmax(self.fusion_logits_shared, dim=0).reshape(1, -1, 1)
                fusion_vector = (stacked * weights).sum(dim=1)
            else:
                fusion_vector = torch.stack(
                    [sh_pool[name] for name in self.modalities], dim=1
                ).mean(dim=1)
        elif self.fuse_embeddings and not self.only_shared:
            if self.adaptive_fusion:
                stacked_sh = torch.stack([sh_pool[name] for name in self.modalities], dim=1)
                w_sh = torch.softmax(self.fusion_logits_shared, dim=0).reshape(1, -1, 1)
                fusion_shared = (stacked_sh * w_sh).sum(dim=1)
                stacked_pr = torch.stack([priv_pool[name] for name in self.modalities], dim=1)
                w_pr = torch.softmax(self.fusion_logits_private, dim=0).reshape(1, -1, 1)
                fusion_private = (stacked_pr * w_pr).sum(dim=1)
            else:
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


def load_pretrained_models(path, device, model_args, modalities=None, backbone='transformer'):
    """Loads the pretrained models from a checkpoint file."""
    backbone_cls = get_backbone_class(backbone)

    checkpoint = torch.load(path, map_location=device)

    selected = modalities if modalities is not None else ['ecg', 'bvp', 'acc', 'temp']
    models = {}
    for name in selected:
        model = backbone_cls(modality_name=name, **model_args).to(device)
        model_state_dict = checkpoint[f'{name}_model_state_dict'] if name != 'eda' else checkpoint[f'model_state_dict']
        model.load_state_dict(model_state_dict)
        models[name] = model

    logging.info(f"Loaded pretrained models ({backbone}) from {path} for modalities={selected}")
    return models

def validate(model, dataloader, criterion, device, three_class=False):
    """Runs a validation loop and returns metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validating")
        for batch in pbar_val:
            ecg = batch['ecg'].to(device)
            bvp = batch['bvp'].to(device)
            acc = batch['acc'].to(device)
            temp = batch['temp'].to(device)
            eda = batch['eda'].to(device) if 'eda' in model.modalities else None
            labels = batch['label'].to(device)
            
            logits = model(ecg, bvp, acc, temp, eda)
            if three_class:
                loss = criterion(logits, labels.long())
            else:
                logits = logits.squeeze()
                loss = criterion(logits, labels.float())
            total_loss += loss.item()

            if three_class:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    if three_class:
        f1 = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        try:
            y_score_arr = np.asarray(all_probs)
            auroc = roc_auc_score(all_labels, y_score_arr, multi_class='ovr', average='macro')
        except Exception:
            auroc = float('nan')
        # Macro one-vs-rest AUPRC across classes
        try:
            labels_np = np.asarray(all_labels)
            y_score_arr = np.asarray(all_probs)
            num_classes = y_score_arr.shape[1] if y_score_arr.ndim == 2 else 0
            aps = []
            for c in range(num_classes):
                y_true_bin = (labels_np == c).astype(int)
                try:
                    aps.append(average_precision_score(y_true_bin, y_score_arr[:, c]))
                except Exception:
                    aps.append(np.nan)
            ap = float(np.nanmean(aps)) if len(aps) > 0 else float('nan')
        except Exception:
            ap = float('nan')
    else:
        f1 = f1_score(all_labels, all_preds, average='binary')
        accuracy = accuracy_score(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auroc = float('nan')
        try:
            ap = average_precision_score(all_labels, all_probs)
        except Exception:
            ap = float('nan')

    # Tune threshold on validation to maximize F1
    tuned_thr = 0.5
    f1_at_tuned = float('nan')
    precision_at_tuned = float('nan')
    recall_at_tuned = float('nan')
    if not three_class:
        try:
            labels_np = np.asarray(all_labels)
            probs_np = np.asarray(all_probs)
            # Guard degenerate cases: constant labels or scores
            if len(np.unique(labels_np)) < 2 or len(np.unique(probs_np)) == 1:
                preds_t = (probs_np >= tuned_thr).astype(int)
                f1_at_tuned = float(f1_score(labels_np, preds_t, average='binary'))
                from sklearn.metrics import precision_score, recall_score
                precision_at_tuned = float(precision_score(labels_np, preds_t, zero_division=0))
                recall_at_tuned = float(recall_score(labels_np, preds_t, zero_division=0))
            else:
                precision, recall, thresholds = precision_recall_curve(labels_np, probs_np)
                if thresholds is not None and len(thresholds) > 0:
                    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
                    best_idx = int(np.nanargmax(f1s))
                    tuned_thr = float(thresholds[best_idx])
                    # Recompute F1/Precision/Recall at tuned threshold using predictions
                    preds_t = (probs_np >= tuned_thr).astype(int)
                    from sklearn.metrics import precision_score, recall_score
                    f1_at_tuned = float(f1_score(labels_np, preds_t, average='binary'))
                    precision_at_tuned = float(precision_score(labels_np, preds_t, zero_division=0))
                    recall_at_tuned = float(recall_score(labels_np, preds_t, zero_division=0))
                else:
                    # Fallback sweep over actual operating points
                    candidates = np.unique(probs_np)
                    best_f1 = -1.0
                    best_t = 0.5
                    best_prec = 0.0
                    best_rec = 0.0
                    from sklearn.metrics import precision_score, recall_score
                    for t in candidates:
                        preds_t = (probs_np >= t).astype(int)
                        f1_t = f1_score(labels_np, preds_t, average='binary')
                        if f1_t > best_f1:
                            best_f1 = f1_t
                            best_t = float(t)
                            best_prec = float(precision_score(labels_np, preds_t, zero_division=0))
                            best_rec = float(recall_score(labels_np, preds_t, zero_division=0))
                    tuned_thr = float(best_t)
                    f1_at_tuned = float(best_f1)
                    precision_at_tuned = best_prec
                    recall_at_tuned = best_rec
        except Exception:
            pass

    return avg_loss, f1, accuracy, auroc, ap, tuned_thr, f1_at_tuned, precision_at_tuned, recall_at_tuned


def train(args, pretrained_run_name=None):
    device = torch.device(args.device)
    run_output_path = os.path.join(args.output_path, f"{args.save_name}/f_{args.fold_number}")
    os.makedirs(run_output_path, exist_ok=True)

    # Determine modalities list
    if getattr(args, 'modalities', 'all') == 'all':
        modalities = ['ecg', 'bvp', 'acc', 'temp']
        if getattr(args, 'include_eda', False):
            modalities.append('eda')
    else:
        parsed = [m.strip().lower() for m in args.modalities.split(',') if m.strip()]
        allowed = {'ecg', 'bvp', 'acc', 'temp', 'eda'}
        invalid = [m for m in parsed if m not in allowed]
        if len(invalid) > 0:
            raise ValueError(f"Invalid modalities specified: {invalid}. Allowed: {sorted(list(allowed))}")
        modalities = parsed
        if getattr(args, 'include_eda', False) and 'eda' not in modalities:
            modalities.append('eda')
    
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
    
    backbone_cls = get_backbone_class(args.backbone)
    logging.info(f"Using backbone: {args.backbone} ({backbone_cls.__name__})")

    if args.from_scratch:
        logging.info("--- Training from Scratch ---")
        base_models = {
            name: backbone_cls(modality_name=name, **model_args).to(device)
            for name in modalities
        }
    else:
        logging.info("--- Loading Pretrained Models for Fine-tuning ---")
        # Interpret run_name as either a bare name (legacy) or an absolute/relative path to a models directory
        # If it points to a directory, look for best_ckpt.pt inside; if it points to a file, use it directly
        candidate = pretrained_run_name if pretrained_run_name else args.run_name
        print("path candidate", candidate)
        ckpt_path = candidate
        if os.path.isfile(candidate):
            ckpt_path = candidate
        elif os.path.isdir(candidate):
            file_candidate = os.path.join(candidate, 'best_ckpt.pt')
            if os.path.isfile(file_candidate):
                ckpt_path = file_candidate
        else:
            # Legacy behavior: assume a run name under results/cheap_maes/<name>/models/best_ckpt.pt
            legacy = os.path.join('/j-jepa-vol/PULSE/results/cheap_maes', candidate, 'models', 'best_ckpt.pt')
            if os.path.isfile(legacy):
                ckpt_path = legacy
        if ckpt_path is None:
            raise FileNotFoundError(f"Could not resolve pretrained checkpoint from run_name='{args.run_name}'. Provide a full path to a file or models dir, or ensure legacy path exists.")
        args.pretrained_ckpt_path = ckpt_path
        base_models = load_pretrained_models(ckpt_path, device, model_args, modalities, backbone=args.backbone)

    # --- Create Classifier ---
    model = StressClassifier(
        base_models,
        args.embed_dim,
        args.freeze_backbone,
        args.linear_classifier,
        args.only_shared,
        args.fuse_embeddings,
        modalities=modalities,
        num_classes=(3 if getattr(args, 'three_class', False) else 1),
        adaptive_fusion=getattr(args, 'adaptive_fusion', False),
    ).to(device)
    
    # --- Optimizer ---
    if args.freeze_backbone:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.head_lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.models.parameters(), 'lr': args.backbone_lr},
            {'params': [p for n, p in model.named_parameters() if n.startswith('classifier') or n.startswith('fusion_logits')], 'lr': args.head_lr}
        ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_restart_epochs, T_mult=args.t_mult)

    # --- Data ---
    # Load the full training dataset for the fold
    full_train_dataset = WESADDataset(data_path=args.data_path, fold_number=args.fold_number, split='train', three_class=getattr(args, 'three_class', False))

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
    
    criterion = nn.CrossEntropyLoss() if getattr(args, 'three_class', False) else nn.BCEWithLogitsLoss()
    
    best_val_ap = float('-inf')
    best_val_acc = float('-inf')
    best_val_f1_tuned = float('-inf')
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
            ecg = batch['ecg'].to(device)
            bvp = batch['bvp'].to(device)
            acc = batch['acc'].to(device)
            temp = batch['temp'].to(device)
            eda = batch['eda'].to(device) if 'eda' in model.modalities else None
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(ecg, bvp, acc, temp, eda)
            if getattr(args, 'three_class', False):
                loss = criterion(logits, labels.long())
            else:
                logits = logits.squeeze()
                loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if getattr(args, 'three_class', False):
                preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).long()
            else:
                preds = (torch.sigmoid(logits) > 0.5).long()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            pbar_train.set_postfix(Train_Acc=batch_acc, Train_Loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average=('macro' if getattr(args, 'three_class', False) else 'binary'))
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        
        # --- Validation Loop ---
        val_loss, val_f1, val_acc, val_auroc, val_ap, val_tuned_thr, val_f1_tuned, val_prec_tuned, val_rec_tuned = validate(model, val_dataloader, criterion, device, three_class=getattr(args, 'three_class', False))
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_ap:.4f} | Val F1@thr: {val_f1_tuned:.4f} (thr={val_tuned_thr:.3f}, P={val_prec_tuned:.3f}, R={val_rec_tuned:.3f})")
        
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auroc': val_auroc,
            'val_auprc': val_ap,
            'val_f1_tuned': val_f1_tuned,
            'val_tuned_thr': val_tuned_thr,
            'val_precision_tuned': val_prec_tuned,
            'val_recall_tuned': val_rec_tuned
        })

        scheduler.step()
        
        # Selection based on chosen validation metric (default: AUPRC). Tie-break with F1@tuned.
        selected_metric = getattr(args, 'val_selection_metric', 'auprc')
        current_val_metric = val_ap if selected_metric == 'auprc' else val_acc
        best_val_metric = best_val_ap if selected_metric == 'auprc' else best_val_acc

        improved = False
        if not np.isnan(current_val_metric):
            if current_val_metric > best_val_metric + 1e-12:
                improved = True
            elif abs(current_val_metric - best_val_metric) <= 1e-12 and val_f1_tuned > best_val_f1_tuned:
                improved = True
        if improved:
            if selected_metric == 'auprc':
                best_val_ap = val_ap
            else:
                best_val_acc = val_acc
            best_val_f1_tuned = val_f1_tuned
            best_model_path = os.path.join(run_output_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            if not getattr(args, 'three_class', False):
                # Persist tuned threshold for LOSO testing
                thr_path = os.path.join(run_output_path, "best_threshold.txt")
                with open(thr_path, 'w') as f:
                    f.write(f"{val_tuned_thr:.6f}\n")
                logging.info(
                    f"Saved new best model to {best_model_path} with {selected_metric.upper()}: {current_val_metric:.4f}, "
                    f"F1@thr: {val_f1_tuned:.4f} (thr={val_tuned_thr:.6f}). Threshold saved to {thr_path}"
                )
            else:
                logging.info(
                    f"Saved new best model to {best_model_path} with {selected_metric.upper()}: {current_val_metric:.4f}"
                )

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
    parser.add_argument('--save_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S_finetune"))
    parser.add_argument('--tag', type=str, default="", help='Optional tag to append to the run name.')
    parser.add_argument('--data_path', type=str, default="/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid", help='Path to the WESAD preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/j-jepa-vol/PULSE/results/finetuned_models", help='Directory to save logs and models')    
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
    parser.add_argument('--backbone', type=str, default='transformer', choices=sorted(BACKBONE_REGISTRY.keys()),
                        help='Encoder backbone architecture (must match pretrained checkpoint).')
    parser.add_argument('--lr_restart_epochs', type=int, default=10, help='Number of epochs to restart the learning rate')
    parser.add_argument('--t_mult', type=int, default=2, help='Multiplier for the learning rate scheduler')
    parser.add_argument('--linear_classifier', action='store_true', help='Use a linear classifier instead of a MLP')
    parser.add_argument('--only_shared', action='store_true', help='Use only the shared embeddings for classification')
    parser.add_argument('--fuse_embeddings', action='store_true', help='Fuse the embeddings of the private and shared modalities')
    parser.add_argument('--modalities', type=str, default='all', help='Comma-separated list of modalities from {ecg,bvp,acc,temp}, or "all"')
    parser.add_argument('--include_eda', action='store_true', help='Include EDA as an additional modality when using all modalities')
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
    parser.add_argument('--val_selection_metric', type=str, default='auprc', choices=['accuracy', 'auprc'], help='Metric to select best model on validation set')
    parser.add_argument('--three_class', action='store_true', help='Use three-class classification (baseline, stress, amusement)')
    parser.add_argument('--adaptive_fusion', action='store_true', help='Use learnable per-modality fusion weights when fusing embeddings')

    args = parser.parse_args()
    
    # Store original run name for loading pretrained model and append tag if provided
    original_run_name = args.run_name
    # if args.tag:
    #     args.run_name = f"{args.run_name}_{args.tag}"

    setup_logging(f"{args.save_name}/f_{args.fold_number}", args.output_path)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    train(args, original_run_name)

if __name__ == '__main__':
    main()
