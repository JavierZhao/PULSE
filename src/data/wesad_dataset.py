import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

class WESADDataset(Dataset):
    """Loads all required modalities, labels, and subject IDs from a preprocessed WESAD .npz fold."""
    def __init__(self, data_path, fold_number, split='train', three_class=False):
        super().__init__()
        file_path = os.path.join(data_path, f"fold_{fold_number}.npz")
        logging.info(f"Loading {split} data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with np.load(file_path, allow_pickle=True) as data:
            if split == 'train':
                self.X = data['X_train']
                self.Y = data['Y_train']
                self.L = data['L_train']
                self.S = data.get('S_train', np.array([-1] * len(self.X))) # Use get for backward compatibility
            elif split == 'test':
                self.X = data['X_test']
                self.Y = data['Y_test']
                self.L = data['L_test']
                self.S = data.get('S_test', np.array([-1] * len(self.X))) # Use get for backward compatibility
            else:
                raise ValueError("split must be 'train' or 'test'")

            self.feature_names = data['feature_names']
            
        # Get the indices for each required modality
        self.ecg_idx = np.where(self.feature_names == 'ecg')[0][0]
        self.bvp_idx = np.where(self.feature_names == 'bvp')[0][0]
        self.acc_idx = np.where(self.feature_names == 'net_acc_wrist')[0][0]
        self.temp_idx = np.where(self.feature_names == 'temp')[0][0]

        # Convert WESAD labels to binary (stress vs. non-stress) or three-class
        # WESAD labels: 1=baseline, 2=stress, 3=amusement.
        if three_class:
            self.labels = (self.L.astype(int) - 1)
        else:
            self.labels = (self.L == 2).astype(int)

        logging.info(f"Loaded {self.X.shape[0]} windows for {split} split from fold {fold_number}.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Extract each signal, adding a channel dimension
        ecg = self.X[idx, :, self.ecg_idx][np.newaxis, :]
        bvp = self.X[idx, :, self.bvp_idx][np.newaxis, :]
        acc = self.X[idx, :, self.acc_idx][np.newaxis, :]
        temp = self.X[idx, :, self.temp_idx][np.newaxis, :]
        eda = self.Y[idx][np.newaxis, :]
        label = self.labels[idx]
        subject_id = self.S[idx]

        return {
            'ecg': torch.from_numpy(ecg).float(),
            'bvp': torch.from_numpy(bvp).float(),
            'acc': torch.from_numpy(acc).float(),
            'temp': torch.from_numpy(temp).float(),
            'eda': torch.from_numpy(eda).float(),
            'label': torch.tensor(label).long(),
            'subject_id': torch.tensor(subject_id).long()
        }
