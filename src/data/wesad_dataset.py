import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

class WESADDataset(Dataset):
    """Loads all required modalities from a single preprocessed WESAD .npz fold."""
    def __init__(self, data_path, fold_number, split='train'):
        super().__init__()
        file_path = os.path.join(data_path, f"fold_{fold_number}.npz")
        logging.info(f"Loading {split} data from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with np.load(file_path, allow_pickle=True) as data:
            if split == 'train':
                self.X = data['X_train']
                self.Y = data['Y_train']
            elif split == 'test':
                self.X = data['X_test']
                self.Y = data['Y_test']
            else:
                raise ValueError("split must be 'train' or 'test'")

            self.feature_names = data['feature_names']
            
        # Get the indices for each required modality
        self.ecg_idx = np.where(self.feature_names == 'ecg')[0][0]
        self.bvp_idx = np.where(self.feature_names == 'bvp')[0][0]
        self.acc_idx = np.where(self.feature_names == 'net_acc_wrist')[0][0]
        self.temp_idx = np.where(self.feature_names == 'temp')[0][0]

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

        return {
            'ecg': torch.from_numpy(ecg).float(),
            'bvp': torch.from_numpy(bvp).float(),
            'acc': torch.from_numpy(acc).float(),
            'temp': torch.from_numpy(temp).float(),
            'eda': torch.from_numpy(eda).float()
        } 