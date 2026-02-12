import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, resample_poly
import argparse


# --- Configuration ---
TARGET_FS = 64  # Hz
WINDOW_SECONDS = 60
STRIDE_SECONDS = 0.25

# Original sampling rates of the WESAD sensors
NATIVE_FS = {
    'acc_chest': 700, 'ecg': 700, 'emg': 700, 'eda_chest': 700, 'temp_chest': 700, 'resp': 700,
    'acc_wrist': 32, 'bvp': 64, 
    'eda': 4, 'temp': 4, 
    'label': 700
}

# Paths
DATA_DIR = '/j-jepa-vol/PULSE/data'
WESAD_PKL_DIR = '/j-jepa-vol/PULSE/data/WESAD'
OUTPUT_DIR = '/j-jepa-vol/PULSE/preprocessed_data/60s_0.25s_sid'
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


# --- Helper Class for Data Loading ---

class SubjectData:
    """A class to load signal data from CSVs and labels from the original .pkl files."""
    def __init__(self, csv_base_path, pkl_base_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_csv_path = os.path.join(csv_base_path, f'{self.name}_raw_data')
        self.subject_pkl_path = os.path.join(pkl_base_path, self.name, f'{self.name}.pkl')
        
        if not os.path.isdir(self.subject_csv_path):
            raise FileNotFoundError(f"CSV data directory not found for subject {self.name} at {self.subject_csv_path}")
        if not os.path.exists(self.subject_pkl_path):
            raise FileNotFoundError(f"Label .pkl file not found for subject {self.name} at {self.subject_pkl_path}")

    def get_all_signals(self):
        """Returns a dictionary of all raw signals by loading from CSVs and .pkl for labels."""
        print(f"  - Loading CSV data from {self.subject_csv_path}...")
        signals = {}
        csv_files_to_load = {
            'acc_chest.csv': 'acc_chest', 'acc_wrist.csv': 'acc_wrist',
            'bvp.csv': 'bvp', 'ecg.csv': 'ecg', 'eda.csv': 'eda', 'temp.csv': 'temp'
        }

        for fname, signal_key in csv_files_to_load.items():
            file_path = os.path.join(self.subject_csv_path, fname)
            if os.path.exists(file_path):
                signals[signal_key] = pd.read_csv(file_path, index_col=0).values
            else:
                print(f"    - Warning: {fname} not found for subject {self.name}.")
        
        # Load the ground-truth labels from the original pkl file.
        with open(self.subject_pkl_path, 'rb') as file:
            original_data = pickle.load(file, encoding='latin1')
        signals['label'] = original_data['label']

        return signals

# --- Processing Functions (to be implemented) ---

def resample_signals(signals_dict, subject_id):
    """
    Step 1: Resample all signals to a common target frequency.
    The `label` signal is downsampled first to serve as the reference length.
    """
    # --- 1. Downsample the label signal to create the reference timeline ---
    label_sig = signals_dict['label'].ravel()
    original_label_fs = NATIVE_FS['label']
    
    # Calculate new length based on the duration of the label signal
    duration_secs = len(label_sig) / original_label_fs
    new_len = int(duration_secs * TARGET_FS)

    # Create the new time indices for the target frequency
    resampled_time_idx = np.linspace(0., duration_secs, new_len, endpoint=False)
    
    # Create the original time indices for the label signal
    original_time_idx = np.linspace(0., duration_secs, len(label_sig), endpoint=False)

    # Find the nearest original label for each new timestamp
    nearest_indices = np.searchsorted(original_time_idx, resampled_time_idx, side='right') - 1
    nearest_indices[nearest_indices < 0] = 0 # Ensure indices are within bounds
    resampled_labels = label_sig[nearest_indices]

    print(f"  - Resampling signals for S{subject_id} to {new_len} samples ({TARGET_FS} Hz)...")

    # --- 2. Resample all other signals to match the new length ---
    resampled_df_data = {'label': resampled_labels}

    for signal_name, data in signals_dict.items():
        if signal_name == 'label':
            continue

        original_fs = NATIVE_FS[signal_name]
        
        if original_fs is None:
            print(f"    - Warning: No sample rate found for {signal_name}. Skipping.")
            continue

        # Correctly distinguish multi-axis signals from single-column ones
        if data.ndim > 1 and data.shape[1] > 1:
            resampled_cols = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                if original_fs <= 4:
                    original_col_time_idx = np.linspace(0., len(col_data) / original_fs, len(col_data), endpoint=False)
                    resampled_col = np.interp(resampled_time_idx, original_col_time_idx, col_data)
                else:
                    resampled_col = resample_poly(col_data, TARGET_FS, original_fs)[:new_len]
                resampled_cols.append(resampled_col)
            
            resampled_df_data[f'{signal_name}_x'] = resampled_cols[0]
            resampled_df_data[f'{signal_name}_y'] = resampled_cols[1]
            resampled_df_data[f'{signal_name}_z'] = resampled_cols[2]
            
        else:
            # Flatten array to ensure it's 1-D for resampling
            data_1d = data.ravel()
            original_sig_time_idx = np.linspace(0., len(data_1d) / original_fs, len(data_1d), endpoint=False)
            if original_fs <= 4:
                resampled_sig = np.interp(resampled_time_idx, original_sig_time_idx, data_1d)
            else:
                resampled_sig = resample_poly(data_1d, TARGET_FS, original_fs)[:new_len]
            
            resampled_df_data[signal_name] = resampled_sig
            
    return pd.DataFrame(resampled_df_data)

def get_filtered_signal(data, f_low, f_high, fs=TARGET_FS, order=4):
    """Applies a bandpass butterworth filter."""
    nyquist = 0.5 * fs
    
    # Robustly handle cutoff frequencies
    if f_high >= nyquist:
        f_high = nyquist - 0.5  # Cap high cutoff at Nyquist limit
        print(f"  - Warning: High cutoff frequency was >= Nyquist. Capping at {f_high} Hz.")
    if f_low <= 0:
        raise ValueError("Low cutoff frequency must be greater than 0.")

    b, a = butter(order, [f_low / nyquist, f_high / nyquist], btype='band')
    return filtfilt(b, a, data)

def filter_signals(signals_df):
    """Step 2: Apply band-pass filters to clean the signals."""
    filtered_df = signals_df.copy()

    if 'eda' in filtered_df:
        nyquist = 0.5 * TARGET_FS
        # --- tonic removal (first-order high-pass) ---
        eda_hp = get_filtered_signal(filtered_df['eda'].values.ravel(),
                                      f_low=0.05, f_high=nyquist-0.5,
                                      order=1)
        # --- keep phasic energy up to 1 Hz ---
        filtered_df['eda'] = get_filtered_signal(eda_hp, 0.05, 1.0)
    
    if 'bvp' in filtered_df: filtered_df['bvp'] = get_filtered_signal(filtered_df['bvp'].values.ravel(), 0.5, 2.0)
    if 'ecg' in filtered_df: filtered_df['ecg'] = get_filtered_signal(filtered_df['ecg'].values.ravel(), 0.5, 40.0)

    acc_wrist_cols = ['acc_wrist_x', 'acc_wrist_y', 'acc_wrist_z']
    if all(c in filtered_df for c in acc_wrist_cols):
        filtered_df['net_acc_wrist'] = np.linalg.norm(filtered_df[acc_wrist_cols].values, axis=1)

    acc_chest_cols = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z']
    if all(c in filtered_df for c in acc_chest_cols):
        filtered_df['net_acc_chest'] = np.linalg.norm(filtered_df[acc_chest_cols].values, axis=1)

    final_cols = ['label', 'ecg', 'bvp', 'eda', 'temp', 'net_acc_wrist', 'net_acc_chest']
    return filtered_df[[c for c in final_cols if c in filtered_df]]

def create_windows(signals_df):
    """Steps 3 & 4: Create overlapping windows and assign a label to each."""
    
    # Pre-emptively drop all transient data points
    signals_df = signals_df[signals_df['label'] != 0].copy()
    
    window_len = int(WINDOW_SECONDS * TARGET_FS)
    stride_len = int(STRIDE_SECONDS * TARGET_FS)
    
    windows = []
    labels = []
    
    # WESAD labels: 1=baseline, 2=stress, 3=amusement
    valid_labels = {1, 2, 3}
    
    feature_cols = [col for col in signals_df.columns if col != 'label']
    indices_to_check = []
    if 'ecg' in feature_cols:
        indices_to_check.append(feature_cols.index('ecg'))
    if 'bvp' in feature_cols:
        indices_to_check.append(feature_cols.index('bvp'))

    data_matrix = signals_df[feature_cols].values
    label_array = signals_df['label'].values.ravel()
    
    for i in range(0, len(signals_df) - window_len + 1, stride_len):
        
        window_labels = label_array[i : i + window_len]
        
        # Check if the window contains only a single, valid class
        unique_labels = np.unique(window_labels)
        
        if len(unique_labels) == 1 and unique_labels[0] in valid_labels:
            
            # This is a clean, valid window
            window_data = data_matrix[i : i + window_len, :]

            # Discard window if ECG or BVP signal is flat
            if indices_to_check:
                flat_thresh = 0.02 # z-units std below which we call it flat
                win_std = window_data[:, indices_to_check].std(axis=0)
                if (win_std < flat_thresh).any():
                    continue # skip this window
            
            majority_label = unique_labels[0]
            
            windows.append(window_data)
            labels.append(majority_label)
            
    if not windows:
        print("  - Warning: No valid windows found for this subject. Check label alignment.")
        return np.array([]), np.array([])

    return np.array(windows), np.array(labels)

def normalize_data(windows, labels):
    """Step 5: Z-score normalization per subject based on baseline stats."""
    
    # Find the indices of the baseline windows (label == 1)
    baseline_indices = np.where(labels == 1)[0]
    
    if len(baseline_indices) == 0:
        print("  - Warning: No baseline windows found for this subject. Cannot calculate stats for normalization.")
        # Return dummy stats and un-normalized data
        num_channels = windows.shape[2]
        dummy_stats = {'mean': np.zeros(num_channels), 'std': np.ones(num_channels)}
        return windows, dummy_stats
        
    baseline_windows = windows[baseline_indices]
    
    # Calculate mean and std for each channel across all baseline windows
    # Reshape to (num_samples, num_channels) to calculate stats
    num_windows, window_len, num_channels = baseline_windows.shape
    baseline_data_flat = baseline_windows.reshape(-1, num_channels)
    
    mean = np.mean(baseline_data_flat, axis=0)
    std = np.std(baseline_data_flat, axis=0)
    
    # Avoid division by zero for channels with no variance
    std[std == 0] = 1.0
    
    # Apply z-score normalization to all windows
    normalized_windows = (windows - mean) / std
    
    subject_stats = {'mean': mean, 'std': std}
    
    print(f"  - Normalized {len(windows)} windows using stats from {len(baseline_windows)} baseline windows.")
    
    return normalized_windows, subject_stats


# --- Main Orchestration ---

def process_subject(subject_id):
    """Runs the full processing pipeline for a single subject."""
    print(f"Processing S{subject_id}...")
    try:
        subject_data = SubjectData(DATA_DIR, WESAD_PKL_DIR, subject_id)
        raw_signals = subject_data.get_all_signals()
        resampled_df = resample_signals(raw_signals, subject_id)
        filtered_df = filter_signals(resampled_df)
        windows, labels = create_windows(filtered_df)
        if windows.size == 0: return np.array([]), np.array([]), {}
        normalized_windows, stats = normalize_data(windows, labels)
        return normalized_windows, labels, stats
    except FileNotFoundError as e:
        print(f"  - Skipping S{subject_id}: {e}")
        return np.array([]), np.array([]), {}

def get_subject_ids_for_windows(cache_data, subject_ids_to_collect):
    """
    Collects an array of subject IDs corresponding to each window
    in a concatenated data array.
    """
    sids = []
    for sid in subject_ids_to_collect:
        if cache_data[sid]['X'].size > 0:
            num_windows = cache_data[sid]['X'].shape[0]
            sids.extend([sid] * num_windows)
    return np.array(sids)

def main():
    """Run LOSO preprocessing and save one .npz file per fold."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nCreating LOSO folds…")

    # Channel order after filter_signals:
    # ['label', 'ecg', 'bvp', 'eda', 'temp', 'net_acc_wrist', 'net_acc_chest']
    FEAT_COLS = [
        'ecg', 'bvp', 'eda', 'temp',
        'net_acc_wrist', 'net_acc_chest'
    ]
    EDA_IDX = FEAT_COLS.index('eda')        # fixed = 2

    # ------------------------------------------------------------------
    # 1. Pre-process every subject once and cache in memory
    # ------------------------------------------------------------------
    cache = {}
    for sid in SUBJECT_IDS:
        X, L, stats = process_subject(sid)   # returns 3 objects
        cache[sid] = dict(X=X, L=L, stats=stats, cols=FEAT_COLS)

    # ------------------------------------------------------------------
    # 2. Leave-one-subject-out folds
    # ------------------------------------------------------------------
    for test_sid in SUBJECT_IDS:
        print(f"\n── Fold with S{test_sid} as test ──")

        test_data = cache[test_sid]
        if test_data['X'].size == 0:
            print("  · Skipping: no test data.")
            continue

        train_sids = [sid for sid in SUBJECT_IDS if sid != test_sid]
        
        train_X, train_L = [], []
        for train_sid in train_sids:
            if cache[train_sid]['X'].size:
                train_X.append(cache[train_sid]['X'])
                train_L.append(cache[train_sid]['L'])

        if not train_X:
            print("  · Skipping: no training data.")
            continue

        X_train_all = np.concatenate(train_X, axis=0)
        L_train = np.concatenate(train_L, axis=0)
        X_test_all = test_data['X']
        L_test = test_data['L']
        test_stats = test_data['stats']
        
        feature_names = list(test_data['cols'])
        try:
            # Create subject ID arrays for train and test sets
            S_train = get_subject_ids_for_windows(cache, train_sids)
            S_test = np.array([test_sid] * len(L_test))
            
            eda_channel_index = feature_names.index('eda')
            Y_train = X_train_all[:, :, eda_channel_index]
            Y_test = X_test_all[:, :, eda_channel_index]
            X_train = np.delete(X_train_all, eda_channel_index, axis=2)
            X_test = np.delete(X_test_all, eda_channel_index, axis=2)
            feature_names.pop(eda_channel_index)
        except ValueError:
            print("  - Warning: 'eda' channel not found. Cannot create Y target.")
            Y_train, Y_test = np.array([]), np.array([])
            X_train, X_test = X_train_all, X_test_all
            S_train, S_test = np.array([]), np.array([]) # Also create empty subject arrays

        fold_path = os.path.join(OUTPUT_DIR, f'fold_{test_sid}.npz')
        np.savez_compressed(
            fold_path,
            X_train=X_train, Y_train=Y_train, L_train=L_train, S_train=S_train,
            X_test=X_test, Y_test=Y_test, L_test=L_test, S_test=S_test,
            test_subject_stats=test_stats,
            feature_names=feature_names
        )
        print(f"  · Saved {fold_path} "
              f"({X_train.shape[0]} train, {X_test.shape[0]} test windows)")

    print("\nPreprocessing complete.")




if __name__ == '__main__':
    main() 