# Copyright (c) 2026 PULSE contributors
# SPDX-License-Identifier: MIT

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.secure_io import load_npz_archive

# ── 1.  Path to the .npz fold you want to inspect ─────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
fold_path = os.path.join(REPO_ROOT, "preprocessed_data", "60s_0.25s_sid", "fold_2.npz")  # adapt as needed

# ── 2.  Load the archive ──────────────────────────────────────────────────
with load_npz_archive(fold_path) as data:
    # list all arrays stored in the file
    print("Arrays in archive:", list(data.keys()))
    
    # training split
    X_train = data["X_train"]          # shape = [N_train, 512, C_in]
    Y_train = data["Y_train"]          # shape = [N_train, 512]  (EDA)
    L_train = data["L_train"]          # shape = [N_train]
    
    # test split (held-out subject)
    X_test  = data["X_test"]
    Y_test  = data["Y_test"]
    L_test  = data["L_test"]
    
    # optional: per-subject baseline stats and feature names
    stats = None
    if "test_subject_mean" in data and "test_subject_std" in data:
        stats = {
            "mean": data["test_subject_mean"],
            "std": data["test_subject_std"],
        }
    feat = np.asarray(data["feature_names"]).astype(str, copy=False)

# ── 3.  Report counts ─────────────────────────────────────────────────────
print(f"\nTRAIN windows  : {X_train.shape[0]:,}")
print(f"TEST  windows  : {X_test.shape[0]:,}")
print(f"Window length  : {X_train.shape[1]} samples  "
      f"({X_train.shape[1]/64:.1f} s at 64 Hz)")
print(f"Input channels : {X_train.shape[2]}")
print(f"Feature names  : {feat}")

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"L_train shape: {L_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
print(f"L_test shape: {L_test.shape}")

# quick sanity-check: Y arrays must have same first dimension as X
assert Y_train.shape[0] == X_train.shape[0]
assert Y_test.shape[0]  == X_test.shape[0]
