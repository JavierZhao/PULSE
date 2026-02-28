# PhysioNet Wearable Device Dataset (Empatica E4) — Reproducible Preprocessing

This folder contains scripts to download and preprocess the PhysioNet “Wearable Device Dataset from Induced Stress and Structured Exercise Sessions” (v1.0.1) into a format compatible with our pipeline.

We focus on the **STRESS** protocol and produce **fixed-length 60s windows** with **0.25s stride** at a unified **64 Hz** sampling rate, using wrist modalities:
- **BVP (64 Hz)**
- **ACC (32 Hz → resampled to 64 Hz; we use acceleration magnitude)**
- **TEMP (4 Hz → resampled to 64 Hz)**
- **EDA (4 Hz → resampled to 64 Hz; kept as an optional “teacher” signal)**

Labels are **protocol-defined** (WESAD-style):
- `0` = non-stress (Baseline / Rest)
- `1` = stress (Tasks)
- `-1` = ignored segments (self-report “SL”, transitions, too-short segments)

---

## Repository Layout

### Code (tracked in git)
```
scripts/
  download_physionet_e4.sh         (optional; requires aws or wget)
  preprocess_one_subject.py        (raw -> aligned 64 Hz)
  label_and_window_subject.py      (aligned -> labeled + windows)
  make_subject_list.py             (build list of valid subjects)
  run_all_subjects.py              (batch: align + label + window)
  summarize_windows.py             (dataset summary CSV)
reproduce.py                       (recommended one-command runner)
requirements.txt                   (python deps)
README.md                    (this file)
```

### Data (NOT tracked in git)
```
data/
  raw/
    physionet_e4/                  (downloaded dataset contents)
      Wearable_Dataset/
        STRESS/
          S01/ BVP.csv ACC.csv EDA.csv TEMP.csv tags.csv ...
          ...
  processed/
    physionet_e4/
      subjects_stress.txt
      intermediate/
        STRESS/
          S01_aligned64.npz
          S01_labeled64.npz
          ...
      STRESS/
        S01_windows.npz
        ...
      STRESS_summary.csv
```

---

## Key Outputs

For each subject `SUBJ`:

1) **Aligned file**
- `data/processed/physionet_e4/intermediate/STRESS/SUBJ_aligned64.npz`
- Contains aligned 64 Hz signals: `t64, bvp, acc_mag, temp, eda, tags`

2) **Labeled file**
- `data/processed/physionet_e4/intermediate/STRESS/SUBJ_labeled64.npz`
- Same as aligned file + `label64` (per-sample label)

3) **Final training windows**
- `data/processed/physionet_e4/STRESS/SUBJ_windows.npz`
- `X`: `(N, 3840, 3)` = [BVP, ACC_mag, TEMP]
- `Y`: `(N, 3840)` = EDA (optional teacher signal)
- `L`: `(N,)` = window labels {0,1}

---

## Environment Variable: `DATA_ROOT`

All scripts support an optional `DATA_ROOT` env var:

- If `DATA_ROOT` is **not** set, scripts use: `./data` (repo-local)
- If `DATA_ROOT` is set (e.g., PVC), scripts use: `$DATA_ROOT/...`

### Examples
Local (default):
- uses `./data`

PVC:
- `DATA_ROOT=/pvc/physionet_e4/data`

---

## Installation

Python deps:
```bash
pip install -r requirements.txt
```

Current preprocessing requires:
- `numpy`

Optional download tools (system-level, not Python):
- `aws` (AWS CLI) **or**
- `wget`

---

## Downloading the Dataset

You can download via either:

### Option A (recommended): AWS S3 sync
```bash
aws s3 sync --no-sign-request s3://physionet-open/wearable-device-dataset/1.0.1/ data/raw/physionet_e4/
```

### Option B: wget mirror
```bash
wget -r -N -c -np -P data/raw/physionet_e4 https://physionet.org/files/wearable-device-dataset/1.0.1/
```

After download, you should have:
```
data/raw/physionet_e4/Wearable_Dataset/STRESS/S01/BVP.csv
```

---

## Reproduce Everything (Recommended)

### One-command reproduction (Python)
This will:
1) Download (if missing and if `aws` or `wget` exists)
2) Build subject list
3) Preprocess all subjects
4) Write a dataset summary CSV

```bash
python reproduce.py
```

---

## Manual Pipeline (Step-by-step)

### 1) Build subject list
```bash
python scripts/make_subject_list.py
```

Writes:
- `data/processed/physionet_e4/subjects_stress.txt`

We exclude known problematic subjects:
- `S02` (duplicated chunks)
- `f07` (dock on, invalid BVP/TEMP)
- `f14_a`, `f14_b` (split recording)

### 2) Run full preprocessing for all subjects
```bash
python scripts/run_all_subjects.py
```

This runs, for each subject:
- `preprocess_one_subject.py` -> aligned64
- `label_and_window_subject.py` -> labeled64 + windows

### 3) Summarize dataset
```bash
python scripts/summarize_windows.py
```

Writes:
- `data/processed/physionet_e4/STRESS_summary.csv`

---

## Labeling Details (Protocol-defined)

We label based on protocol stage type (WESAD-like):

### V1 subjects (`S01–S18`)
Stages include:
- Rest: `Baseline`, `First Rest`, `Second Rest`
- Stress tasks: `Stroop`, `TMCT`, `Real Opinion`, `Opposite Opinion`, `Subtract`
- Ignore: `SL` (self-report segments)

### V2 subjects (`f01–f18`)
In our files, tag boundaries are fewer (often 9 tags → 10 segments).
We handle this robustly by:
- splitting by tags
- ignoring “short” segments (< 1 minute)
- assigning remaining “major segments” in the canonical stage order
- mapping stage types to stress/non-stress

---

## Quick Verification Checklist

After a full run:

1) You should have 33 window files (given exclusions):
```bash
ls data/processed/physionet_e4/STRESS/*_windows.npz | wc -l
```

2) Summary CSV should exist:
- `data/processed/physionet_e4/STRESS_summary.csv`

3) Spot-check a subject file:
```bash
python scripts/label_and_window_subject.py --subject S01
python scripts/label_and_window_subject.py --subject f01
```

---

## Notes / Common Issues

### Windows: running bash scripts
- If you want to run `.sh` scripts on Windows, use **Git Bash**.
- PowerShell calling `bash` may route to WSL; if WSL is not installed it will error.

### Download tools not found
If `aws` or `wget` is missing:
- install AWS CLI (Windows installer) or use a container image that includes aws/wget
- or manually download once and place under `data/raw/physionet_e4/`

---

