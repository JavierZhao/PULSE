# scripts/preprocess_one_subject.py
# Align raw Empatica E4 signals to a common 64 Hz timeline and save *_aligned64.npz
# Works locally and on PVC via DATA_ROOT env var.

from pathlib import Path
from datetime import datetime, timezone
import os
import numpy as np

# ---- Portable paths ----
SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../physionet_preprocessing
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_ROOT = DATA_ROOT / "raw" / "physionet_e4" / "Wearable_Dataset" / "STRESS"
OUT_ROOT = DATA_ROOT / "processed" / "physionet_e4" / "intermediate" / "STRESS"

TARGET_FS = 64.0


def parse_time_to_unix_seconds(x: str) -> float:
    """
    Accepts either:
      - unix timestamp (float/int)
      - datetime like '2013-02-20 17:55:19' (assume UTC)
    Returns unix seconds as float.
    """
    s = str(x).strip()
    try:
        return float(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()


def load_e4_csv(path: Path):
    """
    Empatica E4-like CSV:
      line 0: start time (unix seconds OR 'YYYY-MM-DD HH:MM:SS')
      line 1: sampling rate (Hz)
      line 2+: samples (1 col for most, 3 cols for ACC)
    Returns: (t0, fs, data_2d_float32)
    """
    with open(path, "r", encoding="utf-8") as f:
        line0 = f.readline().strip()
        line1 = f.readline().strip()

    start_token = line0.split(",")[0]
    fs_token = line1.split(",")[0]

    t0 = parse_time_to_unix_seconds(start_token)
    fs = float(fs_token)

    data = np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=2)

    # Ensure 2D: (N,) -> (N,1)
    data = np.atleast_2d(data)
    if data.shape[0] == 1 and data.shape[1] > 1:
        # np.atleast_2d on a 1D array makes shape (1,N); fix to (N,1)
        data = data.T

    return t0, fs, data.astype(np.float32)


def load_tags(path: Path):
    """
    tags.csv: one timestamp per row (may be datetime string or unix seconds).
    Returns: (num_tags,) float64 unix seconds
    """
    tags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tags.append(parse_time_to_unix_seconds(s.split(",")[0]))
    return np.array(tags, dtype=np.float64)


def make_time_axis(t0: float, fs: float, n: int):
    return t0 + np.arange(n, dtype=np.float64) / fs


def interp_to(t_src, x_src, t_tgt):
    return np.interp(t_tgt, t_src, x_src).astype(np.float32)


def main(subject="S01"):
    subj_dir = RAW_ROOT / subject
    if not subj_dir.exists():
        raise FileNotFoundError(f"Missing subject folder: {subj_dir.resolve()}")

    # Load streams
    t0_bvp, fs_bvp, bvp = load_e4_csv(subj_dir / "BVP.csv")    # (N,1)
    t0_acc, fs_acc, acc = load_e4_csv(subj_dir / "ACC.csv")    # (N,3)
    t0_eda, fs_eda, eda = load_e4_csv(subj_dir / "EDA.csv")    # (N,1)
    t0_tmp, fs_tmp, tmp = load_e4_csv(subj_dir / "TEMP.csv")   # (N,1)
    tags = load_tags(subj_dir / "tags.csv")

    # Flatten single-channel signals to 1D
    bvp = bvp[:, 0]
    eda = eda[:, 0]
    tmp = tmp[:, 0]

    # ACC magnitude
    if acc.shape[1] != 3:
        raise ValueError(f"Expected ACC to have 3 columns, got shape {acc.shape}")
    acc_mag = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2).astype(np.float32)

    # Time axes
    t_bvp = make_time_axis(t0_bvp, fs_bvp, len(bvp))
    t_acc = make_time_axis(t0_acc, fs_acc, len(acc_mag))
    t_eda = make_time_axis(t0_eda, fs_eda, len(eda))
    t_tmp = make_time_axis(t0_tmp, fs_tmp, len(tmp))

    # Common intersection so all modalities exist
    t_start = max(t_bvp[0], t_acc[0], t_eda[0], t_tmp[0])
    t_end = min(t_bvp[-1], t_acc[-1], t_eda[-1], t_tmp[-1])

    # Target timeline at 64 Hz
    t64 = np.arange(t_start, t_end, 1.0 / TARGET_FS, dtype=np.float64)

    # Resample onto t64
    bvp64 = interp_to(t_bvp, bvp, t64)
    acc64 = interp_to(t_acc, acc_mag, t64)
    eda64 = interp_to(t_eda, eda, t64)
    tmp64 = interp_to(t_tmp, tmp, t64)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{subject}_aligned64.npz"
    np.savez_compressed(
        out_path,
        t64=t64,
        bvp=bvp64,
        acc_mag=acc64,
        eda=eda64,
        temp=tmp64,
        tags=tags,
        meta=np.array([t_start, t_end, TARGET_FS], dtype=np.float64),
    )

    dur_min = (t64[-1] - t64[0]) / 60.0
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Saved: {out_path}")
    print(f"Duration: {dur_min:.2f} min | Samples@64Hz: {len(t64)}")
    print(f"Native fs: BVP={fs_bvp}, ACC={fs_acc}, EDA={fs_eda}, TEMP={fs_tmp}")
    print(f"Num tags: {len(tags)} | First 5 tags: {tags[:5]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01")
    args = ap.parse_args()
    main(args.subject)