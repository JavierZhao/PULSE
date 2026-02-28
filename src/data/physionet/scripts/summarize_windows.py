# scripts/summarize_windows.py
# Summarize all processed *_windows.npz files and write summary.csv

from pathlib import Path
import csv
import numpy as np

WINDOW_DIR = Path("data/processed/physionet_e4/STRESS")
OUT_CSV = Path("data/processed/physionet_e4/STRESS_summary.csv")

def summarize_one(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # Expected keys from your pipeline:
    #   X: (N, 3840, 3)
    #   Y: (N, 3840)
    #   L: (N,)
    X = d["X"]
    L = d["L"]

    n = int(X.shape[0])
    unique, counts = np.unique(L, return_counts=True)
    label_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    n0 = label_counts.get(0, 0)
    n1 = label_counts.get(1, 0)
    frac1 = (n1 / n) if n > 0 else 0.0

    return {
        "subject": npz_path.stem.replace("_windows", ""),
        "windows_total": n,
        "label0": n0,
        "label1": n1,
        "frac_label1": frac1,
        "win_len": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
    }

def main():
    files = sorted(WINDOW_DIR.glob("*_windows.npz"))
    if not files:
        raise FileNotFoundError(f"No *_windows.npz files found in {WINDOW_DIR.resolve()}")

    rows = []
    totals = {"windows_total": 0, "label0": 0, "label1": 0}
    bad = []

    for f in files:
        try:
            row = summarize_one(f)
            rows.append(row)
            totals["windows_total"] += row["windows_total"]
            totals["label0"] += row["label0"]
            totals["label1"] += row["label1"]
        except Exception as e:
            bad.append((f.name, str(e)))

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "windows_total", "label0", "label1", "frac_label1", "win_len", "n_channels"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print summary
    print(f"Found {len(files)} files.")
    print(f"Wrote: {OUT_CSV}")
    if totals["windows_total"] > 0:
        frac1 = totals["label1"] / totals["windows_total"]
    else:
        frac1 = 0.0

    print("Totals:")
    print(f"  windows_total = {totals['windows_total']}")
    print(f"  label0        = {totals['label0']}")
    print(f"  label1        = {totals['label1']}")
    print(f"  frac_label1   = {frac1:.4f}")

    if bad:
        print("\nWARNING: some files failed to read:")
        for name, msg in bad:
            print(" ", name, "->", msg)

if __name__ == "__main__":
    main()