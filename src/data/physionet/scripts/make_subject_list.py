# scripts/make_subject_list.py
# Build a subject list for the STRESS protocol (exclude known-bad subjects).
# Works locally and on PVC via DATA_ROOT env var.

from pathlib import Path
import os

# ---- Portable paths ----
SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../physionet_preprocessing
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_ROOT = DATA_ROOT / "raw" / "physionet_e4" / "Wearable_Dataset" / "STRESS"
OUT_PATH = DATA_ROOT / "processed" / "physionet_e4" / "subjects_stress.txt"

# Known issues from data_constraints.txt:
EXCLUDE = {
    "S02",   # duplicated chunk mid-file
    "f07",   # dock on -> BVP/TEMP invalid
    "f14",   # split into f14_a + f14_b (handle later)
    "f14_a",
    "f14_b",
}

def is_subject_dir(p: Path) -> bool:
    # Require at least BVP + ACC for a valid subject folder
    return p.is_dir() and (p / "BVP.csv").exists() and (p / "ACC.csv").exists()

def main():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Can't find raw STRESS folder: {RAW_ROOT.resolve()}")

    subjects = sorted([p.name for p in RAW_ROOT.iterdir() if is_subject_dir(p)])
    kept = [s for s in subjects if s not in EXCLUDE]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Found {len(subjects)} subject folders.")
    print(f"Kept  {len(kept)} after exclusions.")
    print(f"Wrote -> {OUT_PATH}")
    print("First 20 kept:", kept[:20])

if __name__ == "__main__":
    main()
