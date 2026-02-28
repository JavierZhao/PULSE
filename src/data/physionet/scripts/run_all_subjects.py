from pathlib import Path
import subprocess
import sys

SUBJECT_LIST = Path("data/processed/physionet_e4/subjects_stress.txt")

SCRIPT_ALIGN = Path("scripts/preprocess_one_subject.py")
SCRIPT_LABEL = Path("scripts/label_and_window_subject.py")

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def main():
    subjects = [s.strip() for s in SUBJECT_LIST.read_text().splitlines() if s.strip()]
    print(f"Running full preprocessing for {len(subjects)} subjects...")

    ok, fail = 0, 0
    failures = []

    for s in subjects:
        print(f"\n=== {s} ===")

        # 1) create aligned64
        r1 = run([sys.executable, str(SCRIPT_ALIGN), "--subject", s])
        if r1.returncode != 0:
            fail += 1
            failures.append((s, "align"))
            print("FAILED at ALIGN")
            print(r1.stdout)
            print(r1.stderr)
            continue

        # 2) label + windows
        r2 = run([sys.executable, str(SCRIPT_LABEL), "--subject", s])
        if r2.returncode != 0:
            fail += 1
            failures.append((s, "label"))
            print("FAILED at LABEL/WINDOW")
            print(r2.stdout)
            print(r2.stderr)
            continue

        ok += 1
        print(r2.stdout.strip())

    print("\n====================")
    print(f"Done. OK={ok}, FAIL={fail}")
    if failures:
        print("Failures:")
        for s, stage in failures:
            print(" ", s, stage)

if __name__ == "__main__":
    main()