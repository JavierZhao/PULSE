import os
import sys
import shutil
import subprocess
from pathlib import Path

def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run(cmd, cwd=None):
    print("\n>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with code {r.returncode}: {' '.join(cmd)}")

def main():
    project_root = Path(__file__).resolve().parent
    data_root = Path(os.environ.get("DATA_ROOT", project_root / "data"))
    raw_root = data_root / "raw" / "physionet_e4"
    processed_root = data_root / "processed" / "physionet_e4"

    print("PROJECT_ROOT:", project_root)
    print("DATA_ROOT:", data_root)
    print("RAW_ROOT:", raw_root)
    print("PROCESSED_ROOT:", processed_root)

    raw_marker = raw_root / "Wearable_Dataset"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    # 1) Download if needed
    if not raw_marker.exists():
        print("\n[download] Raw dataset not found; attempting download...")

        if which("aws"):
            run([
                "aws", "s3", "sync", "--no-sign-request",
                "s3://physionet-open/wearable-device-dataset/1.0.1/",
                str(raw_root)
            ])
        elif which("wget"):
            # NOTE: wget -r may create nested physionet.org/... directories. We'll handle if needed.
            run([
                "wget", "-r", "-N", "-c", "-np",
                "-P", str(raw_root),
                "https://physionet.org/files/wearable-device-dataset/1.0.1/"
            ])
        else:
            raise RuntimeError(
                "Cannot download: neither 'aws' nor 'wget' is available.\n"
                "Install AWS CLI (recommended) or wget, OR manually place the dataset under:\n"
                f"  {raw_root}\n"
                "Expected to contain:\n"
                "  Wearable_Dataset/STRESS/S01/BVP.csv ..."
            )

        # After download, verify marker exists (or explain nested wget layout)
        if not raw_marker.exists():
            # Common wget layout: raw_root/physionet.org/files/wearable-device-dataset/1.0.1/...
            nested = raw_root / "physionet.org" / "files" / "wearable-device-dataset" / "1.0.1"
            if nested.exists():
                print("\n[download] Detected wget nested layout. Moving contents up...")
                # Move nested contents into raw_root
                for item in nested.iterdir():
                    target = raw_root / item.name
                    if target.exists():
                        continue
                    item.rename(target)

            if not raw_marker.exists():
                raise RuntimeError(
                    "Download completed but expected folder 'Wearable_Dataset' not found.\n"
                    f"Look inside: {raw_root}\n"
                    "and confirm where the dataset extracted."
                )
    else:
        print("\n[download] Raw dataset already present; skipping download.")

    # 2) Run preprocessing pipeline (uses DATA_ROOT internally via your updated scripts)
    env = os.environ.copy()
    env["DATA_ROOT"] = str(data_root)

    def run_py(script_rel):
        script_path = project_root / script_rel
        run([sys.executable, str(script_path)], cwd=project_root)

    # If your scripts already use DATA_ROOT, just run them
    print("\n[preprocess] Running subject list + preprocessing + summary...")
    subprocess.check_call([sys.executable, str(project_root / "scripts" / "make_subject_list.py")], cwd=project_root, env=env)
    subprocess.check_call([sys.executable, str(project_root / "scripts" / "run_all_subjects.py")], cwd=project_root, env=env)
    subprocess.check_call([sys.executable, str(project_root / "scripts" / "summarize_windows.py")], cwd=project_root, env=env)

    print("\nDone.")
    print("Processed outputs at:", processed_root)

if __name__ == "__main__":
    main()