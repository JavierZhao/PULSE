#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DATA_ROOT=/pvc/physionet_e4/data bash scripts/download_physionet_e4.sh
#
# This will download into:
#   $DATA_ROOT/raw/physionet_e4/

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
RAW_DIR="$DATA_ROOT/raw/physionet_e4"

mkdir -p "$RAW_DIR"

echo "DATA_ROOT: $DATA_ROOT"
echo "Downloading to: $RAW_DIR"

# Option A: AWS S3 (recommended: cleaner, resumable, no-sign-request)
if command -v aws >/dev/null 2>&1; then
  echo "[download] Using aws s3 sync..."
  aws s3 sync --no-sign-request \
    "s3://physionet-open/wearable-device-dataset/1.0.1/" \
    "$RAW_DIR/"
  echo "[download] Done."
  exit 0
fi

# Option B: wget mirror
if command -v wget >/dev/null 2>&1; then
  echo "[download] Using wget mirror..."
  wget -r -N -c -np \
    -P "$RAW_DIR" \
    "https://physionet.org/files/wearable-device-dataset/1.0.1/"
  echo "[download] Done."
  exit 0
fi

echo "ERROR: Neither 'aws' nor 'wget' is available in this environment."
echo "Install awscli or wget in your container, or download manually once to $RAW_DIR."
exit 1