# PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors\\for Embodied Multi-Sensory Learning
---

## 1) Environment

Conda (recommended):

```bash
conda env create -f env/dl.yml
conda activate dl
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Repository Structure (time-series)

- `src/data/preprocess.py`: Build LOSO folds from raw CSV + WESAD pickle
- `src/data/wesad_dataset.py`: Dataset for time-series folds
- `src/model/CheapSensorMAE.py`: Time-series MAE backbone

- **Pretraining:**
  - `src/pretrain/pretrain.py`: Self-supervised pretraining for CheapSensorMAE backbones

- **Finetuning:**
  - `src/finetune/finetune.py`: Finetuning script for stress classification
  - `src/finetune/run_folds.sh`: tmux launcher for multi-fold runs

- **Knowledge Transfer (Distillation):**
  - `src/kd/train_kd.py`: Knowledge distillation (teacher-student) training script

- **Testing & Evaluation:**
  - `src/finetune/test.py`: Evaluate trained models and output metrics
  - `src/finetune/summarize_folds.py`: Aggregate and summarize results across folds

- `results/`: Outputs (models, logs) created at runtime

---

## 3) Data Preparation (LOSO time-series)

### Step 1: Download WESAD Dataset

Download the WESAD dataset:

```bash
mkdir data
cd data
wget --content-disposition "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
cd ..
```

This will download `WESAD.zip` to the `data/` directory.

### Step 2: Extract WESAD Dataset

Extract the WESAD dataset to `data/WESAD/`:

```bash
unzip data/WESAD.zip -d data/
```

This will create the directory structure: `data/WESAD/SXX/SXX.pkl` for all subjects.

### Step 3: Generate Raw CSV Files

Convert the pickle files to raw CSVs using the data wrangling script:

```bash
python src/data/data_wrangling.py --mode raw
```

This creates `data/SXX_raw_data/*.csv` directories with sensor data for each subject (ECG, BVP, EDA, temperature, accelerometer).

### Step 4: Preprocess Data into LOSO Folds

Edit `src/data/preprocess.py` to set:
- `DATA_DIR`, `WESAD_PKL_DIR`, `OUTPUT_DIR`
- `TARGET_FS`, `WINDOW_SECONDS`, `STRIDE_SECONDS`

Then run:

```bash
python -m src.data.preprocess
```

This creates `fold_*.npz` under `OUTPUT_DIR` containing train/test windows, labels and subject IDs.

---

## 4) Pretraining Backbones (Time-Series)

```bash
python -m src.pretrain.train_cheap_mae \
  --run_name pretrain_job \
  --data_path ./preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --num_epochs 300 \
  --batch_size 128
```
