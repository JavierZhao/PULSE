# PULSE

**PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning**

This repository contains the research code for **PULSE**, a framework for learning from a rich sensor during training while deploying on cheaper sensors at inference time. In the wearable stress-monitoring setting studied in the paper, **electrodermal activity (EDA)** acts as a privileged teacher modality, while **ECG, BVP, accelerometry, and temperature** are the deployable student modalities.

PULSE is motivated by a common embodied-sensing problem: the most informative modality in the lab is often too fragile, expensive, or inconvenient to rely on after deployment. The code in this repository implements the training pipeline used to transfer that privileged information into representations learned by lower-cost sensors.

The accompanying paper reports that, on **WESAD leave-one-subject-out evaluation**, PULSE reaches **0.994 AUROC** and **0.988 AUPRC** for binary stress detection **without EDA at inference time**. The paper also reports cross-dataset results on PhysioNet STRESS; the public preprocessing utilities in this repository are currently centered on **WESAD**.

## Paper

**Zihan Zhao, Kaushik Pendiyala, Masood Mortazavi, Ning Yan**  
*PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning*  
Accepted at the **CVPR 2026 Workshop on Sense of Space**

## What This Repository Includes

- WESAD preprocessing into leave-one-subject-out fold archives
- Self-supervised pretraining for:
  - privileged EDA teacher models
  - cheap-sensor student encoders
  - multiple backbone families, including transformer, ResNet1D, and TCN
- Knowledge distillation from a frozen privileged teacher to deployable students
- Supervised finetuning for binary or three-class stress prediction
- Evaluation, fold summarization, and plotting utilities

## Method Overview

PULSE is organized as a three-stage pipeline:

1. **Privileged teacher pretraining**
   - Train an EDA-only encoder with masked reconstruction.
2. **Student pretraining**
   - Pretrain the deployable sensor encoders with shared/private embeddings and cross-modal alignment.
3. **Privileged knowledge transfer + finetuning**
   - Distill the frozen teacher into the student encoders, then finetune a classifier that runs without the privileged sensor.

The core design idea is to split each student representation into:

- **shared embeddings**, which capture modality-invariant information aligned across sensors and matched to the teacher
- **private embeddings**, which preserve modality-specific structure needed for reconstruction and collapse prevention

## Repository Layout

- `src/data/preprocess.py`
  - Builds leave-one-subject-out WESAD folds from raw CSV plus original WESAD labels
- `src/data/wesad_dataset.py`
  - Dataset loader for preprocessed fold files
- `src/pretrain/train_eda_mae.py`
  - Trains the privileged EDA teacher
- `src/pretrain/train_cheap_mae.py`
  - Pretrains student encoders on deployable sensors
- `src/kd/train_kd.py`
  - Distills the frozen teacher into student encoders
- `src/finetune/finetune.py`
  - Trains the downstream stress classifier
- `src/finetune/test.py`
  - Evaluates saved finetuning runs and writes metrics/plots
- `src/finetune/summarize_folds.py`
  - Aggregates metrics across folds
- `src/model/`
  - Backbone implementations and registry
- `env/dl.yml`, `env/requirements.txt`
  - Environment definitions

## Setup

Conda:

```bash
conda env create -f env/dl.yml
conda activate dl
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

## Data

### WESAD

Download and extract the WESAD dataset from the official source:

```bash
mkdir -p data
cd data
wget --content-disposition "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
unzip WESAD.zip
cd ..
```

Official dataset download and documentation are available from the [WESAD download page](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download); this repository does not bundle the original WESAD documentation PDF.

The expected raw layout is:

```text
data/WESAD/S2/S2.pkl
data/WESAD/S3/S3.pkl
...
```

### Convert raw WESAD files to CSV

The preprocessing pipeline expects per-subject raw CSV exports. Generate them with:

```bash
python src/data/data_wrangling.py --mode raw
```

This creates directories such as:

```text
data/S2_raw_data/
data/S3_raw_data/
...
```

### Build leave-one-subject-out folds

Before running the fold builder, update the path constants near the top of `src/data/preprocess.py`:

- `DATA_DIR`
- `WESAD_PKL_DIR`
- `OUTPUT_DIR`

Then run:

```bash
python -m src.data.preprocess
```

This produces `fold_<subject>.npz` archives containing train/test windows, labels, subject IDs, and normalization statistics.

## Reproducing the Main Pipeline

The examples below use explicit paths rather than relying on the repository's built-in default paths, which currently reflect the authors' research environment.

### 1. Train the privileged EDA teacher

```bash
python -m src.pretrain.train_eda_mae \
  --run_name eda_teacher \
  --data_path /path/to/preprocessed_wesad \
  --output_path ./results/eda_mae \
  --fold_number 17 \
  --device cuda:0 \
  --backbone transformer
```

Output checkpoint:

```text
./results/eda_mae/eda_teacher/models/best_ckpt.pt
```

### 2. Pretrain deployable student encoders

```bash
python -m src.pretrain.train_cheap_mae \
  --run_name cheap_pretrain \
  --data_path /path/to/preprocessed_wesad \
  --output_path ./results/cheap_maes \
  --fold_number 17 \
  --device cuda:0 \
  --modalities ecg,bvp,acc,temp \
  --backbone transformer
```

Output checkpoint:

```text
./results/cheap_maes/cheap_pretrain/models/best_ckpt.pt
```

### 3. Distill from the privileged teacher

```bash
python -m src.kd.train_kd \
  --run_name pulse_kd \
  --data_path /path/to/preprocessed_wesad \
  --output_path ./results/kd \
  --fold_number 17 \
  --device cuda:0 \
  --modalities ecg,bvp,acc,temp \
  --teacher_ckpt_path ./results/eda_mae/eda_teacher/models/best_ckpt.pt \
  --students_ckpt_path ./results/cheap_maes/cheap_pretrain/models/best_ckpt.pt
```

Outputs:

```text
./results/kd/pulse_kd/models/best_ckpt_full_kd.pt
./results/kd/pulse_kd/models/best_ckpt.pt
```

The `best_ckpt.pt` file is the student-only checkpoint intended for finetuning.

### 4. Finetune the downstream classifier

```bash
python -m src.finetune.finetune \
  --run_name ./results/kd/pulse_kd/models/best_ckpt.pt \
  --save_name pulse_finetune \
  --data_path /path/to/preprocessed_wesad \
  --output_path ./results/finetuned_models \
  --fold_number 17 \
  --val_subject_id 16 \
  --device cuda:0 \
  --modalities ecg,bvp,acc,temp \
  --num_epochs 150 \
  --batch_size 128 \
  --fuse_embeddings
```

Notes:

- `--run_name` is also used to resolve the pretrained checkpoint path unless `--from_scratch` is set.
- `--save_name` controls the finetuning output directory.
- `--modalities` accepts a comma-separated subset of `{ecg,bvp,acc,temp,eda}` or `all`.
- Use `--three_class` for baseline/stress/amusement classification.

### 5. Evaluate a run or a parent directory of folds

```bash
python -m src.finetune.test \
  --run_dir ./results/finetuned_models/pulse_finetune \
  --data_path /path/to/preprocessed_wesad \
  --device cuda:0 \
  --batch_size 256 \
  --modalities ecg,bvp,acc,temp
```

Typical outputs include:

- `test_metrics.json`
- `test_classification_report*.txt`
- `test_confusion_matrix*.png`
- `test_roc_curve.png`
- `test_pr_curve.png`
- fold summary CSV/JSON files when evaluating a parent directory

## Other Training Utilities

The repository also includes alternative pretraining scripts:

- `src/pretrain/train_contrastive.py`
- `src/pretrain/train_vicreg.py`
- `src/pretrain/train_autoregressive.py`

These are useful for baseline comparisons or ablations against the main masked-reconstruction pipeline used in the paper.

## Practical Notes

- The public release is currently **WESAD-first**. Some paper results use additional datasets that are not yet exposed through a matching public preprocessing script here.
- Several preprocessing and visualization scripts still contain editable path constants from the original research workflow. Public users should override CLI paths where available and update those constants where necessary.
- `src/finetune/run_folds.sh` is a convenience launcher for multi-fold experiments in a `tmux` plus multi-GPU environment. It is optional and assumes some shell-level setup.
- Check `--help` on the main scripts for the full list of architecture and optimization options.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{zhao2026pulse,
  title={PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning},
  author={Zhao, Zihan and Pendiyala, Kaushik and Mortazavi, Masood and Yan, Ning},
  booktitle={Proceedings of the CVPR 2026 Workshop on Sense of Space},
  year={2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
