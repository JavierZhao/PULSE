# EDA_Gen (Time-Series) — Server Guide

This guide uses concrete server paths to reproduce runs on the same machine layout.

---

## Environment

```bash
conda env create -f dl.yml
conda activate dl
```

---

## Data Preprocessing (LOSO)

Edit `src/data/preprocess.py` to use these paths:
- `DATA_DIR="/fd24T/zzhao3/EDA/data"`
- `WESAD_PKL_DIR="/fd24T/zzhao3/EDA/data/WESAD"`
- `OUTPUT_DIR="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid"`

Run:

```bash
python -m src.data.preprocess
```

---

## Finetuning (single fold)

```bash
python -m src.finetune.finetune \
  --run_name finetune_baseline \
  --data_path /fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --num_epochs 150 \
  --batch_size 128
```

Using pretrained cheap MAEs:

```bash
python -m src.finetune.finetune \
  --run_name finetune_pretrained \
  --data_path /fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --num_epochs 150 \
  --batch_size 128
```

---

## Pretraining Backbones (Time-Series)

You can warm start by training from scratch with finetune (supervised) or point to an existing cheap MAE pretraining run:

```bash
python -m src.finetune.finetune \
  --run_name pretrain_like_supervised \
  --data_path /fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --val_subject_id 16 \
  --modalities all \
  --from_scratch \
  --num_epochs 150 \
  --batch_size 128
```

Ensure a students checkpoint under:
```
/fd24T/zzhao3/EDA/results/cheap_maes/<run>/models/best_ckpt.pt
```
so finetune loads it automatically when `--from_scratch` is not set.

---

## Knowledge Distillation (EDA → CheapSensor)

```bash
python -m src.kd.train_kd \
  --run_name kd_example \
  --data_path /fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid \
  --device cuda:0 \
  --fold_number 17 \
  --modalities ecg,bvp,acc,temp \
  --batch_size 128 \
  --num_epochs 300 \
  --learning_rate 1e-4 \
  --mask_ratio 0.75 \
  --layers_to_match 3,5,7 \
  --kd_loss cosine \
  --lambda_hid 1.0 \
  --lambda_emb 1.0 \
  --perp_weight 0.0 \
  --recon_weight 0.0 \
  --teacher_ckpt_path /fd24T/zzhao3/EDA/results/eda_mae/300p/models/best_ckpt.pt \
  --students_ckpt_path /fd24T/zzhao3/EDA/results/cheap_maes/hinge_loss/default/models/best_ckpt.pt
```

Outputs:
```
/fd24T/zzhao3/EDA/results/kd/<run_name>/models/best_ckpt_full_kd.pt
/fd24T/zzhao3/EDA/results/kd/<run_name>/models/best_ckpt.pt
```

---

## Multi-Fold Launcher

Use `src/finetune/run_folds.sh` with:

```bash
WORKDIR="/fd24T/zzhao3/EDA/EDA_Gen/src/finetune"
ENV_ACTIVATE="$HOME/envs/dl/bin/activate"
RUN_NAME="hinge_loss/default"
SESSION_PREFIX="hid-4_finetune"
```

Launch:

```bash
bash /fd24T/zzhao3/EDA/EDA_Gen/src/finetune/run_folds.sh
```

Logs go to `${WORKDIR}/logs`. Each tmux session binds a GPU via `CUDA_VISIBLE_DEVICES`.

---

## Testing (evaluate runs)

Evaluate one run directory or a parent directory of runs:

```bash
python -m src.finetune.test \
  --run_dir /fd24T/zzhao3/EDA/results/finetuned_models/your_parent_dir \
  --device cuda:0 \
  --batch_size 256 \
  --data_path /fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s_sid \
  --modalities all
```

Outputs per run include `test_metrics.json`, confusion matrices, ROC/PR curves; aggregate CSV/JSON summaries are written at the parent level.

---

## Outputs

Models and logs are stored under:
```
/fd24T/zzhao3/EDA/EDA_Gen/src/finetune/logs
/fd24T/zzhao3/EDA/results/finetuned_models/<run_name>
```


