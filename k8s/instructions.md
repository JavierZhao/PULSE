Here are the commands for all new pretraining configurations:

### New Backbone Architectures (Masked Reconstruction)

These use the same MAE pretraining objective but with different encoder/decoder architectures:

```bash
# ResNet1D backbone
python -m src.pretrain.train_cheap_mae \
    --backbone resnet1d \
    --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100

# TCN backbone
python -m src.pretrain.train_cheap_mae \
    --backbone tcn \
    --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100
```

### New Pretraining Strategies (Transformer backbone)

These use the default Transformer encoder but with different self-supervised objectives:

```bash
# Contrastive (SimCLR-style NT-Xent)
python -m src.pretrain.train_contrastive \
    --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100

# VICReg (Variance-Invariance-Covariance Regularization)
python -m src.pretrain.train_vicreg \
    --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100

# Autoregressive (GPT-style next-patch prediction)
python -m src.pretrain.train_autoregressive \
    --data_path /path/to/preprocessed --fold_number 17 --num_epochs 100
```

### Finetuning with non-default backbones

After pretraining, pass `--backbone` to match the pretrained checkpoint:

```bash
# Finetune a ResNet1D-pretrained model
python -m src.finetune.finetune \
    --backbone resnet1d \
    --run_name /path/to/resnet1d/pretrained/models/best_ckpt.pt \
    --fold_number 17

# Finetune a contrastive-pretrained model
python -m src.finetune.finetune \
    --backbone contrastive \
    --run_name /path/to/contrastive/pretrained/models/best_ckpt.pt \
    --fold_number 17

# Finetune from scratch (no pretraining) with a different backbone
python -m src.finetune.finetune \
    --backbone autoregressive \
    --from_scratch \
    --fold_number 17
```

### Testing with non-default backbones

```bash
python -m src.finetune.test \
    --backbone resnet1d \
    --run_dir /path/to/finetuned/run
```

### Knowledge Distillation with non-default backbones

```bash
# Teacher=transformer, Students=resnet1d
python -m src.kd.train_kd \
    --teacher_backbone transformer \
    --backbone resnet1d \
    --teacher_ckpt_path /path/to/teacher/best_ckpt.pt \
    --students_ckpt_path /path/to/students/best_ckpt.pt
```

### Key method-specific hyperparameters

| Script | Notable flags |
| --- | --- |
| `train_contrastive` | `--proj_dim 128`, `--temperature 0.1` |
| `train_vicreg` | `--expand_dim 2048`, `--inv_weight 25`, `--var_weight 25`, `--cov_weight 1` |
| `train_autoregressive` | `--dropout 0.0` |
| `train_cheap_mae --backbone resnet1d` | Same MAE flags (`--mask_ratio`, `--embed_dim`, etc.) |
| `train_cheap_mae --backbone tcn` | Same MAE flags |

All scripts also accept `--include_eda`, `--alignment_loss_weight`, `--hinge_alpha`, and the standard training flags (`--batch_size`, `--learning_rate`, `--device`, etc.).