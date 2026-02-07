#!/usr/bin/env bash
set -euo pipefail

# --- config you might tweak ---
WORKDIR="~/EDA/EDA_Gen/src/finetune"
ENV_ACTIVATE="$HOME/envs/dl/bin/activate"
RUN_NAME="~/EDA/results/eda_mae/300p/models/best_ckpt.pt"
BATCH_SIZE=128
EPOCHS=300
RESTART_EPOCHS=300
SAMPLE_RATE=40
SESSION_PREFIX="eda_finetune"
LOGDIR="./tmux_logs/${SESSION_PREFIX}"
MODALITIES='eda'
SAVE_NAME='eda_only'
# ------------------------------


mkdir -p $LOGDIR

# Ensure tmux exists
TMUX_BIN="$(command -v tmux || true)"
if [[ -z "${TMUX_BIN}" ]]; then
  echo "ERROR: tmux not found in PATH." >&2
  exit 127
fi
echo "Using tmux at: ${TMUX_BIN} \("$(${TMUX_BIN} -V)"\)"
echo "Logs -> ${LOGDIR}"

# Detect GPUs and cycle across them
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -eq 0 ]]; then
  echo "ERROR: No GPUs detected by nvidia-smi." >&2
  exit 2
fi
echo "Detected ${NUM_GPUS} GPU\(s\). Will assign one GPU per session via CUDA_VISIBLE_DEVICES."
echo

# Folds to run
FOLDS=(2 3 4 5 6 7 8 9 10 11 13 14 15)

for i in "${!FOLDS[@]}"; do
  FOLD="${FOLDS[$i]}"
  GPU="$(( i % NUM_GPUS ))"  # pick a physical GPU by index
  SESS="${SESSION_PREFIX}_f${FOLD}"
  TAG="/f_${FOLD}"
  LOGFILE="${LOGDIR}/fold_${FOLD}.log"

  # Kill any existing session with the same name (optional)
  if ${TMUX_BIN} has-session -t "$SESS" 2>/dev/null; then
    ${TMUX_BIN} kill-session -t "$SESS"
  fi

  echo "Launching: session=${SESS}  fold=${FOLD}  phys_gpu=${GPU}"
  echo "  -> log: ${LOGFILE}"

  # Build a per-fold runner script to avoid quoting issues
  RUNNER="${LOGDIR}/run_fold_${FOLD}.sh"
  cat > "${RUNNER}" <<'EOF'
set -Eeuo pipefail
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
cd __WORKDIR__
source __ENV_ACTIVATE__
python finetune.py \
  --run_name __RUN_NAME__ \
  --batch_size __BATCH_SIZE__ \
  --device cuda:0 \
  --num_epochs __EPOCHS__ \
  --lr_restart_epochs __RESTART_EPOCHS__ \
  --tag __TAG__ \
  --freeze_backbone \
  --fuse_embeddings \
  --modalities __MODALITIES__ \
  --finetune_sample_rate __SAMPLE_RATE__ \
  --fold_number __FOLD__ \
  --save_name __SAVE_NAME__ \
  --three_class \
  --adaptive
EOF

  sed -i \
    -e "s#__WORKDIR__#${WORKDIR}#g" \
    -e "s#__ENV_ACTIVATE__#${ENV_ACTIVATE}#g" \
    -e "s#__RUN_NAME__#${RUN_NAME}#g" \
    -e "s#__BATCH_SIZE__#${BATCH_SIZE}#g" \
    -e "s#__EPOCHS__#${EPOCHS}#g" \
    -e "s#__RESTART_EPOCHS__#${RESTART_EPOCHS}#g" \
    -e "s#__TAG__#${TAG}#g" \
    -e "s#__SAMPLE_RATE__#${SAMPLE_RATE}#g" \
    -e "s#__MODALITIES__#${MODALITIES}#g" \
    -e "s#__SAVE_NAME__#${SAVE_NAME}#g" \
    -e "s#__FOLD__#${FOLD}#g" \
    "${RUNNER}"
  chmod +x "${RUNNER}"

  # Start detached session; bind one physical GPU to that session
  ${TMUX_BIN} new-session -d -s "$SESS" "bash -lc 'export CUDA_VISIBLE_DEVICES=${GPU}; ${RUNNER} > ${LOGFILE} 2>&1'"
  sleep 0.2
  if ! ${TMUX_BIN} has-session -t "$SESS" 2>/dev/null; then
    echo "[WARN] Session ${SESS} exited immediately. Check log: ${LOGFILE}"
  fi
done

echo
echo "All launch attempts made. Current tmux sessions:"
${TMUX_BIN} ls || true

echo
echo "Attach:   tmux attach -t ${SESSION_PREFIX}_f2   \(example\)"
echo "Tail log: tail -f ${LOGDIR}/fold_2.log          \(example\)"
echo "Kill one: tmux kill-session -t ${SESSION_PREFIX}_f2"
echo "Kill all: tmux kill-server"
