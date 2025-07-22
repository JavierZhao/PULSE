# ---------- user-editable settings ------------------------------------------
SESSION="eda_runs"                    # tmux session name
ENV="dl"                              # conda environment to activate
SCRIPT="python -u train_eda_diffusion.py"   # add -u for unbuffered output
# If Conda lives somewhere else, adjust the path below:
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

# ---------- argument list ----------------------------------------------------
ARGS=(
  "--run_name sp_0p5_lg_bn_1p   --dataset_percentage 0.01 --device cuda:1"
  "--run_name sp_0p5_lg_bn_10p  --dataset_percentage 0.1  --device cuda:2"
  "--run_name sp_0p5_lg_bn_100p --dataset_percentage 1    --device cuda:3"

  "--run_name lg_bn_1p          --dataset_percentage 0.01 --device cuda:4 --spec_loss_weight 0"
  "--run_name lg_bn_10p         --dataset_percentage 0.1  --device cuda:5 --spec_loss_weight 0"
  "--run_name lg_bn_100p        --dataset_percentage 1    --device cuda:6 --spec_loss_weight 0"
)

###############################################################################
# helper: open a new pane running “conda activate dl && python -u …”          #
###############################################################################
spawn_pane() {  # usage: spawn_pane <target> "<arg string>"
  local target="$1" args="$2"
  tmux split-window -t "$target" -v \
      "bash -ic 'source \"$CONDA_SH\" && conda activate \"$ENV\" && exec $SCRIPT $args'"
}

###############################################################################
# 1️⃣  create detached session with the first run -----------------------------
###############################################################################
tmux new-session -d -s "$SESSION" -n trainings \
    "bash -ic 'source \"$CONDA_SH\" && conda activate \"$ENV\" && exec $SCRIPT ${ARGS[0]}'"
tmux rename-pane -t "${SESSION}:0.0" run0

###############################################################################
# 2️⃣  add the remaining five panes ------------------------------------------
###############################################################################
for i in {1..5}; do
  spawn_pane "${SESSION}:0" "${ARGS[$i]}"
  tmux rename-pane -t "${SESSION}:0.$i" "run$i"
  tmux select-layout -t "${SESSION}:0" tiled   # keep a tidy 2×3 grid
done

# keep panes open when a run finishes (so you can read logs/errors) ----------
tmux set-option -t "${SESSION}:0" remain-on-exit on

###############################################################################
# 3️⃣  attach so you can monitor progress -------------------------------------
###############################################################################
tmux attach -t "$SESSION"
