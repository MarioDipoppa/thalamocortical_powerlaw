#!/bin/bash
#$ -N train-Ringach
#$ -cwd
#$ -V
#$ -l gpu,RTX2080Ti,cuda=2,h_rt=12:00:00,h_vmem=32G,highp=TRUE
#$ -j y
#$ -o joblog/train_Ringach.$JOB_ID.$TASK_ID
#$ -M sakinkirti@g.ucla.edu
#$ -m bea

### Parallel Job Array
#$ -t 1-24
# 1-60

# load the right modules
module load gcc/11.3.0
module load cuda/12.3
module load cudnn/8.9.7

# Define the grid of parameters
LGN_VALUES=(256 512 1024) #(32 64 128 256 512 1024)
V1_VALUES=(64 256 512 1024 2048 4096 8192 16384) #(32 64 128 256 512 1024 2048 4096 8192 16384)

# optimize JAX on cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
DATA_PATH="/u/home/s/skirti/scratch/dipoppa-lab/thalamocortical-expansion/01_data/natural_movies/IMG_3625_train_patches.npy"
DATA_KEY="allPatches"
OUT_DIR="train_results_ringach_unconstrained"
BATCH_SIZE=48
EPOCHS=200
LR=0.0001
MARGIN=3.

mkdir -p $OUT_DIR
mkdir -p joblog

# Calculate indices for this specific task
num_v1=${#V1_VALUES[@]}
zero_indexed=$((SGE_TASK_ID - 1))
lgn_idx=$((zero_indexed / num_v1))
v1_idx=$((zero_indexed % num_v1))

lgn=${LGN_VALUES[$lgn_idx]}
v1=${V1_VALUES[$v1_idx]}

echo "Task $SGE_TASK_ID: Training LGN=$lgn, V1=$v1"

# Resume/Skip Logic
HISTORY_FILE="$OUT_DIR/train_history_LGN${lgn}_V1${v1}.pkl"
if [ -f "$HISTORY_FILE" ]; then
    echo "Task $SGE_TASK_ID (LGN=$lgn, V1=$v1) already has a history file. Skipping."
    exit 0
fi

# Check for early stopping in existing logs
# Note: This is an extra precaution for models that finished but history wasn't saved (unlikely but possible)
latest_log=$(ls -t joblog/train_Ringach.*.$SGE_TASK_ID 2>/dev/null | head -n 1)
if [ -n "$latest_log" ]; then
    if grep -q "Early stopping triggered." "$latest_log"; then
        echo "Task $SGE_TASK_ID (LGN=$lgn, V1=$v1) previously early-stopped according to log. Skipping."
        exit 0
    fi
fi

$PYTHON_EXE ringach_train.py \
    --lgn $lgn \
    --v1 $v1 \
    --data $DATA_PATH \
    --data-key $DATA_KEY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --margin $MARGIN \
    --out $OUT_DIR

echo "Task $SGE_TASK_ID training completed."
