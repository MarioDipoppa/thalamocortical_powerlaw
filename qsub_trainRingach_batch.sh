#!/bin/bash
#$ -N train-Ringach
#$ -cwd
#$ -V
#$ -l h_rt=08:00:00,h_vmem=64G
#$ -j y
#$ -o joblog/train_Ringach.$JOB_ID.$TASK_ID
#$ -M sakinkirti@g.ucla.edu
#$ -m ea

### Parallel Job Array
#$ -t 1-60

# Define the grid of parameters
LGN_VALUES=(32 64 128 256 512 1024)
V1_VALUES=(32 64 128 256 512 1024 2048 4096 8192 16384)

# optimize JAX on cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
DATA_PATH="/u/home/s/skirti/dipoppa-lab/dipoppa-lab/thalamocortical-expansion/01_data/natural_movies/IMG_3625_train_patches.npy"
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
