#!/bin/bash
#$ -N train-Ringach
#$ -cwd
#$ -V
#$ -l h_rt=08:00:00,h_data=32G
#$ -j y
#$ -o joblog/train_Ringach.$JOB_ID.$TASK_ID
#$ -M $USER@mail

### Parallel Job Array for 1 combination (LGN=100) and 5 V1 dimensions
#$ -t 1-5

# Load environment
. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate tce_v2

# Define the grid of parameters
LGN_VALUES=(100)
V1_VALUES=(200 400 600 800 1000)

PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
DATA_PATH="/u/home/s/skirti/project-mdipoppa/thalamocortical-expansion/01_data/natural_movies/IMG_3625_patches.npy"
DATA_KEY="allPatches"
OUT_DIR="train_results_ringach"
BATCH_SIZE=64
EPOCHS=200
LR=0.0001
MARGIN=3.

mkdir -p $OUT_DIR
mkdir -p joblog

# Calculate indices for this specific task
zero_indexed=$((SGE_TASK_ID - 1))
lgn_idx=$((zero_indexed / 5))
v1_idx=$((zero_indexed % 5))

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
