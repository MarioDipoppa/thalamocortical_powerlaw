#!/bin/bash
#$ -N test-Ringach
#$ -cwd
#$ -V
#$ -l h_rt=04:00:00,h_data=32G
#$ -j y
#$ -o joblog/test_Ringach.$JOB_ID.$TASK_ID
#$ -M $USER@mail
#$ -t 1-5

PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
INPUT_DATA="/u/home/s/skirti/project-mdipoppa/thalamocortical-expansion/01_data/natural_movies/test.npy"
OUT_DIR="results_ringach_grid"
BATCH_SIZE=64

# Define the grid of parameters (5 values each)
LGN_VALUES=(100)
V1_VALUES=(200 400 600 800 1000)

mkdir -p $OUT_DIR
mkdir -p joblog

# Calculate indices for this specific task (1-based task ID)
# Task 1 -> LGN[0], V1[0]
# Task 5 -> LGN[0], V1[4]
# Task 6 -> LGN[1], V1[0]
zero_indexed=$((SGE_TASK_ID - 1))
lgn_idx=$((zero_indexed / 5))
v1_idx=$((zero_indexed % 5))

lgn=${LGN_VALUES[$lgn_idx]}
v1=${V1_VALUES[$v1_idx]}

echo "Task $SGE_TASK_ID: Running LGN=$lgn, V1=$v1"

$PYTHON_EXE test_ringach.py \
    --lgn $lgn \
    --v1 $v1 \
    --input $INPUT_DATA \
    --batch_size $BATCH_SIZE \
    --out $OUT_DIR

echo "Task $SGE_TASK_ID completed."
