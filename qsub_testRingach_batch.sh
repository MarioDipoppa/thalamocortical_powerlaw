#!//bin/bash
#$ -N test-Ringach
#$ -cwd
#$ -V
#$ -l h_rt=08:00:00,h_vmem=64G
#$ -j y
#$ -o joblog/test_Ringach.$JOB_ID.$TASK_ID
#$ -M sakinkirti@g.ucla.edu
#$ -t 1-60

# optimize JAX on cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
INPUT_DATA="/u/home/s/skirti/dipoppa-lab/dipoppa-lab/thalamocortical-expansion/01_data/natural_movies/IMG_3625_test_patches.npy"
OUT_DIR="results_ringach_grid"
BATCH_SIZE=48
MARGIN=3.

# Define the grid of parameters
LGN_VALUES=(32 64 128 256 512 1024)
V1_VALUES=(32 64 128 256 512 1024 2048 4096 8192 16384)

mkdir -p $OUT_DIR
mkdir -p joblog

# Calculate indices for this specific task
num_v1=${#V1_VALUES[@]}
zero_indexed=$((SGE_TASK_ID - 1))
lgn_idx=$((zero_indexed / num_v1))
v1_idx=$((zero_indexed % num_v1))

lgn=${LGN_VALUES[$lgn_idx]}
v1=${V1_VALUES[$v1_idx]}

echo "Task $SGE_TASK_ID: Testing LGN=$lgn, V1=$v1"

$PYTHON_EXE ringach_test.py \
    --lgn $lgn \
    --v1 $v1 \
    --input $INPUT_DATA \
    --batch_size $BATCH_SIZE \
    --margin $MARGIN \
    --out $OUT_DIR

echo "Task $SGE_TASK_ID testing completed."
