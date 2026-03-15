#!/bin/bash
#$ -N un-m0-cpu
#$ -cwd
#$ -l h_rt=04:00:00,h_vmem=64G
#$ -j y
#$ -o joblog/un-m0-cpu.$JOB_ID.$TASK_ID
#$ -M sakinkirti@g.ucla.edu
#$ -m a

### Parallel Job Array
#$ -t 1-60

# optimize JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# load modules to make sure were running the right versions
module load gcc/11.3.0

# changes based on which model/margin value to test
PYTHON_EXE="/u/home/s/skirti/miniforge3/envs/tce_v2/bin/python"
INPUT_DATA="/u/home/s/skirti/scratch/dipoppa-lab/thalamocortical-expansion/01_data/natural_movies/IMG_3625_patches_64.npy"
PARAMS_DIR="/u/home/s/skirti/scratch/dipoppa-lab/thalamocortical-expansion/02_code/thalamocortical_powerlaw/train_unconstrained_margin0"
OUT_DIR="results_probabilistic_margin0_patch64"
BATCH_SIZE=48
MARGIN=0.
TEST_TRAINED=false # Set to false to test untrained models even if params exist

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

# Construct path to trained parameters
if [ "$TEST_TRAINED" = true ]; then
    PARAM_FILE="$PARAMS_DIR/best_params_LGN${lgn}_V1${v1}.pkl"
    if [ ! -f "$PARAM_FILE" ]; then
        echo "Warning: Parameter file $PARAM_FILE not found. Initializing with random/fixed weights."
        PARAM_ARG=""
    else
        echo "Using trained parameters from $PARAM_FILE"
        PARAM_ARG="--params $PARAM_FILE"
    fi
else
    echo "Trained testing disabled. Initializing with random/fixed weights."
    PARAM_ARG=""
fi

# some logging so i can debug just in case
echo "Task $SGE_TASK_ID: Testing LGN=$lgn, V1=$v1"
echo "Python:"
which python

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "JAX_PLATFORMS=$JAX_PLATFORMS"

$PYTHON_EXE - <<EOF
import jax
print("Devices:", jax.devices())
EOF

# run the actual code
$PYTHON_EXE ringach_test.py \
    --lgn $lgn \
    --v1 $v1 \
    --input $INPUT_DATA \
    --batch_size $BATCH_SIZE \
    --margin $MARGIN \
    --out $OUT_DIR \
    $PARAM_ARG

echo "Task $SGE_TASK_ID testing completed."
