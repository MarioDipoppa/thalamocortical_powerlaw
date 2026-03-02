# thalamocortical_expansion
When light enters the eye, retinal ganglion cells (RGCs) are stimulated. They pass this information down the optic nerve to the lateral geniculate nucleus (LGN) in a near 1:1 neuron ratio. Following this, there is a large expansion in the number of neurons in the mammalian brain, where the few LGN neurons pass information to the primary visual cortex (V1) neurons at a ratio of 1:300 (in humans). The goal of this project is to explore the thalamocortical expansion and assess how this impacts processing in the brain.

## files
I've split the organization of these files into those used for the project and those files that are primarily for setup and information.

### project

#### models
- [`model.py`](model.py) contains the base RGC2LGN and LGN2V1 models and modified triplet loss.
- [`ringach_model.py`](ringach_model.py) implements the Ringach VVS model using JAX for differentiable forward passes.
- [`feedforward_model.py`](feedforward_model.py) implements a standard convolutional feedforward VVS model.
- [`slapik_model.py`](slapik_model.py) implements the early visual system model based on Slapik & Shouval (2026).

#### training & data
- [`ringach_train.py`](ringach_train.py) JAX-based training script for the Ringach model.
- [`feedforward_train.py`](feedforward_train.py) PyTorch-based training script for feedforward models.
- [`generate_triplets.py`](generate_triplets.py) script to generate anchor-positive-negative triplets from video data.
- [`ringach_test.py`](ringach_test.py) benchmarking and testing script for the Ringach model.
- [`params.yml`](params.yml) configuration parameters for training scripts.
- [`utils.py`](utils.py) shared custom utility functions.

#### cluster submission
- `qsub_trainRingach_batch.sh` / `qsub_testRingach_batch.sh` grid engine scripts for training/testing the Ringach model on the cluster.
- `qsub_trainTCE_batch.sh` / `qsub_trainTCE_sequential.sh` scripts for batch training of TCE models.

#### analysis & testing
- `slapik_analysis.ipynb` formal analysis of representations in the Slapik model.
- `testing.ipynb` interactive notebook for exploratory tests and verification.

### organization
- `environment.yml` conda environment specification for reproducibility.
- `README.md` __this__ file, containing project documentation.
- `TODO.md` checklist for tracking project progress.
- `.gitignore` git exclusion patterns.

## setup
1. you can install `miniconda` by following the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
2. set up a conda environment for this project. this will install the proper versions of python and all relevant packages needed.
    ```bash
    conda create -f environment.yml
    ```
    you can activate the environment before running code
    ```bash
    conda activate tce_env
    ```
    and deactivate the environment after completing
    ```bash
    conda deactivate
    ```