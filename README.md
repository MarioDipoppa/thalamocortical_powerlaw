# thalamocortical_expansion
When light enters the eye, retinal ganglion cells (RGCs) are stimulated. They pass this information down the optic nerve to the lateral geniculate nucleus (LGN) in a near 1:1 neuron ratio. Following this, there is a large expansion in the number of neurons in the mammalian brain, where the few LGN neurons pass information to the primary visual cortex (V1) neurons at a ratio of 1:300 (in humans). The goal of this project is to explore the thalamocortical expansion and assess how this impacts processing in the brain.

## files
I've split the organization of these files into those used for the project and those files that are primarily for setup and information.

### project
- `model.py` contains code for the RGC2LGN and LGN2V1 models, along with modified triplet loss. These files are the essential components of the system
- `train.py` contains training loops for training the relevant networks
- `params.yml` contains the parameters used as input to run the `train.py` script
- `utils.py` contains a few custom functions that are used in multiple places across training and analysis, so they are implemented once for consistency
- `testing.ipynb` is a python notebook that contains small tests that I did to ensure things worked as I expected
- `analysis.ipynb` contains the formal analysis of the produced networks

### organization
- `environment.yml` contains all packages required to reproduce results for this project
- `README.md` __this__ file, contains essential information regarding the project and important documentation
- `TODO.md` a todo list that I'm using to keep track of progress
- `.gitignore` keeps track of files/folders to not track using git

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