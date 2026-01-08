# thalamocortical_expansion
When light enters the eye, retinal ganglion cells (RGCs) are stimulated. They pass this information down the optic nerve to the lateral geniculate nucleus (LGN) in a near 1:1 neuron ratio. Following this, there is a large expansion in the number of neurons in the mammalian brain, where the few LGN neurons pass information to the primary visual cortex (V1) neurons at a ratio of 1:300 (in humans). The goal of this project is to explore the thalamocortical expansion and assess how this impacts processing in the brain.

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
3. 

## files
- `model.py` contains code for the RGC2LGN and LGN2V1 models, along with modified triplet loss. These files are the essential components of the system.
- `testing.ipynb` is a python notebook that contains small tests that I did to ensure things worked as I expected.