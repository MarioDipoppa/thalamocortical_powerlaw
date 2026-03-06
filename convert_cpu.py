#!/usr/bin/env python
"""
Convert GPU pickles with JAX arrays to CPU-safe pickles using NumPy arrays.
Run this on a GPU-enabled machine.
"""

import glob
import pickle
import os
import jax
import jax.tree_util as jtu
import numpy as np
from tqdm import tqdm

# Path to your GPU pickles
INPUT_DIR = "/u/home/s/skirti/scratch/dipoppa-lab/thalamocortical-expansion/02_code/thalamocortical_powerlaw/train_unconstrained_margin3/"
OUTPUT_DIR = INPUT_DIR + "cpu_safe/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert(x):
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return x

# Use glob to find all train_history*.pkl files
pickle_files = glob.glob(os.path.join(INPUT_DIR, "train_history*.pkl"))
print(f"Found {len(pickle_files)} files")

for fpath in tqdm(pickle_files):
    
    # Load the pickle (safe on GPU machine)
    with open(fpath, "rb") as f:
        history = pickle.load(f)
    
    history = jtu.tree_map(convert, history)
    # Convert any JAX arrays to NumPy arrays
    # history_cpu = jax.tree_util.tree_map(
    #     lambda x: np.array(x) if isinstance(x, jax.Array) else x,
    #     history
    # )
    
    # Save to a new CPU-safe pickle
    fname = os.path.basename(fpath)
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, "wb") as f:
        pickle.dump(history, f)
    
