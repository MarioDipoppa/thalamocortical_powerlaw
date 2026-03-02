import os
import time
import argparse
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import pickle
from tqdm import tqdm
from ringach_model import ringach_VVS
from utils import Utils

def main():
    parser = argparse.ArgumentParser(description="Test Ringach model with triplets and benchmarking")
    parser.add_argument("--lgn", type=int, required=True, help="Number of LGN neurons (RGC is 0.4x)")
    parser.add_argument("--v1", type=int, required=True, help="Number of V1 neurons")
    parser.add_argument("--input", type=str, required=True, help="Path to input triplets .npy file (shape: [N, 3, H, W])")
    parser.add_argument("--params", type=str, help="Path to trained parameters .pkl file")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing (number of triplets)")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for loss calculation")
    parser.add_argument("--l1_lambda", type=float, default=0.0, help="L1 lambda for loss calculation")
    parser.add_argument("--out", type=str, default="results_ringach", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Configuration
    shape = (224, 224)
    n_rgc_side = int(np.sqrt(args.lgn * 2 // 2.5))
    v1_side = int(np.sqrt(args.v1))
    print(f"Initializing model with LGN={args.lgn} (RGC_grid={n_rgc_side}) and V1_grid={v1_side}...")
    
    start_init = time.time()
    model = ringach_VVS(shape=shape, n_RGC=n_rgc_side, v1_dim=v1_side, eta_0=[3,8])
    init_time = time.time() - start_init
    print(f"Initialization took {init_time:.2f}s")
    
    # Use dense weights as parameters (default to initial if not provided)
    params = model.LGN_V1_conn
    if args.params:
        print(f"Loading trained parameters from {args.params}...")
        with open(args.params, "rb") as f:
            params = pickle.load(f)
            
    # Load input triplets using memory-mapping
    print(f"Loading data from {args.input}...")
    triplets = np.load(args.input, mmap_mode='r')
    if len(triplets.shape) == 3:
        print("Warning: Input shape is [N, H, W], expected [N, 3, H, W] for triplets.")
    
    n_triplets = triplets.shape[0]
    print(f"Loaded {n_triplets} triplets")
    
    # Benchmark JIT forward pass
    # Using vmap to handle the batch dimension, passing weights explicitly
    def model_v1_apply(batch, w):
        return vmap(lambda x, weights: model.forward(x, weights=weights)[2], in_axes=(0, None))(batch, w)
    
    jit_forward = jax.jit(vmap(lambda x, w: model.forward(x, weights=w), in_axes=(0, None)))
    
    print(f"Running forward pass on triplets (batch_size={args.batch_size})...")
    start_run = time.time()
    
    all_r, all_l, all_v = [], [], []
    
    n_batches = (n_triplets + args.batch_size - 1) // args.batch_size
    for i, batch in enumerate(tqdm(Utils.batch_generator(triplets, args.batch_size), total=n_batches)):
        # batch is [B, 3, H, W], flatten to [3B, H, W] for the model
        B = batch.shape[0]
        flattened_batch = batch.reshape(-1, 224, 224)
        
        r_batch, l_batch, v_batch = jit_forward(flattened_batch, params)
        
        # Ensure completion for timing if it's the first or last batch
        if i == 0 or i == n_batches - 1:
            jax.block_until_ready((r_batch, l_batch, v_batch))
            
        all_r.append(np.array(r_batch).reshape(B, 3, -1))
        all_l.append(np.array(l_batch).reshape(B, 3, -1))
        all_v.append(np.array(v_batch).reshape(B, 3, -1))

    r_out = np.concatenate(all_r, axis=0) # [N, 3, N_RGC]
    l_out = np.concatenate(all_l, axis=0) # [N, 3, N_LGN]
    v_out = np.concatenate(all_v, axis=0) # [N, 3, N_V1]
    
    run_time = time.time() - start_run
    print(f"Forward pass took {run_time:.4f}s")
    
    # --- Benchmarking ---
    print("Computing metrics...")
    # Gini Index for V1 activations (mean over all images and images in triplets)
    # Reshaping back to [3*N, N_V1] to get flat list of activations
    v1_acts_flat = v_out.reshape(-1, v_out.shape[-1])
    # Apply ReLU to treat inhibited neurons as inactive (0) for sparsity
    v1_acts_rectified = np.maximum(0, v1_acts_flat)
    v1_gini = Utils.gini(v1_acts_rectified.mean(axis=0))
    print(f"V1 Gini Index: {v1_gini:.4f}")
    
    # Triplet-based evaluation
    print("Computing triplet loss and violations...")
    # We can use the already computed activations to avoid another forward pass
    # But for consistency with the Utils implementation, let's use the generator if possible
    # Or just compute it directly here since we have the activations.
    
    a_out = v_out[:, 0]
    p_out = v_out[:, 1]
    n_out = v_out[:, 2]
    
    ap_dist = np.sum((a_out - p_out)**2, axis=1)
    an_dist = np.sum((a_out - n_out)**2, axis=1)
    
    test_loss = np.mean(np.maximum(ap_dist - an_dist + args.margin, 0))
    if args.l1_lambda > 0:
        l1_penalty = (np.mean(np.abs(a_out)) + np.mean(np.abs(p_out)) + np.mean(np.abs(n_out))) / 3.0
        test_loss += args.l1_lambda * l1_penalty
        
    test_viol = np.mean(ap_dist + args.margin > an_dist)
    print(f"Test Loss: {test_loss:.4f} - Test Violations: {test_viol*100:.2f}%")

    # Save results
    res_path = os.path.join(args.out, f"res_LGN{args.lgn}_V1{args.v1}.pkl")
    results = {
        "lgn_params": args.lgn,
        "v1_params": args.v1,
        "batch_size": args.batch_size,
        "margin": args.margin,
        "l1_lambda": args.l1_lambda,
        "init_time": init_time,
        "run_time": run_time,
        "n_triplets": n_triplets,
        "rgc_shape": r_out.shape[2:],
        "lgn_shape": l_out.shape[2:],
        "v1_shape": v_out.shape[2:],
        "v1_gini": v1_gini,
        "test_loss": test_loss,
        "test_viol": test_viol,
        "rgc_out": r_out,
        "lgn_out": l_out,
        "v1_out": v_out
    }
    
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()
