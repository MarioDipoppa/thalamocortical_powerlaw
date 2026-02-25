import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm import tqdm
from ringach_model import ringach_VVS

def main():
    parser = argparse.ArgumentParser(description="Test Ringach model with varying LGN and V1 neurons")
    parser.add_argument("--lgn", type=int, required=True, help="Number of LGN neurons (RGC is 0.4x)")
    parser.add_argument("--v1", type=int, required=True, help="Number of V1 neurons")
    parser.add_argument("--input", type=str, required=True, help="Path to input images .npy file (shape: [N, H, W])")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--out", type=str, default="results_ringach", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Configuration
    shape = (224, 224)
    print(f"Initializing model with LGN={int(args.lgn)} (RGC={args.lgn // 2.5}) and V1={args.v1}...")
    
    start_init = time.time()
    model = ringach_VVS(shape=shape, n_RGC=int(np.sqrt(args.lgn // 2.5)), v1_dim=int(np.sqrt(args.v1)))
    init_time = time.time() - start_init
    print(f"Initialization took {init_time:.2f}s")
    
    # Load input images and unroll the triplets into one large
    raw_images = np.load(args.input).reshape(-1, 224, 224)
    n_images = raw_images.shape[0]
    print(f"Loaded {n_images} images from {args.input}")
    
    # Benchmark JIT forward pass
    # Using vmap to handle the batch dimension
    jit_forward = jax.jit(jax.vmap(model.forward))
    
    print(f"Running forward pass on input images (batch_size={args.batch_size})...")
    start_run = time.time()
    
    all_r, all_l, all_v = [], [], []
    
    for i in tqdm(range(0, n_images, args.batch_size)):
        end = min(i + args.batch_size, n_images)
        batch = jnp.array(raw_images[i:end]).astype(jnp.float32)
        
        r_batch, l_batch, v_batch = jit_forward(batch)
        # Ensure completion for timing if it's the first or last batch
        if i == 0 or end == n_images:
            jax.block_until_ready((r_batch, l_batch, v_batch))
            
        all_r.append(np.array(r_batch))
        all_l.append(np.array(l_batch))
        all_v.append(np.array(v_batch))

    r_out = np.concatenate(all_r, axis=0)
    l_out = np.concatenate(all_l, axis=0)
    v_out = np.concatenate(all_v, axis=0)
    
    run_time = time.time() - start_run
    print(f"Forward pass took {run_time:.4f}s")
    
    # Save results
    res_path = os.path.join(args.out, f"res_LGN{args.lgn}_V1{args.v1}.pkl")
    results = {
        "lgn_params": args.lgn,
        "v1_params": args.v1,
        "init_time": init_time,
        "run_time": run_time,
        "rgc_shape": r_out.shape[1:],
        "lgn_shape": l_out.shape[1:],
        "v1_shape": v_out.shape[1:],
        "rgc_out": r_out,
        "lgn_out": l_out,
        "v1_out": v_out
    }
    
    with open(res_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()
