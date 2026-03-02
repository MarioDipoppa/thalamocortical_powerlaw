import os
import time
import argparse
import pickle
import yaml
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

import scipy.io
import traceback
from tqdm import tqdm
from ringach_model import ringach_VVS

# --- Loss Function ---
def loss_fn(weights, images_a, images_p, images_n, model_apply, margin, l1_lambda):
    a_out = model_apply(images_a, weights)
    p_out = model_apply(images_p, weights)
    n_out = model_apply(images_n, weights)
    
    # Triplet Loss
    ap_dist = jnp.sum((a_out - p_out)**2, axis=1)
    an_dist = jnp.sum((a_out - n_out)**2, axis=1)
    triplet_loss = jnp.mean(jax.nn.relu(ap_dist - an_dist + margin))
    
    # L1 Penalty
    if l1_lambda > 0:
        l1_penalty = (jnp.mean(jnp.abs(a_out)) + jnp.mean(jnp.abs(p_out)) + jnp.mean(jnp.abs(n_out))) / 3.0
        return triplet_loss + l1_lambda * l1_penalty
    return triplet_loss

def train_one_epoch(params, opt_state, train_generator, train_step, epoch_num):
    """Performs a single epoch of training."""
    n_batches = 0
    epoch_loss = 0.0
    
    # Using tqdm for progress bar if stdout is a terminal
    for batch in train_generator:
        a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
        
        params, opt_state, loss_val = train_step(params, opt_state, a, p, n)
        
        # Diagnostic: Numerical check
        if jnp.isnan(loss_val) or jnp.isinf(loss_val):
            print(f"NAN/INF detected in Loss at Epoch {epoch_num}, Batch {n_batches}!")
            raise ValueError(f"Numerical instability: loss is {loss_val}")

        epoch_loss += loss_val
        n_batches += 1
        
        # Diagnostic: Frequent logging
        # if n_batches % 5 == 0:
        #      print(f"  [Epoch {epoch_num}] Batch {n_batches}/{len(train_indices)//batch_size} - Current Loss: {loss_val:.4f}")
        
    return params, opt_state, epoch_loss / n_batches

def train_model(params, opt_state, args, train_step, val_step,
                batch_generator, triplets, train_start, train_end, val_start, val_end, batch_size):
    """Master training function with validation and early stopping."""
    best_val_loss = float('inf')
    wait = 0
    start_time = time.time()

    # init some storage
    train_losses = []
    val_losses = []
    train_viols = []
    val_viols = []
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # create our generators for training and validation
        train_generator = batch_generator(triplets, train_start, train_end, batch_size)
        val_generator = batch_generator(triplets, val_start, val_end, batch_size)
        
        # 1. Train
        params, opt_state, train_loss = train_one_epoch(params, opt_state, train_generator, train_step, epoch + 1)
        
        # 2. Validate
        val_loss = 0.0
        n_val_batches = 0
        for val_batch in val_generator:
            a, p, n = val_batch[:, 0], val_batch[:, 1], val_batch[:, 2]
            val_loss += val_step(params, a, p, n)
            n_val_batches += 1
        val_loss /= n_val_batches
        
	# store the train/val losses
	train_losses.append(train_loss)
	val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - time: {time.time()-t0:.2f}s")
        
        # 3. Checkpoint/Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best params
            save_path = os.path.join(args.out, f"best_params_LGN{args.lgn}_V1{args.v1}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(params, f)
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break
        
        # Periodic Checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(args.out, f"checkpoint_epoch_{epoch+1}.pkl")
            with open(periodic_path, "wb") as f:
                pickle.dump(params, f)
            print(f"Periodic checkpoint saved to {periodic_path}")
                
    print(f"Training finished in {time.time() - start_time:.2f}s")
    return params

def main():
    parser = argparse.ArgumentParser(description="Train Ringach VVS model with JAX")
    parser.add_argument("--lgn", type=int, required=True, help="Total LGN neurons")
    parser.add_argument("--v1", type=int, required=True, help="Total V1 neurons")
    parser.add_argument("--data", type=str, required=True, help="Path to triplets .mat file")
    parser.add_argument("--data-key", type=str, required=True, help="Key for data in .mat file")
    parser.add_argument("--out", type=str, default="train_results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # 1. Load and Prepare Data
    print(f"Loading data from {args.data}...")
    triplets = np.load(args.data, mmap_mode='r')  # memory-map instead of loading all at once
    n_triplets = triplets.shape[0]
    train_size = int(0.9 * n_triplets)
    print(f"loading {train_size} triplets with batch size {args.batch_size} for training")
    print(f"loading {n_triplets - train_size} triplets for validation")
    
    # Batch generator for sequential batches
    def batch_generator(arr, start_idx, end_idx, batch_size):
        for i in range(start_idx, end_idx, batch_size):
            batch = arr[i:i+batch_size].astype(np.float32)
            yield batch

    # 2. Initialize Model and Parameters
    shape = (224, 224)
    n_rgc_side = int(np.sqrt(args.lgn // 2.5))
    v1_side = int(np.sqrt(args.v1))
    print(f"Initializing model (RGC_grid={n_rgc_side}, V1_grid={v1_side})...")
    model = ringach_VVS(shape=shape, n_RGC=n_rgc_side, v1_dim=v1_side, eta_0=[3,8])
    
    # Use dense weights as parameters
    params = model.LGN_V1_conn
    
    # 3. Setup Optimizer with Gradient Clipping
    # Clipping prevents exploding gradients that can sometimes crash the XLA executor silently.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=args.lr)
    )
    opt_state = optimizer.init(params)
    
    # 4. Define Differentiable Steps
    model_v1_apply = vmap(lambda x, w: model.forward(x, weights=w)[2], in_axes=(0, None))
    
    @jit
    def train_step(params, opt_state, batch_a, batch_p, batch_n):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, batch_a, batch_p, batch_n, model_v1_apply, args.margin, args.l1_lambda)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    @jit
    def val_step(params, batch_a, batch_p, batch_n):
        return loss_fn(params, batch_a, batch_p, batch_n, model_v1_apply, args.margin, args.l1_lambda)

    # 5. Run Training
    print("Starting training loop...")
    try:
        train_model(params, opt_state, args, train_step, val_step,
                    batch_generator, triplets, 0, train_size, train_size, n_triplets, args.batch_size)
    except Exception as e:
        print("\n!!! TRAINING CRASHED !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        # Save emergency checkpoint
        save_path = os.path.join(args.out, f"emergency_checkpoint_LGN{args.lgn}_V1{args.v1}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Emergency checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
