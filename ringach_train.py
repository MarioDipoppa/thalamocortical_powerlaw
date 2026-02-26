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

def train_one_epoch(params, opt_state, train_indices, triplets, batch_size, train_step):
    """Performs a single epoch of training."""
    np.random.shuffle(train_indices)
    n_batches = 0
    epoch_loss = 0.0
    
    # Using tqdm for progress bar if stdout is a terminal
    for i in range(0, len(train_indices), batch_size):
        idx = train_indices[i:i+batch_size]
        batch = triplets[idx] # [B, 3, H, W]
        a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
        
        params, opt_state, loss_val = train_step(params, opt_state, a, p, n)
        epoch_loss += loss_val
        n_batches += 1
        
    return params, opt_state, epoch_loss / n_batches

def train_model(params, opt_state, triplets, train_indices, val_indices, args, train_step, val_step):
    """Master training function with validation and early stopping."""
    best_val_loss = float('inf')
    wait = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # 1. Train
        params, opt_state, train_loss = train_one_epoch(params, opt_state, train_indices, triplets, args.batch_size, train_step)
        
        # 2. Validate
        val_loss = 0.0
        n_val_batches = 0
        for i in range(0, len(val_indices), args.batch_size):
            idx = val_indices[i:i+args.batch_size]
            batch = triplets[idx]
            a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
            val_loss += val_step(params, a, p, n)
            n_val_batches += 1
        val_loss /= n_val_batches
        
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
    #mat = scipy.io.loadmat(args.data)
    #triplets = mat[args.data_key]
    #if triplets.ndim == 4 and triplets.shape[1] != 3:
    #     triplets = triplets.transpose(3, 2, 0, 1)
    
    triplets = np.load(args.data).astype(np.float32)
    triplets = (triplets - np.mean(triplets)) / (np.std(triplets) + 1e-8)
    
    n_triplets = triplets.shape[0]
    indices = np.random.permutation(n_triplets)
    train_size = int(0.9 * n_triplets)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 2. Initialize Model and Parameters
    shape = (224, 224)
    n_rgc_side = int(np.sqrt(args.lgn // 2.5))
    v1_side = int(np.sqrt(args.v1))
    print(f"Initializing model (RGC_grid={n_rgc_side}, V1_grid={v1_side})...")
    model = ringach_VVS(shape=shape, n_RGC=n_rgc_side, v1_dim=v1_side, eta_0=[3,8])
    
    # Use dense weights as parameters
    params = model.LGN_V1_conn
    
    # 3. Setup Optimizer
    optimizer = optax.adam(learning_rate=args.lr)
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
    train_model(params, opt_state, triplets, train_indices, val_indices, args, train_step, val_step)

if __name__ == "__main__":
    main()
