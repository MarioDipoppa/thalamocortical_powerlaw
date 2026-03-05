import os
import time
import argparse
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import optax

import traceback
from tqdm import tqdm
from ringach_model import ringach_VVS
from utils import Utils

# Set random seed for reproducibility in model architecture
np.random.seed(42)

def pad_to_multiple(x, multiple, axis=0):
    """Pads an array along a specific axis to be a multiple of 'multiple'."""
    size = x.shape[axis]
    if size % multiple == 0:
        return x
    pad_size = multiple - (size % multiple)
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_size)
    return jnp.pad(x, pad_width, mode='constant', constant_values=0)

def shard_pytree(tree, sharding, replicated_sharding):
    """Shards a pytree, applying 'sharding' to arrays and 'replicated_sharding' to scalars."""
    def _shard_leaf(leaf):
        if not hasattr(leaf, 'shape') or len(leaf.shape) < 2:
            return jax.device_put(leaf, replicated_sharding)
        return jax.device_put(leaf, sharding)
    return jax.tree_map(_shard_leaf, tree)

def train_one_epoch(params, opt_state, train_generator, train_step, epoch_num):
    """Performs a single epoch of training."""
    n_batches = 0
    epoch_loss = 0.0
    
    # Using tqdm for progress bar if stdout is a terminal
    for batch in train_generator:
        # Move batch explicitly to device (replicated across GPUs for model sharding)
        # We use jax.device_put with a replicated sharding or None (default is reasonable)
        # But to be explicit for model sharding:
        batch_dev = jax.device_put(batch, train_generator.sharding if hasattr(train_generator, 'sharding') else None)
        a, p, n = batch_dev[:, 0], batch_dev[:, 1], batch_dev[:, 2]
        
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

def train_model(params, opt_state, args, train_step, val_step, model_v1_apply,
                batch_generator, triplets, train_start, train_end, val_start, val_end, batch_size,
                start_epoch=0, best_val_loss=float('inf'), wait=0,
                train_losses=None, val_losses=None, train_viols=None, val_viols=None,
                start_time_offset=0, sharding=None):
    """Master training function with validation and early stopping."""
    if train_losses is None: train_losses = []
    if val_losses is None: val_losses = []
    if train_viols is None: train_viols = []
    if val_viols is None: val_viols = []
    
    start_time = time.time() - start_time_offset
    
    for epoch in tqdm(range(start_epoch, args.epochs)):
        t0 = time.time()
        
        # create our generators for training and validation
        train_generator = batch_generator(triplets, batch_size, train_start, train_end)
        val_generator = batch_generator(triplets, batch_size, val_start, val_end)
        
        # 1. Train
        params, opt_state, train_loss = train_one_epoch(params, opt_state, train_generator, train_step, epoch + 1)
        
        # 2. Validate
        # Ensure validation generator also uses replicated sharding
        val_loss = Utils.evaluate_loss_jax(params, val_generator, model_v1_apply, args.margin, args.l1_lambda, sharding=sharding)
        
        # 3. Benchmark violations
        # Re-fetch generators for stats
        train_gen_stats = batch_generator(triplets, batch_size, train_start, train_end)
        val_gen_stats = batch_generator(triplets, batch_size, val_start, val_end)
        
        _, _, train_viol = Utils.compute_triplet_margin_stats_jax(params, train_gen_stats, model_v1_apply, args.margin, sharding=sharding)
        _, _, val_viol = Utils.compute_triplet_margin_stats_jax(params, val_gen_stats, model_v1_apply, args.margin, sharding=sharding)
        
        # store the train/val losses and violations
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_viols.append(train_viol)
        val_viols.append(val_viol)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Viol: {train_viol*100:.2f}% - Val Viol: {val_viol*100:.2f}% - time: {time.time()-t0:.2f}s")
        
        # 3. Checkpoint/Early Stopping
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best params (slice back to original size for the user)
            save_path = os.path.join(args.out, f"best_params_LGN{args.lgn}_V1{args.v1}.pkl")
            with open(save_path, "wb") as f:
                # Use a slice to remove padding before saving
                original_v1 = params.shape[0] if not hasattr(args, 'original_v1') else args.original_v1
                original_lgn = params.shape[1] if not hasattr(args, 'original_lgn') else args.original_lgn
                pickle.dump(np.array(params[:original_v1, :original_lgn]), f)
            wait = 0
            is_best = True
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                # Save final history immediately on early stop
                save_history(args, train_losses, val_losses, train_viols, val_viols, 
                             time.time() - start_time, best_val_loss, early_stopped=True)
                return params
        
        # 4. State Checkpointing (every epoch for robust resume)
        state = {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "wait": wait,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_viols": train_viols,
            "val_viols": val_viols,
            "start_time_offset": time.time() - start_time
        }
        checkpoint_path = os.path.join(args.out, f"state_checkpoint_LGN{args.lgn}_V1{args.v1}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
                
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    
    save_history(args, train_losses, val_losses, train_viols, val_viols, 
                 total_time, best_val_loss, early_stopped=False)
    
    # Remove checkpoint on successful completion
    checkpoint_path = os.path.join(args.out, f"state_checkpoint_LGN{args.lgn}_V1{args.v1}.pkl")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    return params

def save_history(args, train_losses, val_losses, train_viols, val_viols, total_time, best_val_loss, early_stopped=False):
    """Saves training history and metadata."""
    history_path = os.path.join(args.out, f"train_history_LGN{args.lgn}_V1{args.v1}.pkl")
    metadata = {
        "lgn": args.lgn,
        "v1": args.v1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "margin": args.margin,
        "l1_lambda": args.l1_lambda,
        "patience": args.patience,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_viols": train_viols,
        "val_viols": val_viols,
        "total_time": total_time,
        "best_val_loss": best_val_loss,
        "early_stopped": early_stopped
    }
    with open(history_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Training history saved to {history_path}")

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

    # 2. Initialize Model and Parameters
    shape = (224, 224)
    n_rgc_side = int(np.sqrt(args.lgn // 2.5))
    v1_side = int(np.sqrt(args.v1))
    print(f"Initializing model (RGC_grid={n_rgc_side}, V1_grid={v1_side})...")
    model = ringach_VVS(shape=shape, n_RGC=n_rgc_side, v1_dim=v1_side, eta_0=[3,8])
    
    # Use dense weights as parameters
    params = model.LGN_V1_conn
    
    # 3. Setup Sharding Mesh
    devices = jax.devices()
    print(f"Detected {len(devices)} devices: {devices}")
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=('model',))
    
    # Define Shardings
    # Shard model weights and activations along the neuron/filter axis
    neuron_sharding = NamedSharding(mesh, P('model', None))
    filter_sharding = NamedSharding(mesh, P('model', None, None))
    replicated_sharding = NamedSharding(mesh, P())
    
    # Pad components to be divisible by the number of devices (model sharding requirement)
    n_devices = len(devices)
    original_rgc_size = model.RGC_activations.shape[0]
    model.RGC_activations = pad_to_multiple(model.RGC_activations, n_devices, axis=0)
    model.RGC_on_mask = pad_to_multiple(model.RGC_on_mask, n_devices, axis=0)
    # Note: LGN_RGC_idx_jnp doesn't need padding as it's replicated and only used for indexing
    
    # Store the original dimensions for results (excluding padding)
    original_v1_size = params.shape[0]
    original_lgn_size = params.shape[1]
    
    # Pad V1 dimension (rows)
    params = pad_to_multiple(params, n_devices, axis=0)
    # Pad LGN dimension (columns) if RGC was padded
    # (The first part of LGN is RGC_act, which is size n_RGC_padded)
    if model.RGC_activations.shape[0] != original_rgc_size:
         # Pad axis 1 to match what enters v1_forward
         r_dummy = jnp.zeros(model.RGC_activations.shape[0])
         l_dummy = model.lgn_forward(r_dummy)
         if params.shape[1] != len(l_dummy):
             params = pad_to_multiple(params, len(l_dummy), axis=1)

    # Add original sizes to args so train_model can slice back
    args.original_v1 = original_v1_size
    args.original_lgn = original_lgn_size
    
    # Shard RGC components
    model.RGC_activations = jax.device_put(model.RGC_activations, filter_sharding)
    model.RGC_on_mask = jax.device_put(model.RGC_on_mask, filter_sharding)
    model.LGN_RGC_idx_jnp = jax.device_put(model.LGN_RGC_idx_jnp, replicated_sharding) # Replicated indexing
    
    # 4. Setup Optimizer with Gradient Clipping
    # Clipping prevents exploding gradients that can sometimes crash the XLA executor silently.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=args.lr)
    )
    
    # Initialize parameters and optimizer state
    # Shard weights along the V1 neuron dimension (axis 0)
    opt_state = optimizer.init(params)
    params = shard_pytree(params, neuron_sharding, replicated_sharding)
    opt_state = shard_pytree(opt_state, neuron_sharding, replicated_sharding)
    
    # Attach sharding info to generators for convenience
    setattr(Utils.batch_generator, 'sharding', replicated_sharding)

    # 5. Define Differentiable Steps
    model_v1_apply = vmap(lambda x, w: model.forward(x, weights=w)[2], in_axes=(0, None))
    
    @jit
    def train_step(params, opt_state, batch_a, batch_p, batch_n):
        loss_val, grads = jax.value_and_grad(Utils.jax_loss_fn)(params, batch_a, batch_p, batch_n, model_v1_apply, args.margin, args.l1_lambda)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    @jit
    def val_step(params, batch_a, batch_p, batch_n):
        return Utils.jax_loss_fn(params, batch_a, batch_p, batch_n, model_v1_apply, args.margin, args.l1_lambda)

    # 5. Resume/Skip Logic
    history_path = os.path.join(args.out, f"train_history_LGN{args.lgn}_V1{args.v1}.pkl")
    checkpoint_path = os.path.join(args.out, f"state_checkpoint_LGN{args.lgn}_V1{args.v1}.pkl")
    
    start_epoch = 0
    best_val_loss = float('inf')
    wait = 0
    train_losses, val_losses, train_viols, val_viols = [], [], [], []
    start_time_offset = 0
    
    # A. Skip if already finished
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            hist = pickle.load(f)
        if hist.get("early_stopped", False) or len(hist.get("train_losses", [])) >= args.epochs:
            print(f"Task LGN{args.lgn}_V1{args.v1} is already finished. Skipping.")
            return

    # B. Resume from state checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint for LGN{args.lgn}_V1{args.v1}. Resuming...")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        params = state["params"]
        opt_state = state["opt_state"]
        start_epoch = state["epoch"]
        best_val_loss = state.get("best_val_loss", float('inf'))
        wait = state.get("wait", 0)
        train_losses = state.get("train_losses", [])
        val_losses = state.get("val_losses", [])
        train_viols = state.get("train_viols", [])
        val_viols = state.get("val_viols", [])
        start_time_offset = state.get("start_time_offset", 0)
        print(f"Resuming from Epoch {start_epoch}")
    
    # C. Bootstrap from logs if no checkpoint but log exists (for previously timed-out runs)
    else:
        # We try to find a log file in the joblog directory. 
        # Since we don't know the exact JOB_ID here, we might need to search or use a pattern.
        # For simplicity, we can look for "train_Ringach.*.<TASK_ID>" where TASK_ID is mapped from args.
        # However, the user said we can check the stdout. 
        # If we are running in a script that has access to the task environment, we might find it.
        # Let's assume we can search the 'joblog' directory if it exists.
        log_dir = "joblog"
        if os.path.exists(log_dir):
            import re
            # Find the most recent log for this parameter combination
            # This is tricky because logs are named by JOB_ID and TASK_ID.
            # But we can grep for "Training LGN=<lgn>, V1=<v1>"
            best_log = None
            for f_name in os.listdir(log_dir):
                full_log_path = os.path.join(log_dir, f_name)
                try:
                    with open(full_log_path, "r") as f:
                        header = f.readline()
                        if f"Training LGN={args.lgn}, V1={args.v1}" in header or f"Training LGN={args.lgn}, V1={args.v1}" in f.read(500):
                            best_log = full_log_path
                            # We don't break, we want the most recent one if multiple exist
                except:
                    continue
            
            if best_log:
                print(f"Found previous log: {best_log}. Attempting bootstrap...")
                try:
                    with open(best_log, "r") as f:
                        content = f.read()
                    
                    if "Early stopping triggered." in content:
                        print("Log indicates early stopping was already triggered. Saving history and exiting.")
                        # Parse history from log if possible, or just exit. 
                        # To be safe, we'll just exit and let the next run see the history or re-train if history is missing.
                        return
                    
                    # Regex to find: Epoch 10/200 - Train Loss: 0.1234 - Val Loss: 0.1234 ...
                    pattern = r"Epoch (\d+)/\d+ - Train Loss: ([\d\.]+) - Val Loss: ([\d\.]+) - Train Viol: ([\d\.]+)% - Val Viol: ([\d\.]+)%"
                    matches = re.findall(pattern, content)
                    if matches:
                        last_epoch = int(matches[-1][0])
                        print(f"Bootstrapping from log: detected {last_epoch} completed epochs.")
                        # We can't recover params or opt_state from the log, 
                        # but we can recover the history lists and best_val_loss
                        # and continue from the LAST SAVED BEST PARAMS if they exist.
                        train_losses = [float(m[1]) for m in matches]
                        val_losses = [float(m[2]) for m in matches]
                        train_viols = [float(m[3])/100.0 for m in matches]
                        val_viols = [float(m[4])/100.0 for m in matches]
                        best_val_loss = min(val_losses)
                        
                        best_params_path = os.path.join(args.out, f"best_params_LGN{args.lgn}_V1{args.v1}.pkl")
                        if os.path.exists(best_params_path):
                            print(f"Loading best params from {best_params_path}")
                            with open(best_params_path, "rb") as f:
                                params = pickle.load(f)
                            opt_state = optimizer.init(params) # Note: we lose opt_state history, but it's better than nothing
                            start_epoch = last_epoch
                        else:
                            print("No best params found to bootstrap from. Starting from scratch.")
                except Exception as e:
                    print(f"Bootstrap failed: {e}")

    # Shard parameters and optimizer state (after loading/initializing)
    params = shard_pytree(params, neuron_sharding, replicated_sharding)
    opt_state = shard_pytree(opt_state, neuron_sharding, replicated_sharding)

    # 6. Run Training
    print("Starting training loop...")
    try:
        # Note: we need to pass the resumed state variables to train_model
        # I'll modify train_model signature or just pass them in args/kwargs
        # Actually, let's just modify the train_model call to handle the start_epoch
        train_model(params, opt_state, args, train_step, val_step, model_v1_apply,
                    Utils.batch_generator, triplets, 0, train_size, train_size, n_triplets, args.batch_size,
                    start_epoch=start_epoch, best_val_loss=best_val_loss, wait=wait,
                    train_losses=train_losses, val_losses=val_losses, train_viols=train_viols, val_viols=val_viols,
                    start_time_offset=start_time_offset, sharding=replicated_sharding)
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
