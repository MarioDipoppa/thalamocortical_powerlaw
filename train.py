import os
import yaml
import argparse
import pickle
import time
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from utils import Utils as u
from model import RGC2LGN, LGN2V1, ModifiedTripletLoss as MTL

def train(model, train_loader, val_loader, optimizer, criterion, device, data_name,
          v1_neurons, out, epochs=10, patience=5, min_delta=1e-4, l1_lambda=0.0,
          margin=0.2):

    model.to(device)
    
    train_losses, val_losses = [], []
    train_triplet_losses, val_triplet_losses = [], []
    train_l1_norms, val_l1_norms = [], []
    train_violations, val_violations = [], []
    
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0
    start_time = time.time()

    # ---- Initial evaluation before training ----
    train_loss, train_triplet_loss, train_l1_norm = u.evaluate_loss(model, train_loader,
                                                            criterion, device)
    train_losses.append(train_loss)
    train_triplet_losses.append(train_triplet_loss)
    train_l1_norms.append(train_l1_norm)
    
    _, _, v_train = u.compute_triplet_margin_stats(model, train_loader, device)
    train_violations.append(v_train)

    val_loss, val_triplet_loss, val_l1_norm = u.evaluate_loss(model, val_loader,
                                                            criterion, device)
    val_losses.append(val_loss)
    val_triplet_losses.append(val_triplet_loss)
    val_l1_norms.append(val_l1_norm)
    
    _, _, v_val = u.compute_triplet_margin_stats(model, val_loader, device)
    val_violations.append(v_val)

    best_val_loss = val_losses[-1]
    best_state = model.state_dict()

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_triplet = 0.0
        total_l1 = 0.0
        
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            a_out, p_out, n_out = model(a), model(p), model(n)
            triplet_loss = criterion(a_out, p_out, n_out)

            l1_penalty = 0.0
            if l1_lambda > 0:
                l1_penalty = (
                    a_out.abs().sum() +
                    p_out.abs().sum() +
                    n_out.abs().sum()
                ) / a_out.shape[0]

            loss = triplet_loss + l1_lambda * l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_triplet += triplet_loss.item()
            total_l1 += l1_penalty
            
        train_losses.append(total_loss / len(train_loader))
        train_triplet_losses.append(total_triplet / len(train_loader))
        train_l1_norms.append(total_l1 / len(train_loader))
        
        u.smooth_weights(model, sigma=0.25)
        
        # compute violations
        _, _, v_train = u.compute_triplet_margin_stats(model, train_loader, device)
        train_violations.append(v_train)

        val_loss, val_triplet_loss, val_l1_norm = u.evaluate_loss(model, val_loader, criterion, device, l1_lambda)
        val_losses.append(val_loss)
        val_triplet_losses.append(val_triplet_loss)
        val_l1_norms.append(val_l1_norm)
        _, _, v_val = u.compute_triplet_margin_stats(model, val_loader, device)
        val_violations.append(v_val)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break

        print('.', end='', flush=True)
        if (epoch + 1) % 10 == 0:
            print(f' Epoch {epoch + 1}')

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    model.load_state_dict(best_state)
        
    # Save model state separately
    #lambda_str = "0" if l1_lambda == 0 else f"{l1_lambda:.0e}".replace('-', 'm')
    margin_str = f"{margin:.1f}".replace('.', 'p')
    model_path = f"{out}/class_{data_name}_weights_M{margin_str}_V1{v1_neurons}_EP{epochs}.pt"
    torch.save(model.state_dict(), model_path)

    # Save metadata separately
    meta_path = f"{out}/class_{data_name}_meta_M{margin_str}_V1{v1_neurons}_EP{epochs}.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_triplet_losses": train_triplet_losses,
            "val_triplet_losses": val_triplet_losses,
            "train_l1_norms": train_l1_norms,
            "val_l1_norms": val_l1_norms,
            "train_viol": train_violations,
            "val_viol": val_violations,
            "train_time": total_time,
            "best_epoch": best_epoch
        }, f)

    return model

def hyperparameter_search(hyper_params:list, train_data:torch.utils.data.Dataset, val_data:torch.utils.data.Dataset, data_name:str, out:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Iterate over all combinations ---
    n_models = len(hyper_params)
    for model_id in range(n_models):
        learning_rate = hyper_params[model_id]["learning_rate"]
        wd = hyper_params[model_id]["weight_decay"]
        bs = hyper_params[model_id]["batch_size"]
        lgn = hyper_params[model_id]["lgn_neurons"]
        v1 = hyper_params[model_id]["v1_neurons"]
        ll = hyper_params[model_id]["l1_lambda"]
        epochs = hyper_params[model_id]["epochs"]
        margin = hyper_params[model_id]["margin"]
        patience = hyper_params[model_id]["patience"]
        min_delta = hyper_params[model_id]["min_delta"]
        
        print(f"\nTraining {model_id+1}/{n_models} with M={margin}, V1={v1}, WD={wd}, BS={bs}")
        
        # Dataloaders
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs)

        # we use our custom model and loss
        model = LGN2V1(input_dim=lgn, output_dim=v1)
        criterion = MTL(margin=margin)
        
        # use Adam (can optimize later)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

        model = train(model, train_loader, val_loader,
                      optimizer, criterion, device,
                      data_name, out=out, v1_neurons=v1, epochs=epochs, patience=patience,
                      min_delta=min_delta, l1_lambda=ll, margin=margin)

    print(f"Grid completed")

class TripletDataset(Dataset):
    """started from Mario's previous code, added labels"""
    
    def __init__(self, triplet_tensor):
        self.triplets = triplet_tensor  # could be a Tensor or a Subset

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        
        # get the correct triplet
        triplet = self.triplets[idx]
        anchor = triplet[0].unsqueeze(0).flatten()  # for now we flatten (later we can use the multiscale gaussians)
        positive = triplet[1].unsqueeze(0).flatten()
        negative = triplet[2].unsqueeze(0).flatten()
        
        return anchor, positive, negative

def main():
    
    # parse args
    parser = argparse.ArgumentParser(description="Train LGN2V1 networks to computationally model thalamocortical expansion")
    parser.add_argument("--params", help="parameters file with details for training", default="params.yml")
    parser.add_argument("--data", help=".mat file where data is stored for training")
    parser.add_argument("--out", help="path where output models should be stored post-training")
    parser.add_argument("--seed", help="random seed for consistency", default=1234)
    args = parser.parse_args()
    
    # read the params from yml
    with open(args.params) as f:
        params = yaml.safe_load(f)
    assert "epochs" in params, f"Key 'epochs' not found in {args.params.split('/')[-1]}"
    assert "patience" in params, f"Key 'patience' not found in {args.params.split('/')[-1]}"
    assert "min_delta" in params, f"Key 'min_delta' not found in {args.params.split('/')[-1]}"
    assert "LGN_neurons" in params, f"Key 'LGN_neurons' not found in {args.params.split('/')[-1]}"
    assert "V1_neurons" in params, f"Key 'V1_neurons' not found in {args.params.split('/')[-1]}"
    assert "l1_lambdas" in params, f"Key 'l1_lambdas' not found in {args.params.split('/')[-1]}"
    assert "margins" in params, f"Key 'margins' not found in {args.params.split('/')[-1]}"
    assert "learning_rates" in params, f"Key 'learning_rates' not found in {args.params.split('/')[-1]}"
    assert "weight_decays" in params, f"Key 'weight_decays' not found in {args.params.split('/')[-1]}"
    assert "batch_sizes" in params, f"Key 'batch_sizes' not found in {args.params.split('/')[-1]}"
    print("parameters and arguments loaded properly...")
        
    # let's load the training data to a train/val/test torch datasets/loaders
    np.random.seed(1234)
    triplets, labels = u.load_mat(args.data, mat_key="triplets")
    indices = np.random.permutation(len(triplets))
    train_size, val_size = int(0.7 * len(triplets)), int(0.1 * len(triplets))
    train_data = TripletDataset(triplets[indices[:train_size]])
    val_data = TripletDataset(triplets[indices[train_size:train_size + val_size]])
    test_data = TripletDataset(triplets[indices[train_size + val_size:]])
    print("training datasets properly loaded...")
    
    # now, some housekeeping for organization before we start training
    hyperparams = []
    os.makedirs(args.out, exist_ok=True)
    for i, (lr, wd, bs, lgn, v1, ll, mm) in enumerate(itertools.product(params["learning_rates"], params["weight_decays"], params["batch_sizes"], params["LGN_neurons"], params["V1_neurons"], params["l1_lambdas"], params["margins"])):
        # cache the hyperparameters for each model
        hyperparam_instance = {
            "model id": i,
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "lgn_neurons": lgn,
            "v1_neurons": v1,
            "l1_lambda": ll,
            "margin": mm,
            "epochs": params["epochs"],
            "patience": params["patience"],
            "min_delta": params["min_delta"],
        }
        hyperparams.append(hyperparam_instance)
        
        # create storage for weights with identifiable names + hyperparameters
        margin_str = f"{mm:.1f}".replace('.', 'p')
        hyper_path = f"{args.out}/class_{args.data.split('/')[-1].split('.')[0]}_hyper_M{margin_str}_V1{v1}_EP{params['epochs']}.pkl"
        with open(hyper_path, 'wb') as f:
            pickle.dump(hyperparam_instance, f)
    print("loaded all combinations of hyperparameters...")
    print(f"Training {len(hyperparams)} thalamocortical expansion models...")
    
    # finally, we train
    hyperparameter_search(hyper_params=hyperparams, train_data=train_data, val_data=val_data, data_name=args.data.split('/')[-1].split('.')[0], out=args.out)
    
if __name__ == "__main__":
    main()
