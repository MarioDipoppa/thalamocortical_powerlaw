import h5py

import numpy as np
import scipy

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp

class Utils:
    
    ##### ------------------------------------- #####
    #####                METRICS                #####
    ##### ------------------------------------- #####
    
    @staticmethod
    def gini(x):
        """Compute the Gini index of an array (NumPy or JAX).
        
        Args:
            x: Input array of non-negative values.
        Returns:
            float: the Gini index; 0 means perfect equality, 1 means maximal inequality.
        """
        if isinstance(x, (jnp.ndarray, jax.Array)):
            x = x.astype(jnp.float64)
            if jnp.all(x == 0):
                return 0.0
            x_sorted = jnp.sort(x)
            n = len(x)
            index = jnp.arange(1, n + 1)
            return (2 * jnp.sum(index * x_sorted) / jnp.sum(x)) / n - (n + 1) / n
        else:
            x = np.array(x, dtype=np.float64)
            if np.all(x == 0):
                return 0.0
            x_sorted = np.sort(x)
            n = len(x)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * x_sorted) / np.sum(x)) / n - (n + 1) / n
    
    ##### ------------------------------------- #####
    #####             LOADING DATA              #####
    ##### ------------------------------------- #####
    
    @staticmethod
    def load_mat(filepath:str, data_key:str, label_key:str="labels", v7:bool=True):
        """Loads a MATLAB .mat file, handles both v7 and earlier.

        Args:
            filepath (str): the path to the .mat file.
        """
        
        if v7 == True:
            # load from MATLAB v7
            mat = scipy.io.loadmat(filepath)
            raw = mat[data_key]
            data = raw
            
            # try to load the labels
            try:
                labels = mat[label_key]
            except:
                labels = None
            
        else:
            # load from HDF5
            with h5py.File(filepath, 'r') as f:
                raw = f[data_key]
                data = np.array(raw)  # this ensures a clean ndarray
            
                # try to load the labels
                try:
                    labels = f[label_key]
                except:
                    labels = None

        # transpose
        data = data.transpose(3, 2, 0, 1)
            
        # convert to float32 tensor
        triplet_data = torch.Tensor(data).float() / 255.0

        # normalize the dataset
        mean = triplet_data.mean()
        std = triplet_data.std()
        triplet_data = (triplet_data - mean) / std

        return triplet_data, labels
    
    ##### ------------------------------------- #####
    #####             COMPUTE LOSS              #####
    ##### ------------------------------------- #####
    
    def compute_triplet_margin_stats(model, dataloader, device, margin=0.2):

        model.eval()
        ap_dists = []
        an_dists = []
        violations = []

        with torch.no_grad():
            for a, p, n in dataloader:
                a, p, n = a.to(device), p.to(device), n.to(device)

                anchor = model(a)
                positive = model(p)
                negative = model(n)

                ap = F.pairwise_distance(anchor, positive)
                an = F.pairwise_distance(anchor, negative)

                ap_dists.append(ap)
                an_dists.append(an)
                violations.append((ap + margin > an).float())

        ap_dists = torch.cat(ap_dists)
        an_dists = torch.cat(an_dists)
        violations = torch.cat(violations)

        mean_ap = ap_dists.mean().item()
        mean_an = an_dists.mean().item()
        violation_rate = violations.mean().item()

        return mean_ap, mean_an, violation_rate
    
    def evaluate_loss(model, data_loader, criterion, device, l1_lambda=0.0):
        model.eval()
        total_loss = 0.0
        total_triplet = 0.0
        total_l1 = 0.0

        with torch.no_grad():
            for a, p, n in data_loader:
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

                total_loss += loss.item()
                total_triplet += triplet_loss.item()
                total_l1 += l1_penalty

        n_batches = len(data_loader)
        return (
            total_loss / n_batches,
            total_triplet / n_batches,
            total_l1 / n_batches
        )
        
    ##### ------------------------------------- #####
    #####             SMOOTHING                 #####
    ##### ------------------------------------- #####
        
    def gaussian_kernel2d(kernel_size=5, sigma=0.25, device='cpu'):
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
        
    def smooth_weights(model, sigma=0.25):
        # Get weights and reshape (embedding_dim, 256) -> (embedding_dim, 1, 16, 16)
        weight = model.fc.weight.data
        emb_dim = weight.shape[0]
        weight_reshaped = weight.view(emb_dim, 1, 16, 16)

        # Build Gaussian kernel
        kernel_size = int(2 * round(3 * sigma) + 1)
        kernel = Utils.gaussian_kernel2d(kernel_size, sigma, device=weight.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Convolve all embeddings in one pass (treat each as separate batch)
        smoothed = F.conv2d(weight_reshaped, kernel, padding=kernel_size // 2)

        # Flatten back and copy into model
        model.fc.weight.data.copy_(smoothed.view(emb_dim, -1))

    @staticmethod
    def batch_generator(arr, batch_size, start_idx=0, end_idx=None):
        """Yields batches from an array with optional start and end indices.
        
        Args:
            arr (np.ndarray): The array to batch.
            batch_size (int): Size of each batch.
            start_idx (int, optional): Starting index. Defaults to 0.
            end_idx (int, optional): Ending index. Defaults to arr.shape[0].
        """
        if end_idx is None:
            end_idx = arr.shape[0]
        for i in range(start_idx, end_idx, batch_size):
            batch = arr[i:i+batch_size].astype(np.float32)
            yield batch

    @staticmethod
    def jax_loss_fn(weights, images_a, images_p, images_n, model_apply, margin, l1_lambda):
        """JAX implementation of Triplet Loss with optional L1 penalty."""
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

    @staticmethod
    def compute_triplet_margin_stats_jax(weights, generator, model_apply, margin=0.2):
        """Computes triplet statistics using JAX."""
        ap_dists = []
        an_dists = []
        violation_counts = 0
        total_samples = 0

        for batch in generator:
            # Batch is [N, 3, H, W]
            a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
            
            a_out = model_apply(a, weights)
            p_out = model_apply(p, weights)
            n_out = model_apply(n, weights)
            
            ap = jnp.sum((a_out - p_out)**2, axis=1)
            an = jnp.sum((a_out - n_out)**2, axis=1)
            
            ap_dists.append(ap)
            an_dists.append(an)
            
            violations = (ap + margin > an)
            violation_counts += jnp.sum(violations)
            total_samples += len(violations)

        all_ap = jnp.concatenate(ap_dists)
        all_an = jnp.concatenate(an_dists)
        
        mean_ap = jnp.mean(all_ap)
        mean_an = jnp.mean(all_an)
        violation_rate = violation_counts / total_samples
        
        return float(mean_ap), float(mean_an), float(violation_rate)

    @staticmethod
    def evaluate_loss_jax(weights, generator, model_apply, margin, l1_lambda):
        """Evaluates JAX loss over a dataset."""
        total_loss = 0.0
        n_batches = 0
        for batch in generator:
            a, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
            total_loss += Utils.jax_loss_fn(weights, a, p, n, model_apply, margin, l1_lambda)
            n_batches += 1
        return float(total_loss / n_batches) if n_batches > 0 else 0.0