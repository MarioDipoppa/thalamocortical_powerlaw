import h5py

import numpy as np
import scipy

import torch
import torch.nn.functional as F

class Utils:
    
    ##### ------------------------------------- #####
    #####                METRICS                #####
    ##### ------------------------------------- #####
    
    @staticmethod
    def gini(x:np.ndarray):
        """Compute the Gini index of a numpy array.
        
        Args:
            x (np.ndarray): Input array of non-negative values.
        Returns:
            float: the Gini index; 0 means perfect equality, 1 means maximal inequality.
        """
        
        x = np.array(x, dtype=np.float64)
        assert np.amin(x) >= 0, "Gini index is only defined for non-negative values"
        
        # if all values are 0, just return 0
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
    def load_mat(filepath:str, mat_key:str, v7:bool=True):
        """Loads a MATLAB .mat file, handles both v7 and earlier.

        Args:
            filepath (str): the path to the .mat file.
        """
        
        if v7 == True:
            # load from MATLAB v7
            mat = scipy.io.loadmat(filepath)
            raw = mat[mat_key]
            data = raw
            
            labels = mat["labels"]
        else:
            # load from HDF5
            with h5py.File(filepath, 'r') as f:
                raw = f[mat_key]
                data = np.array(raw)  # this ensures a clean ndarray

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