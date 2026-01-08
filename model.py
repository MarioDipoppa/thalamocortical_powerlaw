import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RGC2LGN(nn.Module):
    """
    Provides a mapping from retinal ganglion cells (RGCs) to the 
    lateral geniculate nucleus (LGN) using multi-scale Gaussian filtering.
    
    - produces a fixed-length representation of the input image

    Input:  (B, 3, H, W)
    Output: (B, 3 * num_scales)
    """

    def __init__(self, sigmas=(0.8, 1.6, 3.2), kernel_size=None, channels=3):
        super().__init__()

        self.sigmas = sigmas
        self.channels = channels
        
        # Choose kernel size automatically if not given
        if kernel_size is None:
            kernel_size = int(2 * np.ceil(3 * max(sigmas)) + 1)
        self.kernel_size = kernel_size

        # Build Gaussian kernels
        kernels = []
        for sigma in sigmas:
            k = self.gaussian_kernel_2d(kernel_size, sigma)
            kernels.append(k)

        kernels = torch.stack(kernels)  # (S, kH, kW)

        # Register as buffer so it moves with the model (CPU/GPU)
        self.register_buffer("kernels", kernels)

    def forward(self, x):
        """
        Apply Gaussian filtering at multiple scales.
        
        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            torch.Tensor: (B, C * S)
        """
        
        B, C, H, W = x.shape
        S = self.kernels.shape[0]

        # Prepare kernels
        kernels = self.kernels.unsqueeze(1)      # (S, 1, kH, kW)
        kernels = kernels.repeat(C, 1, 1, 1)     # (C*S, 1, kH, kW)

        # Depthwise multi-filter convolution
        y = F.conv2d(x, kernels, padding=self.kernel_size // 2, groups=C)

        # y: (B, C*S, H, W)
        y = y.view(B, C, S, H, W)

        # Global average pooling
        y = y.mean(dim=(-2, -1))   # (B, C, S)

        # Flatten
        y = y.flatten(start_dim=1) # (B, C*S)

        return y
    
    def gaussian_kernel_2d(self, kernel_size: int, sigma: float, device=None):
        """
        Create a normalized 2D Gaussian kernel.
        
        Args:
            kernel_size (int): Size of the kernel (must be odd).
            sigma (float): Standard deviation of the Gaussian.
            device: Device to create the kernel on.
        Returns:
            torch.Tensor: 2D Gaussian kernel of shape (kernel_size, kernel_size).
        """
        
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

class LGN2V1(nn.Module):
    """
    We can model thalamocortical expansion as a simple 1-layer 
    feedforward neural network where the input are the LGN features 
    and the output are the V1 features.
    """
    
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x:torch.Tensor):
        return self.relu(self.fc(x))
    
class ModifiedTripletLoss(nn.Module):
    """
    Modified Triplet Loss that I can mess with.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor):
        """computes the triplet loss with modifications.

        Args:
            anchor (torch.Tensor): The anchor feature vector.
            positive (torch.Tensor): The positive feature vector.
            negative (torch.Tensor): The negative feature vector.

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        
        # compute the distance between anchor-positive and anchor-negative
        ap_dist = (anchor - positive).pow(2).sum(1).sqrt()
        an_dist = (anchor - negative).pow(2).sum(1).sqrt()
        
        # we can add additional terms here
        # for example, we can penalize using an arbitrarily 
        # large number of neurons (since more neurons = more power)
        
        # compute the actual loss value
        loss = F.relu(ap_dist - an_dist + self.margin)
        
        return loss.mean()