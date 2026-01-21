import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class thalamocorticol_expansion(nn.Module):
    """a unified model for thalamocortical expansion
    """

    def __init__(self, input_dim:tuple[int, int], lgn_dim:tuple[int, int], v1_dim:int, device:str):
        """initializes our tce model

        Args:
            input_dim (tuple[int, int]): the input image dimensions, expects square image
            lgn_dim (tuple[int, int]): the lgn dimensions, should be square and smaller than input_dim
            v1_dim (int): the v1 dimensions
            device (str["cuda", "cpu"]): the device to run on
        """
        super().__init__()

        # just double check the shapes
        assert input_dim[0] == input_dim[1], "Input dimensions must be square"
        assert lgn_dim[0] == lgn_dim[1], "LGN dimensions must be square"
        assert lgn_dim[0] < input_dim[0], "LGN dimensions must be smaller than input dimensions"

        # just store our inputs
        self.input_dim = input_dim
        self.lgn_dim = lgn_dim
        self.v1_dim = v1_dim

        # build our lgn DoG model
        self.lgn = LGNLayer(input_size=input_dim[0], lgn_size=lgn_dim[0], kernel_size=input_dim[0]//lgn_dim[0], sigma_center_range=(0.8, 1.6), sigma_surround_range=(1.6, 3.2), device=device)
        self.v1 = nn.Linear(lgn_dim[0]*lgn_dim[1], v1_dim) #, use_bias=False, device=device) # this just becomes matrix multiplication

    def forward(self, x):
        """forward pass

        Args:
            x (torch.Tensor): image tensor

        Returns:
            torch.Tensor: the output representation
        """

        # toss the data through each step
        x = self.lgn(x)
        return x
        x = x.flatten() # just flatten as input for v1 Linear layer
        x = self.v1(x)
        return x

class LGNLayer(nn.Module):
    def __init__(self, input_size:int=16, lgn_size:int=4, kernel_size:int=4, sigma_center_range:tuple[float, float]=(0.5, 1.0), sigma_surround_range:tuple[float, float]=(1.2, 2.0), device="cpu"):
        super().__init__()

        self.input_size = input_size
        self.lgn_size = lgn_size
        self.kernel_size = kernel_size
        self.stride = input_size // lgn_size

        kernels = []
        positions = []

        for i in range(lgn_size):
            for j in range(lgn_size):
                sigma_c = torch.empty(1).uniform_(*sigma_center_range).item()
                sigma_s = torch.empty(1).uniform_(
                    max(sigma_c + 0.1, sigma_surround_range[0]),
                    sigma_surround_range[1]
                ).item()
                sign = 1 if torch.rand(1).item() < 0.5 else -1

                kernel = self.dog_kernel(kernel_size, sigma_c, sigma_s, sign, device=device)

                kernels.append(kernel)
                positions.append((i * self.stride, j * self.stride))

        # Register as buffers (fixed, non-trainable)
        self.register_buffer("kernels", torch.stack(kernels))  # (16, 4, 4)
        self.positions = positions  # Python list is fine

    def dog_kernel(self, size:int, sigma_center:float, sigma_surround:float, sign:int=1, device:str="cpu"):
        """
        Create a 2D Difference-of-Gaussians (DoG) kernel.

        Args:
            size (int): the size of the kernel
            sigma_center (float): the standard deviation of the center kernel
            sigma_surround (float): the standard deviation of the surround kernel
            sign (int): +1 (ON) or -1 (OFF)

        Returns:
            kernel (torch.Tensor): (size, size) tensor
        """
        assert sigma_center < sigma_surround

        ax = torch.arange(size, device=device) - (size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")

        center = torch.exp(-(xx**2 + yy**2) / (2 * sigma_center**2))
        surround = torch.exp(-(xx**2 + yy**2) / (2 * sigma_surround**2))

        kernel = sign * (center - surround)
        kernel = kernel / (kernel.norm() + 1e-8)

        return kernel

    def forward(self, x):
        """
        Args:
            x: (16, 16) or (1, 16, 16)
        Returns:
            h: (4, 4)
        """
        if x.dim() == 3:
            x = x.squeeze(0)

        h = torch.zeros(
            self.lgn_size,
            self.lgn_size,
            device=x.device
        )

        idx = 0
        for i in range(self.lgn_size):
            for j in range(self.lgn_size):
                r, c = self.positions[idx]
                patch = x[r:r+self.kernel_size, c:c+self.kernel_size]
                h[i, j] = torch.sum(patch * self.kernels[idx])
                idx += 1

        return h

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
        assert C == self.channels, f"Expected input with {self.channels} channels, got {C}"

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
        x = self.fc(x)
        x = self.relu(x)
        return x
    
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
        ap_dist = (anchor - positive).pow(2).sum(1) # removed sqrt bc it was running into issues
        an_dist = (anchor - negative).pow(2).sum(1)
        
        # print(f"in dtypes: {anchor.dtype, positive.dtype, negative.dtype}, dist dtypes: {ap_dist.dtype, an_dist.dtype}, data shapes: {anchor.shape, positive.shape, negative.shape}")
        
        # we can add additional terms here
        # for example, we can penalize using an arbitrarily 
        # large number of neurons (since more neurons = more power)
        
        # compute the actual loss value
        loss = F.relu(ap_dist - an_dist + self.margin)
        
        return loss.mean()