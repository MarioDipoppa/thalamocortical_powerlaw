import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class feedforward_VVS(nn.Module):

    def __init__(self, input_shape,
        lgn_kernel_size=3, lgn_sx1:int=0.5, lgn_sy1:int=0.5, lgn_sx2:int=1, lgn_sy2:int=1
        ):
        super().__init__()

        self.B, self.C, self.H, self.W = input_shape # should be (B, C, H, W)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set up the lgn kernel as difference of gaussians (output_shape, input_shape, H, W)
        ax = np.arange(-lgn_kernel_size//2 + 1, lgn_kernel_size//2 + 1)
        x, y = np.meshgrid(ax, ax)
        on_lgn_kernel = torch.Tensor(
            (1/(2*np.pi*lgn_sx1*lgn_sy1)) * np.exp(-(x**2/(2*lgn_sx1**2) + y**2/(2*lgn_sy1**2))) -  # first guassian
            (1/(2*np.pi*lgn_sx2*lgn_sy2)) * np.exp(-(x**2/(2*lgn_sx2**2) + y**2/(2*lgn_sy2**2)))    # second guassian
        )

        # the off center cells are just the negative kernel
        off_lgn_kernel = -on_lgn_kernel

        # stack and set up the kernels
        self.lgn_kernel = torch.stack((on_lgn_kernel, off_lgn_kernel), dim=0).unsqueeze(1)

        # create our v1 layer
        self.v1 = nn.Conv2d(self.C * 2, self.C*2 * 8, kernel_size=3, stride=2, bias=False, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # go through our layers
        lgn = self.sigmoid( F.conv2d(x, self.lgn_kernel, stride=1) )
        v1 = self.sigmoid( self.v1(lgn) )

        return (lgn, v1)