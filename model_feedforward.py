from typing import Literal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class feedforward_VVS(nn.Module):

    def __init__(self, input_shape,
        lgn_kernel_size=3, lgn_kernel_stride=2, lgn_sx1:int=0.5, lgn_sy1:int=0.5, lgn_sx2:int=1, lgn_sy2:int=1,
        v1_kernel_size=3, v1_kernel_stride=2,
        activation:Literal["relu", "sigmoid", "none"]="sigmoid"
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

        # stack and set up the lgn kernels
        self.register_buffer("lgn_kernel", torch.stack((on_lgn_kernel, off_lgn_kernel), dim=0).unsqueeze(1))
        self.lgn_stride = lgn_kernel_stride

        # create our v1 layer
        self.v1 = nn.Conv2d(self.C * 2, self.C * 8, kernel_size=v1_kernel_size, stride=v1_kernel_stride, bias=False, device=self.device)
        
        # set up the activation
        match activation:
            case "none":
                self.activation = None
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "relu":
                self.activation = nn.ReLU()

    def forward(self, x, rep:Literal["all", "final"]="final"):

        # go through our layers
        if self.activation == None:
            lgn = F.conv2d(x, self.lgn_kernel, stride=self.lgn_stride)
            v1 = self.v1(lgn)
        else:
            lgn = self.activation( F.conv2d(x, self.lgn_kernel, stride=self.lgn_stride) )
            v1 = self.activation( self.v1(lgn) )

        if rep == "all":
            return (lgn, v1)
        else:
            return v1