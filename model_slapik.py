import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class VVS(nn.Module):
    """implements the early visual system model as detailed in
    Mitchell B. Slapik, Harel Z. Shouval; Simulated Complex Cells Contribute to Object Recognition Through Representational Untangling. 
    Neural Comput 2026; 38 (2): 145-164. doi: https://doi.org/10.1162/NECO.a.1480
    """

    def __init__(self, input_shape:tuple[int, int, int, int], 
        lgn_kernel_size=15, lgn_sx1:int=1, lgn_sy1:int=1, lgn_sx2:int=2, lgn_sy2:int=2,
        n_simple_cell_types:int=8, simple_kernel_size:int=15, simple_sx:int=3, simple_sy:int=3, spatial_freq:float=0.8, simple_offset:float=0
    ):
        super().__init__()

        self.B, self.C, self.H, self.W = input_shape # should be (B, C, H, W)
        
        # set up the lgn kernel as difference of gaussians (output_shape, input_shape, H, W)
        ax = np.arange(-lgn_kernel_size//2 + 1, lgn_kernel_size//2 + 1)
        x, y = np.meshgrid(ax, ax)
        lgn_kernel = torch.Tensor(
            (1/(2*np.pi*lgn_sx1*lgn_sy1)) * np.exp(-(x**2/(2*lgn_sx1**2) + y**2/(2*lgn_sy1**2))) -  # first guassian
            (1/(2*np.pi*lgn_sx2*lgn_sy2)) * np.exp(-(x**2/(2*lgn_sx2**2) + y**2/(2*lgn_sy2**2)))    # second guassian
        )
        self.lgn_kernel = torch.reshape(lgn_kernel, (self. C, self.C, lgn_kernel_size, lgn_kernel_size))
        self.lgn_kernel = self.lgn_kernel.repeat(self.B, 1, 1, 1)

        # set up the simple cell kernel(s) as gabor filter and the complex (simple even) which is based on the simple
        self.theta = np.linspace(0, np.pi, n_simple_cell_types, endpoint=False)
        ax = np.arange(-simple_kernel_size // 2 + 1, simple_kernel_size // 2 + 1)
        x, y = np.meshgrid(ax, ax)
        gaussian = (1 / (2 * np.pi * simple_sx * simple_sy)) * np.exp(-(x**2 / (2 * simple_sx**2) + y**2 / (2 * simple_sy**2))) # gaussian portion is constant
        simple_even_kernels = torch.Tensor( np.array([gaussian * np.cos( spatial_freq * (x * np.cos(t) + y * np.sin(t)) - simple_offset ) for t in self.theta]) )
        self.simple_even_kernels = torch.reshape(simple_even_kernels, (self.C * self.theta.shape[0], self.C, simple_kernel_size, simple_kernel_size))
        self.simple_even_kernels = self.simple_even_kernels.repeat(self.B, 1, 1, 1)

        # set up the complex cells, we need to simulate a second set of simple cells to generate the complex cells, where offset = simple_offset + pi/2
        simple_odd_kernels = torch.Tensor( np.array([gaussian * np.cos( spatial_freq * (x * np.cos(t) + y * np.sin(t)) - (simple_offset + np.pi / 2) ) for t in self.theta]) )
        self.simple_odd_kernels = torch.reshape(simple_odd_kernels, (self.C * self.theta.shape[0], self.C, simple_kernel_size, simple_kernel_size))
        self.simple_odd_kernels = self.simple_odd_kernels.repeat(self.B, 1, 1, 1)

    def forward(self, x:torch.Tensor, return_rep:str=None):
        """implements the forward pass through the model

        Args:
            x (torch.Tensor): the input image (n_pixels, n_pixels)
        """

        # view change for easy convolutions
        x = x.view(1, self.B * self.C, self.H, self.W)

        # pass the original image through each layer
        x_retina = self.norm( self.retina(x) ).view(self.B, self.C, self.H, self.W)
        x_lgn = self.norm( self.lgn(x) ).view(self.B, self.C, self.H, self.W)
        x_simple = self.norm( self.simple(x) ).view(self.B, self.C * self.theta.shape[0], self.H, self.W)
        x_complex = self.norm( self.complex(x, flag="independent") ).view(self.B, self.C * self.theta.shape[0], self.H, self.W)

        return (x_retina, x_lgn, x_simple, x_complex)

        """# pass through each layer
        x = x.view(1, self.B * self.C, self.H, self.W)
        x_retina = self.norm( self.retina(x) )
        x_lgn = self.norm( self.lgn(x_retina) )
        x_simple = self.norm( self.simple(x_lgn) )
        x_complex = self.norm( self.complex(x_simple, x_lgn, flag="feed forward") )

        # return the final rep and the requested representation
        match return_rep:
            case "retina":
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), x_retina)
            case "lgn":
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), x_lgn)
            case "simple":
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), x_simple_even)
            case "complex":
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), x_complex)
            case "all":
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), (x_retina.view(self.B, self.C, self.H, self.W), x_lgn.view(self.B, self.C, self.H, self.W), x_simple.view(self.B, self.C*self.theta.shape[0], self.H, self.W), x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W)))
            case None:
                return (x_complex.view(self.B, self.C*self.theta.shape[0], self.H, self.W), None)"""

    def retina(self, x):
        """retina can be modeled by a simple sigmoid function

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the output of the retina
        """

        return 1 / (1 + torch.exp(-x))

    def lgn(self, x):
        """lgn can be modeled by a difference of gaussians

        Args:
            x (torch.Tensor): the input image (B, C, H, W)
        Returns:
            torch.Tensor: the output of the LGN
        """

        # this is just a convolution with DoG kernel
        return F.conv2d(x, self.lgn_kernel, padding="same", groups=self.B)

    def simple(self, x):
        """simple cells can be modeled by a gabor with given theta

        Args:
            x (torch.Tensor): the input image
        Returns:
            torch.Tensor: the output of the LGN (K, H, W)
        """

        # we only use simple even kernels as the simple cell representations
        return F.conv2d(x, self.simple_even_kernels, padding="same", groups=self.B)

    def complex(self, x, l=None, flag="feed forward"):
        """complex cells modeled as a simple even + simple odd on top of the simple

        Args:
            x (torch.Tensor): the input simple representation
            l (torch.Tensor): the lgn representation
        Returns:
            torch.Tensor: the output of simple cells
        """

        # the complex output a second simple computation on top of the already simple cell representation
        # even = F.conv2d(x, self.simple_even_kernels, padding="same", groups=self.B*self.theta.shape[0])
        # odd = F.conv2d(x, self.simple_odd_kernels, padding="same", groups=self.B*self.theta.shape[0])

        if flag == "independent":
            even = F.conv2d(x, self.simple_even_kernels, padding="same", groups=self.B)
            odd = F.conv2d(x, self.simple_odd_kernels, padding="same", groups=self.B)
            return even**2 + odd**2
        elif flag == "feed forward":
            # create the odd rep
            odd = self.norm( F.conv2d(l, self.simple_odd_kernels, padding="same", groups=self.B) )
            return x**2 + odd**2
        else:
            raise ValueError("flag must be 'independent' or 'feed forward'")

    def norm(self, x):
        """normalizes the input tensor from 0-1 using a sigmoid non-linearity

        Args:
            x (torch.Tensor): the tensor for normalization

        Returns:
            torch.Tensor: the normalized tensor
        """

        z = (x - torch.mean(x)) / torch.std(x)
        a = F.sigmoid(z)

        return a
