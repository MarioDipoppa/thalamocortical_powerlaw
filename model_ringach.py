import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Utils as u
from train import TripletDataset

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

class ringach_VVS(nn.Module):
    
    def __init__(self, shape:tuple=(224, 224), l:float=1, n_RGC:int=100, k_pos=0.155, 
                 shift:bool=False, alpha:float=1, eta_0:list=[0, 0],
                 rotate:bool=False, theta:float=np.pi/2,
                 v1_dim:int=224):
        super().__init__()
        
        # store the lambda parameter which pretty much modulates everything
        self.l = l
        
        ##### ----- RGC layer set up ----- #####
        
        # some other rotation/translation params
        self.eta = shift * self.l * np.array(eta_0)
        self.R = np.eye(2) if rotate is False else np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # let's set up our on/off RGC layer
        self.pos = pos = np.array([[i,j] for i in range(-n_RGC // 2, n_RGC // 2) for j in range(-n_RGC // 2, n_RGC // 2)])
        M = self.l * np.array([[1, 1], [3**(1/2), -(3**(1/2))]])
        self.on_RGC_locs = ((pos @ M.T) + self.eta + np.random.normal(0, k_pos*self.l, size=pos.shape)) @ self.R.T
        self.off_RGC_locs = (pos @ M.T) + np.random.normal(0, k_pos*self.l, size=pos.shape)
        
        # let's crop the grid so it works with square images
        half_size = self.l * (n_RGC // 2)  # or smaller if you want tighter crop
        xmin, xmax = -half_size, half_size
        ymin, ymax = -half_size, half_size
        mask = ((self.on_RGC_locs[:,0] >= xmin) & (self.on_RGC_locs[:,0] <= xmax) & (self.on_RGC_locs[:,1] >= ymin) & (self.on_RGC_locs[:,1] <= ymax))
        self.on_RGC_locs = self.on_RGC_locs[mask]
        mask = ((self.off_RGC_locs[:,0] >= xmin) & (self.off_RGC_locs[:,0] <= xmax) & (self.off_RGC_locs[:,1] >= ymin) & (self.off_RGC_locs[:,1] <= ymax))
        self.off_RGC_locs = self.off_RGC_locs[mask]
        
        # scale to give the right size
        scale = shape[0] / max(xmax - xmin, ymax - ymin)
        self.on_RGC_locs = ((self.on_RGC_locs - [xmin, ymin]) * scale)
        self.off_RGC_locs = ((self.off_RGC_locs - [xmin, ymin]) * scale)
        
        # let's add the on/off functionality
        self.on_RGC_locs = np.hstack([self.on_RGC_locs, np.full((self.on_RGC_locs.shape[0], 1), 1)])
        self.off_RGC_locs = np.hstack([self.off_RGC_locs, np.full((self.off_RGC_locs.shape[0], 1), -1)])
        self.RGC_locs = np.concatenate([self.on_RGC_locs, self.off_RGC_locs])
        
        # define the RGC activations
        self.RGC_activation_sigma = 0.7 * self.l * scale
        RGC_acts = []
        for loc in self.RGC_locs:
            x = np.arange(-loc[0], shape[0]-loc[0])
            y = np.arange(-loc[1], shape[1]-loc[1])
            xx, yy = np.meshgrid(x, y)
            RGC_activation = np.exp(-(xx**2 + yy**2)/(2*self.RGC_activation_sigma**2))
            RGC_activation = np.where(RGC_activation < 10**(-5), 0., RGC_activation)
            RGC_activation /= np.sum(RGC_activation)
            RGC_acts.append(RGC_activation * loc[2])
        self.RGC_activations = np.array(RGC_acts)
        
        ##### ----- LGN layer set up ----- #####
        
        # now let's define the LGN locs and activations
        self.LGN_RGC_idx = np.random.choice(self.RGC_activations.shape[0], size=int(self.RGC_activations.shape[0]*1.5), replace=True)
        self.LGN_locs = np.concatenate([self.RGC_locs, self.RGC_locs[self.LGN_RGC_idx]])
        self.LGN_activations = np.concatenate([self.RGC_activations, self.RGC_activations[self.LGN_RGC_idx]])
        
        ##### ----- V1 layer set up ----- #####
        
        # v1 maps various inputs from LGN to v1 space
        X, Y = np.meshgrid(np.linspace(0, shape[0], num=v1_dim, endpoint=True), np.linspace(0, shape[1], num=v1_dim, endpoint=True))
        self.V1_locs = np.vstack([X.ravel(), Y.ravel()]).T
        
        radius = 200 * self.RGC_activation_sigma

        # build spatial index on LGN
        tree = cKDTree(self.LGN_locs[:, :2])  # use spatial coords only

        rows = []
        cols = []
        vals = []

        for i, v in enumerate(self.V1_locs):
            # find nearby LGN cells
            idx = tree.query_ball_point(v[:2], radius)

            if not idx:
                continue
            
            # compute weight of connection
            dist2 = np.sum((self.LGN_locs[idx, :2] - v[:2])**2, axis=1)
            p = np.array([1 if np.random.random() >= 0.85 * np.exp( - d / (2 * self.RGC_activation_sigma*0.97) ) else 0 for d in dist2])
            w = p * np.exp( - dist2 / (2 * self.RGC_activation_sigma*1.1) )

            rows.extend([i] * len(idx))  # V1 index
            cols.extend(idx)             # LGN index
            vals.extend(w)

        self.LGN_V1_conn = coo_matrix( (vals, (rows, cols)), shape=(len(self.V1_locs), len(self.LGN_locs)) )
        self.LGN_V1_conn = self.LGN_V1_conn.tocsr()
        
    def forward(self, x):
        """performs a forward pass

        Args:
            x (torch.Tensor): the input image

        Returns:
            torch.Tensor: the output v1 representation
        """
        
        r = self.rgc_vectorized(x)
        l = self.lgn_vectorized(r)
        v = self.v1_vectorized(l)
        
        return (r, l, v)
    
    def rgc_vectorized(self, x):
        on = x * self.RGC_activations[self.RGC_locs[:,2] == 1]
        off = (1 - x) * self.RGC_activations[self.RGC_locs[:,2] == -1]
        act = np.concatenate([on, off])
        return act.reshape((act.shape[0], -1))
    
    def lgn_vectorized(self, x):
        return np.concatenate([x, x[self.LGN_RGC_idx]])
    
    def v1_vectorized(self, x):
        return self.LGN_V1_conn @ x.reshape(x.shape[0], -1)
    