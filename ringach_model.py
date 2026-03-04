import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

import jax
import jax.numpy as jnp

class ringach_VVS:
    """
    A JAX implementation of the Ringach VVS model.
    Pre-calculates geometry and weights to enable fast, differentiable forward passes.
    """
    
    def __init__(self, shape:tuple=(224, 224), l:float=1, n_RGC:int=100, k_pos=0.155, 
                 shift:bool=False, alpha:float=1, eta_0:list=[0, 0],
                 rotate:bool=False, theta:float=np.pi/2,
                 v1_dim:int=224):
        
        # store the lambda parameter
        self.l = l
        self.shape = shape
        
        ##### ----- RGC layer set up ----- #####
        # Pre-calculating fixed geometry using NumPy (only done once)
        self.eta = shift * self.l * np.array(eta_0)
        self.R = np.eye(2) if rotate is False else np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        pos = np.array([[i,j] for i in range(-n_RGC // 2, n_RGC // 2) for j in range(-n_RGC // 2, n_RGC // 2)])
        M = self.l * np.array([[1, 1], [3**(1/2), -(3**(1/2))]])
        on_RGC_locs = ((pos @ M.T) + self.eta + np.random.normal(0, k_pos*self.l, size=pos.shape)) @ self.R.T
        off_RGC_locs = (pos @ M.T) + np.random.normal(0, k_pos*self.l, size=pos.shape)
        
        half_size = self.l * (n_RGC // 2)
        xmin, xmax = -half_size, half_size
        ymin, ymax = -half_size, half_size
        
        mask_on = ((on_RGC_locs[:,0] >= xmin) & (on_RGC_locs[:,0] <= xmax) & (on_RGC_locs[:,1] >= ymin) & (on_RGC_locs[:,1] <= ymax))
        on_RGC_locs = on_RGC_locs[mask_on]
        mask_off = ((off_RGC_locs[:,0] >= xmin) & (off_RGC_locs[:,0] <= xmax) & (off_RGC_locs[:,1] >= ymin) & (off_RGC_locs[:,1] <= ymax))
        off_RGC_locs = off_RGC_locs[mask_off]
        
        scale = shape[0] / max(xmax - xmin, ymax - ymin)
        on_RGC_locs = ((on_RGC_locs - [xmin, ymin]) * scale)
        off_RGC_locs = ((off_RGC_locs - [xmin, ymin]) * scale)
        
        on_locs_with_type = np.hstack([on_RGC_locs, np.full((on_RGC_locs.shape[0], 1), 1)])
        off_locs_with_type = np.hstack([off_RGC_locs, np.full((off_RGC_locs.shape[0], 1), -1)])
        self.RGC_locs = np.concatenate([on_locs_with_type, off_locs_with_type])
        
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
        
        # Store RGC activations as JAX array for differentiation
        self.RGC_activations = jnp.array(RGC_acts)
        self.RGC_types = jnp.array(self.RGC_locs[:, 2])
        self.RGC_on_mask = (self.RGC_types == 1).reshape(-1, 1, 1)
        
        ##### ----- LGN layer set up ----- #####
        self.LGN_RGC_idx = np.random.choice(self.RGC_activations.shape[0], size=int(self.RGC_activations.shape[0]*1.5), replace=True)
        # We don't need to store full LGN activations if they are just indexing RGC ones
        self.LGN_RGC_idx_jnp = jnp.array(self.LGN_RGC_idx)
        
        ##### ----- V1 layer set up ----- #####
        X, Y = np.meshgrid(np.linspace(0, shape[0], num=v1_dim, endpoint=True), np.linspace(0, shape[1], num=v1_dim, endpoint=True))
        self.V1_locs = np.vstack([X.ravel(), Y.ravel()]).T
        
        radius = 200 * self.RGC_activation_sigma
        # Note: The original code concatenated RGC_locs and RGC_locs[LGN_RGC_idx]
        # Let's rebuild the LGN_locs for the tree to match indexing
        lgn_locs_total = np.concatenate([self.RGC_locs, self.RGC_locs[self.LGN_RGC_idx]])
        tree = cKDTree(lgn_locs_total[:, :2])

        rows, cols, vals = [], [], []
        for i, v in enumerate(self.V1_locs):
            idx = tree.query_ball_point(v[:2], radius)
            if not idx: continue
            dist2 = np.sum((lgn_locs_total[idx, :2] - v[:2])**2, axis=1)
            p = np.array([1 if np.random.random() >= 0.85 * np.exp( - d / (2 * self.RGC_activation_sigma*0.97) ) else 0 for d in dist2])
            w = p * np.exp( - dist2 / (2 * self.RGC_activation_sigma*1.1) )
            rows.extend([i] * len(idx))
            cols.extend(idx)
            vals.extend(w)

        # Convert connectivity to dense JAX matrix
        # Note: If memory is an issue for your scale, we can revisit sparse.
        dense_conn = np.zeros((len(self.V1_locs), len(lgn_locs_total)), dtype=np.float32)
        dense_conn[rows, cols] = vals
        self.LGN_V1_conn = jnp.array(dense_conn)

    def rgc_forward(self, x):
        """
        Differentiable RGC pass.
        x: input image (H, W) or (B, H, W)
        """
        # x is (H, W), reshape to (1, H*W) for broadcasting if needed
        x_flat = x.reshape(-1)
        
        # Original logic:
        # on = x * self.RGC_activations[self.RGC_types == 1]
        # off = (1 - x) * self.RGC_activations[self.RGC_types == -1]
        
        # Vectorized version:
        # We can multiply the whole RGC_activations by x or (1-x) depending on the type
        # RGC_activations is (N, H, W). RGC_types is (N,)
        
        # Use pre-calculated mask
        image_mod = jnp.where(self.RGC_on_mask, x, 1.0 - x)
        
        act = image_mod * self.RGC_activations
        # Result is (N, H, W), flatten each filter's output
        return act.reshape(act.shape[0], -1).sum(axis=1) # Sum over pixel contributions

    def lgn_forward(self, rgc_act):
        """
        rgc_act: (N_RGC,)
        """
        return jnp.concatenate([rgc_act, rgc_act[self.LGN_RGC_idx_jnp]])

    def v1_forward(self, lgn_act, weights=None):
        """
        lgn_act: (N_LGN,)
        weights: Optional trainable dense weights for the V1 connectivity (D_V1, D_LGN)
        """
        # Dense multiplication
        if weights is not None:
            return weights @ lgn_act
        return self.LGN_V1_conn @ lgn_act

    def forward(self, x, weights=None):
        """
        The full differentiable pass. 
        Note: JAX works best with functions. 
        You can use `jax.jit(model.forward)` or `jax.grad(lambda x: model.forward(x).sum())`.
        Note: If weights is provided, it must be the data array for the sparse V1 connectivity.
        """
        r = self.rgc_forward(x)
        l = self.lgn_forward(r)
        v = jax.nn.relu(self.v1_forward(l, weights=weights))
        return (r, l, v)
