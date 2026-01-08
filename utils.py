import h5py

import numpy as np
import scipy

import torch

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
        assert np.amin(x) < 0, "Gini index is only defined for non-negative values"
        
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