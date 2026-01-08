import numpy as np
import scipy

class Utils:
    
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