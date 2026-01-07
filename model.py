import torch
import torch.nn as nn
import torch.nn.functional as F

class TCEModel(nn.Module):
    """
    We can model thalamocortical expansion as a simple 1-layer feedforward neural network
    where the input are the LGN features and the output are the V1 features.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))
    
class ModifiedTripletLoss(nn.Module):
    """
    Modified Triplet Loss that I can mess with.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
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