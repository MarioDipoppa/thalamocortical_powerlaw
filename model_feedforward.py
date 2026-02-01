import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class feedforward_VVS(nn.Module):

    def __init__(self):
        super().__init__()

        