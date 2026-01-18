import torch
from torch import nn

class FFTMLP(nn.Module):
    """ FFT + MLP """
    

    def __init__(self, dims: tuple[int] = (100, 50)):
        super.__init__()
        input_dim = 
        self.fc1 = nn.Linear()
        self.net = nn.Sequential(*[nn.Linear()])
