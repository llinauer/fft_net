import torch
from torch import nn

class FFTMLP(nn.Module):
    """ FFT + MLP """
    

    def __init__(self, img_size: tuple[int, int], dims: tuple[int, ...] = (100, 50)):
        super.__init__()
        input_dim = img_size[0] * img_size[1]
        self.fc1 = nn.Linear()
        self.net = nn.Sequential(*[nn.Linear()])
