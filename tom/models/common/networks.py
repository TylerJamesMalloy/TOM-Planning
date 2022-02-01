import torch as th
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        layers):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = th.sigmoid(self.fc3(x))
        return x

