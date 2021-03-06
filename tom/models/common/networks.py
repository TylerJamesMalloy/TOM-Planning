import torch as th
import torch.nn as nn
from torch.nn import functional as F

class Policy(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        layers):
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Belief(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        layers):
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

class Transition(nn.Module):
    def __init__(self,
        state_size,
        action_size,
        mask_size,
        layers):
        super().__init__()
        
        # define network layers
        self.fc1 = nn.Linear(state_size+action_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.state = nn.Linear(layers[1], state_size)
        self.reward = nn.Linear(layers[1], 1)
        self.done = nn.Linear(layers[1], 1)
        self.mask = nn.Linear(layers[1], mask_size)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        shared_layer = F.relu(self.fc2(x))

        state = th.sigmoid(self.state(shared_layer))
        reward = F.sigmoid(self.reward(shared_layer))
        done = th.sigmoid(self.done(shared_layer))
        mask = th.sigmoid(self.mask(shared_layer))
        return state, reward, done, mask 

class TOMTransition(nn.Module):
    def __init__(self,
        state_size,
        action_size,
        mask_size,
        belief_size,
        num_players,
        layers):
        super().__init__()
        
        # define network layers
        self.fc1 = nn.Linear(state_size+action_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.state = nn.Linear(layers[1], state_size)
        self.reward = nn.Linear(layers[1], 1)
        self.done = nn.Linear(layers[1], 1)
        self.mask = nn.Linear(layers[1], mask_size)
        self.belief = nn.Linear(layers[1], belief_size * num_players)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        shared_layer = F.relu(self.fc2(x))

        state = th.sigmoid(self.state(shared_layer))
        reward = self.reward(shared_layer) # fix this 
        done = th.sigmoid(self.done(shared_layer))
        mask = th.sigmoid(self.mask(shared_layer))
        belief = th.sigmoid(self.belief(shared_layer))
        return state, reward, done, mask, belief

class FactoredTransition(nn.Module):
    def __init__(self,
        observation_size,
        action_size,
        layers,
        activation="sigmoid"):
        super().__init__()
        
        self.activation = activation
        # define network layers
        self.fc1 = nn.Linear(observation_size+action_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.state = nn.Linear(layers[1], observation_size)
        
    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        shared_layer = F.relu(self.fc2(x))
        assert(False)
        if(self.activation == "sigmoid"):
            state = th.sigmoid(self.state(shared_layer))
        elif(self.activation == "relu"):
            state = th.relu(self.state(shared_layer))

        return state