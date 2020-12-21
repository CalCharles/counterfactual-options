import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork

class DiagGaussianForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = BasicMLPNetwork(**kwargs)
        self.std = BasicMLPNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.layers += [self.mean, self.std]

        self.train()
        self.reset_parameters()
    
    def forward(self, x):
        x = self.normalization(x)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + 1e-2

forward_nets = {"base": DiagGaussianForwardNetwork}