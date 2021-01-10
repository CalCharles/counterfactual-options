import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork

class InteractionNetwork(BasicMLPNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_linear = nn.Linear(self.num_outputs, self.num_outputs)        
        self.normalization = kwargs['normalization_function']
        self.train()
        self.reset_parameters()
        
    def forward(self, x):
        x = self.normalization(x)
        v = super().forward(x)
        v = self.final_linear(v)
        v = torch.sigmoid(v)
        return v

interaction_nets = {"base": InteractionNetwork}