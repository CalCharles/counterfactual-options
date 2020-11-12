import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, BasicMLPNetwork

class InteractionNetwork(BasicMLPNetwork):
	def __init__(self, **kwargs):
		super().__init__(kwargs)
		self.final_linear = nn.Linear(self.num_outputs, self.num_outputs)
		self.train()
        self.reset_parameters()
        
	def forward(self, x):
		v = super(x)
		v = self.final_linear(v)
		v = F.sigmoid(v)
		return v
