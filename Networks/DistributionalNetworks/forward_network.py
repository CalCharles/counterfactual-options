import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Network.network import Network, BasicMLPNetwork

class DiagGaussianForwardNetwork(Network):
	def __init__(self, **kwargs):
		super().__init__(kwargs)
		self.mean = BasicMLPNetwork(kwargs)
		self.variance = BasicMLPNetwork(kwargs)
		self.normalization = kwargs['normalization_function']

		self.train()
        self.reset_parameters()
	
	def forward(self, x):
		x = self.normalization(x)
		return self.mean(x), self.variance(x)
