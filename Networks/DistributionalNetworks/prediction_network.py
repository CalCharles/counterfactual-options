import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Networks.network import Network

class DistributionalPredictionNetwork(Network):
	def __init__(self, **kwargs):
		super().__init__(self, **kwargs)
		self.training_parameters = kwargs['training_parameters']

	def forward(self, )