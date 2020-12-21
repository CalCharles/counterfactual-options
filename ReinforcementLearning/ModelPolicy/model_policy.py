import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy, os
from file_management import default_value_arg
from ReinforcementLearning.Policy.policy import RLoutput

class ModelPolicy():
	def __init__(self, **kwargs):
		self.model = kwargs['model']

	def forward(self): 
		return RLoutput()

class 