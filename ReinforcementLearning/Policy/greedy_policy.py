import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy, os
from file_management import default_value_arg

class GreedyActionPolicy(Policy):
	def __init__(self, **kwargs):

	def forward(self, x):
		# computes the RLoutput with the most important components: RLoutput(values, dist_entropy, probs, log_probs, action_values, std, Q_vals, dist)
		# only adds the probs, which are one hot for the behavior that moves the agent closer to the desired behavior
		# needs to know: the parameter of the current step
		# the parameter of the target step
		# for actions and paddle: look at the parameter of the target step and see which parameter of the current step maps best
		# for paddle and ball: look at the parameter of the target step and see which parameter of the 
		