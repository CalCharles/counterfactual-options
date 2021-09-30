import numpy as np
import torch
from Networks.network import pytorch_model

class radial_basis_fn():
	def __call__(self, val, variance):
		return torch.exp(-(val).pow(2) / variance)

class BasisExpansion():
	def __init__(self, **kwargs): 
		self.iscuda = False
		self.num_bases = kwargs["num_base"]
		self.range = np.array([-kwargs["range"], kwargs["range"]])
		self.variance = pytorch_model.wrap(kwargs["variance"], cuda=self.iscuda)
		self.basis_values = pytorch_model.wrap(np.linspace(self.range[0], self.range[1], self.num_bases), cuda=self.iscuda) if self.num_bases > 0 else 0
		self.basis_function = bases[kwargs["basis_function"]]()

	def cuda(self):
		self.iscuda=True
		self.basis_values = pytorch_model.wrap(self.basis_values, cuda=self.iscuda)
		self.variance = pytorch_model.wrap(self.variance, cuda=self.iscuda)

	def cpu(self):
		self.iscuda=False
		self.basis_values = pytorch_model.wrap(self.basis_values, cuda=self.iscuda)
		self.variance = pytorch_model.wrap(self.variance, cuda=self.iscuda)


	def __call__(self, state_val):
		# state value of shape [batch, 1] or [1]
		if self.num_bases < 2:
			return state_val.unsqueeze(-1)
		else:
			return self.basis_function(state_val.unsqueeze(-1) - self.basis_values, self.variance)

class MultiBasisExpansion():
	def __init__(self, **kwargs):
		self.input_dim = kwargs["obs"].shape[0]
		self.num_bases = kwargs["num_bases"]
		self.basis_functions = list()
		for i in range(self.input_dim):
			basis_fn = BasisExpansion(num_base = self.num_bases[i], variance=kwargs["variance"], range = kwargs["range"], basis_function = kwargs["basis_function"])
			self.basis_functions.append(basis_fn)
		self.expanded_dim = np.sum(self.num_bases)
		self.iscuda=False

	def cuda(self):
		for bf in self.basis_functions:
			bf.cuda()

	def cpu(self):
		for bf in self.basis_functions:
			bf.cpu()

	def __call__(self, state):
		expanded_vals = list()
		for i, bf in enumerate(self.basis_functions):
			expanded_vals.append(bf(state[...,i]))
		return torch.cat(expanded_vals, axis=-1)

bases = {"rbf": radial_basis_fn}