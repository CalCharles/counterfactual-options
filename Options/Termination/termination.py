# termination conditions
import numpy as np
import os, cv2, time

class Termination():
	def __init__(self, **kwargs):
		pass

	def check(self, state, diff, param):
		return True

class ParameterizedStateTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome
		self.use_both = kwargs['use_both'] # supercedes use_diff
		self.epsilon = kwargs['epsilon']
		self.name = kwargs['name']
		self.discrete = True #kwargs['discrete']
		# in the discrete parameter space case
		dataset_model = kwargs['dataset_model']
		self.min_use = kwargs['min_use']
		self.assign_parameters(dataset_model)
		print(self.name)

	def assign_parameters(self, dataset_model):
		def assign_dict(observed, totals):
			new_observed = list()
			new_totals = list()
			for i in range(len(observed)):
				if totals[i] >= self.min_use and observed[i][1].sum() > 0: # should have a minimum number, and a nonzero mask
					new_observed.append(observed[i])
					new_totals.append(totals[i])
			return new_observed, new_totals
		if self.use_both:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_both[self.name], dataset_model.both_counts[self.name])
			dataset_model.observed_both[self.name], dataset_model.both_counts[self.name] = self.discrete_parameters, self.counts 
		elif self.use_diff:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name])
			print(dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name], self.discrete_parameters)
			dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name] = self.discrete_parameters, self.counts 
		else:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name])
			dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name] = self.discrete_parameters, self.counts 

	def check(self, state, diff, param):
		# param = self.convert_param(param)
		if self.use_both:
			if len(diff.shape) == 1:
				s = torch.cat((state, diff), dim=0)
				return (s - param).norm(p=1) <= self.epsilon
			else:
				s = torch.cat((state, diff), dim=1)
				return (s - param).norm(p=1, dim=1) <= self.epsilon
		elif self.use_diff:
			if len(diff.shape) == 1:
				return (diff - param).norm(p=1) <= self.epsilon
			else:
				return (diff - param).norm(p=1, dim=1) <= self.epsilon
		else:
			if len(state.shape) == 1:
				return (state - param).norm(p=1) <= self.epsilon
			else:
				return (state - param).norm(p=1, dim=1) <= self.epsilon

terminal_forms = {'param': ParameterizedStateTermination}