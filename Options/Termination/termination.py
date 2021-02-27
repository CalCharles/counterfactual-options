# termination conditions
import numpy as np
import os, cv2, time, torch
from ReinforcementLearning.Policy.policy import pytorch_model

class Termination():
	def __init__(self, **kwargs):
		self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome

	def check(self, state, param):
		return True

class ParameterizedStateTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.use_both = kwargs['use_both'] # supercedes use_diff
		self.epsilon = kwargs['epsilon']
		self.name = kwargs['name']
		self.discrete = True #kwargs['discrete']
		# in the discrete parameter space case
		self.dataset_model = kwargs['dataset_model']
		self.min_use = kwargs['min_use']
		# self.assign_parameters(self.dataset_model) # this line is commented out because we aren't using a stored parameter list
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
			# print(dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name], self.discrete_parameters)
			dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name] = self.discrete_parameters, self.counts 
		else:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name])
			dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name] = self.discrete_parameters, self.counts 

	def check(self, input_state, state, param, true_done=0): # handling diff/both outside
		# param = self.convert_param(param)
		# if self.use_both:
			# if len(diff.shape) == 1:
			# 	s = torch.cat((state, diff), dim=0)
			# 	return (s - param).norm(p=1) <= self.epsilon
			# else:
				# s = torch.cat((state, diff), dim=1)
				# return (s - param).norm(p=1, dim=1) <= self.epsilon
		# elif self.use_diff:
		# 	if len(diff.shape) == 1:
		# 		return (diff - param).norm(p=1) <= self.epsilon
		# 	else:
		# 		return (diff - param).norm(p=1, dim=1) <= self.epsilon
		# else:
		if len(state.shape) == 1:
			return (state - param).norm(p=1) <= self.epsilon
		else:
			return (state - param).norm(p=1, dim=1) <= self.epsilon

class InteractionTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_model = kwargs["dataset_model"]
		self.epsilon = kwargs["epsilon"]

	def check(self, input_state, state, param, true_done=0):
		interaction_pred = self.interaction_model(input_state)
		return interaction_pred > 1 - self.epsilon

class CombinedTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dataset_model = kwargs["dataset_model"]
		self.epsilon = kwargs["epsilon"]
		self.interaction_model = self.dataset_model.interaction_model
		self.parameterized_termination = ParameterizedStateTermination(**kwargs)
		self.interaction_probability = kwargs["interaction_probability"]

	def check(self, input_state, state, param, true_done=0):
		# terminates if the parameter matches and interaction is true
		# has some probability of terminating if interaction is true
		interaction_pred = self.interaction_model(input_state).squeeze() 
		inter = interaction_pred > (1 - self.epsilon)
		param_term = self.parameterized_termination.check(input_state, state, param)
		if self.interaction_probability > 0:
			chances = pytorch_model.wrap(torch.rand(interaction_pred.shape) > self.interaction_probability, cuda=self.dataset_model.iscuda)
			chosen = inter * chances + param_term * inter
			chosen[chosen > 1] = 1
			return chosen
		# print(inter, param_term, state, (state - param), self.parameterized_termination.epsilon)
		return inter * param_term

class TrueTermination(Termination):
	def check(self, input_state, state, param, true_done=0):
		return true_done

terminal_forms = {'param': ParameterizedStateTermination, 'comb': CombinedTermination, 'true': TrueTermination}