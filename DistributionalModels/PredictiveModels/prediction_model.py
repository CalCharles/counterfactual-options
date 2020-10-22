import numpy as np
import os, cv2, time
import torch
from DistributionalModels.distributional_model import DistributionalModel

def get_corresponding_mean(states, state_diff, rollouts):
	input_states = torch.cat((states, state_diff), dim = 1)
	indexes = rollouts.values["done"].nonzero()[:,0].flatten()
	last_idx = 0
	means = []
	for idx in indexes: # inefficient, should have a one-liner for this
		if idx != self.filled-1:
			means.append(input_states[last_idx:idx+1].mean())
			last_idx = idx+1
		else:
			break
	return torch.stack(means, dim=0)

def get_corresponding_first(rollouts)
	states, state_diff, action = rollouts.get_values("state"), rollouts.get_values("state_diff"), rollouts.get_values("action")
	input_states = torch.cat((states, state_diff, action), dim = 1)
	idxes = torch.cat((torch.zeros((1,)), rollouts.get_values("done").nonzero() + 1)).long()
	input_states = input_states[idxes]
	return input_states

class PredictionInputModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		predict the output distribution from the action and the first state of the counterfactual distribution using regression, predicting
		a distribution and a probability for each value
		'''
		super().__init__(**kwargs)
		self.prediction_model = kwargs["predictive_model"](**kwargs)
		self.output_distribution = kwargs["output_distribution"](**kwargs)
		self.flatten = kwargs["environment_model"].flatten_factored_state
		self.use_counterfactual_mask = kwargs["use_counterfactual_mask"] 

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		input_states = get_corresponding_first(states, state_diff, counterfactual_rollouts)
		# classify out of distribution values
		nc_input_states = get_corresponding_first(states, state_diff, non_counterfactual_rollouts)
		input_states = torch.cat((input_states, nc_input_states), dim = 0)
		output_states = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		counterfactual_distribution = torch.cat(torch.ones(output_states.size(0)), torch.zeros(nc_input_states.size(0)))
		output_states = torch.cat((output_states, torch.zeros((nc_input_states.size(0), output_states.size(1)))), dim=0)
		if self.use_counterfactual_mask:
			training_probability, counterfactual_component_masks = counterfactual_factored_mask(outcome_rollouts)
			counterfactual_mask = self.flatten(counterfactual_factored_mask)
			self.prediction_model.train(input_states, output_states, counterfactual_distribution, output_mask = counterfactual_component_masks[name])
		else:
			self.prediction_model.train(input_states, output_states, counterfactual_distribution)

	def predict(self, rollouts):
		data = torch.cat((rollouts.get_values("state"), rollouts.get_values("state_diff")), dim = 1)
		mu, sigma, dist = self.prediction_model.test(data)
		return dist

class FactoredOutcomePredictionModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		predict the output distribution from the action
		a distribution and a probability for each value
		'''
		super().__init__(**kwargs)
		self.unflatten = kwargs["environment_model"].unflatten_state
		self.names = kwargs["environment_model"].object_names
        self.modes = dict()
        self.num_modes = 0
		self.prediction_models = {n: kwargs["predictive_model"](**kwargs) for n in names}
		self.use_counterfactual_mask = kwargs["use_counterfactual_mask"] 

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		c_state, c_diff = counterfactual_rollouts.get_values("state"), counterfactual_rollouts.get_values("state_diff")
		input_states = get_corresponding_first(c_state, c_diff, counterfactual_rollouts)
		nc_state, nc_diff = non_counterfactual_rollouts.get_values("state"), non_counterfactual_rollouts.get_values("state_diff")
		nc_input_states = get_corresponding_first(nc_state, nc_diff, non_counterfactual_rollouts)
		out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, instanced=False)
		out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, instanced=False)
		input_states = torch.cat((input_states, nc_input_states), dim = 0)
		if self.use_counterfactual_mask:
			training_probability, counterfactual_component_masks = counterfactual_factored_mask(outcome_rollouts)
		for name in self.names:
			output_states = torch.cat((out_state[name], out_diff[name]), dim = 1)
			output_distribution = torch.cat(torch.ones(output_states.size(0)), torch.zeros(nc_input_states.size(0)))
			output_states = torch.cat((output_states, torch.zeros((nc_input_states.size(0), output_states.size(1)))))
			if self.use_counterfactual_mask:
				self.prediction_models[name].train(input_states, output_states, output_distribution, output_mask = counterfactual_component_masks[name])
			else:
				self.prediction_models[name].train(input_states, output_states, output_distribution)

	def predict(self, rollouts):
		data = torch.cat((rollouts.get_values("state"), rollouts.get_values("state_diff")), dim = 1)
		mu, sigma, dist = self.prediction_model.test(data)
		return dist

# TODO: put this in the right place
def counterfactual_factored_mask(outcome_rollouts):
	out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, instanced=False)
	out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, instanced=False)
	factored_data = dict()
	name_counts = Counter()
	for name in self.names:
		factored_data[name] = torch.cat((out_state[name], out_diff[name]), dim = 1)
	counterfactual_factored_data = {n: [] for n in self.names}
	EPSILON = 1e-10 # TODO: move this outside?
	counterfactual_component_probabilities = {n: torch.zeros(factored_data[n].shape) for n in self.names}
	for i in [k*self.num_options for k in range(outcome_rollouts.length // self.num_options)]: # TODO: assert evenly divisible
		for n in self.names:
			s = factored_data[n][i]
			state_equals = lambda x,y: (x-y).norm(p=1) < EPSILON
			components_unequal = lambda x,y: ((x-y).abs() > EPSILON).float()
			if sum([int(not state_equals(s, factored_data[n][i+j])) for j in range(1, self.num_options)]):
				counterfactual_factored_data[n] += [factored_data[n][i+j] for j in range(self.num_options)]
				counterfactual_component_masks[n] += sum([components_unequal(s, factored_data[n][i+j]) for j in range(1, self.num_options)]).clamp(0,1)
				name_counts[name] += 1
	training_probability = {n:name_counts[n] / sum(name_counts.values()) for n in self.names}
	for n in self.names:
		counterfactual_component_probabilities[n] = counterfactual_component_masks[n] / (outcome_rollouts.length // self.num_options)
	return training_probability, counterfactual_component_masks