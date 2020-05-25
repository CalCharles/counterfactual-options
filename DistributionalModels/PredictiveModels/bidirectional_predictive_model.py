import numpy as np
import os, cv2, time
import torch
from DistributionalModels.distributional_model import DistributionalModel
from DistributionalModels.PredictionModels.prediction_model import get_corresponding_first

class BidirectionalModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		Predict both the relative states that produce a particular action, and the output distribution that resulted from those states
		TODO: pretty much duplicated from the Prediction model, which suggests we can somehow merge the code
		'''
		super().__init__(**kwargs)
		self.prediction_model = kwargs["predictive_model"](**kwargs)
		self.relative_model = kwargs["relative_model"](**kwargs)
		self.flatten = kwargs["environment_model"].flatten_factored_state
		self.use_counterfactual_mask = kwargs["use_counterfactual_mask"] 

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		states, state_diff = counterfactual_rollouts.get_values("state"), counterfactual_rollouts.get_values("state_diff")
		input_states = get_corresponding_first(states, state_diff, rollouts)
		# classify out of distribution values
		states, state_diff = non_counterfactual_rollouts.get_values("state"), non_counterfactual_rollouts.get_values("state_diff")
		nc_input_states = get_corresponding_first(states, state_diff, rollouts)
		input_states = torch.cat((input_states, nc_input_states), dim = 0)
		output_states = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		output_distribution = torch.cat(torch.ones(output_states.size(0)), torch.zeros(nc_input_states.size(0)))
		output_states = torch.cat((output_states, torch.zeros((nc_input_states.size(0), output_states.size(1)))), dim=0)
		if self.use_counterfactual_mask:
			training_probability, counterfactual_component_masks = counterfactual_factored_mask(outcome_rollouts)
			counterfactual_mask = self.flatten(counterfactual_factored_mask)
			self.prediction_model.train(input_states, output_states, output_distribution, output_mask = counterfactual_masks)
			self.relative_model.train(output_states, input_states, output_distribution, input_mask = counterfactual_mask)
		else:
			self.prediction_model.train(input_states, output_states, output_distribution)
			self.relative_model.train(output_states, input_states, output_distribution)

	def predict(self, rollouts):
		data = torch.cat((rollouts.get_values("state"), rollouts.get_values("state_diff")), dim = 1)
		mu, sigma, dist = self.prediction_model.test(data)
		return dist
