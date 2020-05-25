import numpy as np
import os, cv2, time
import torch
from DistributionalModels.distributional_model import DistributionalModel

class LatentPredictionModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		predict the output distribution from the action and the first state of the counterfactual distribution using regression, predicting
		a unimodal gaussian (subclass performs multimodal gaussian prediction)
		trains with negative log likelihood
		'''
		super().__init__(**kwargs)
		self.prediction_model = kwargs["predictive_model"](**kwargs)

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		# input_states = torch.cat((counterfactual_rollouts.get_values("state"), counterfactual_rollouts.get_values("state_diff")), dim = 1)
		# idxes = torch.cat((torch.zeros((1,)), counterfactual_rollouts.get_values("done").nonzero() + 1)).long()
		# input_states = input_states[idxes]
		# # classify out of distribution values
		# nc_input_states = torch.cat((non_counterfactual_rollouts.get_values("state"), non_counterfactual_rollouts.get_values("state_diff")), dim = 1)
		# idxes = torch.cat((torch.zeros((1,)), non_counterfactual_rollouts.get_values("done").nonzero() + 1)).long()
		# nc_input_states = nc_input_states[idxes]
		# input_states = torch.cat((input_states, nc_input_states), dim = 0)
		# output_states = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		# output_distribution = torch.cat(torch.ones(output_states.size(0)), torch.zeros(nc_input_states.size(0)))
		# output_states = torch.cat((output_states, torch.zeros((nc_input_states.size(0), output_states.size(1)))))
		# self.prediction_model.train(input_states, output_states, output_distribution)

	def predict(self, rollouts):
		# data = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		# mu, sigma, dist = self.prediction_model.test(data)
		# return dist
