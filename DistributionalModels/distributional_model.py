import numpy as np
import os, cv2, time

class DistributionalModel():
	def __init__(self, **kwargs):
		'''
		conditionally models the distribution of some output variable from some input variable. This can be used to sample different
		outcomes for a reward function, predict the outcome distribution for different states
		@attr has_relative indicates if the model also models the input from some output
		@attr outcome is the outcome distribution 
		'''
		self.has_relative = False
		self.outcome = None
		self.num_options = kwargs["option_level"].num_options

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		'''
		handles the conditional component of the rollout internally
		'''
		pass

	def predict(self, rollouts):
		'''
		predicts for a number of flattened factored states the next state distribution for each input 
		'''
		return

	def sample(self, state):
		'''
		samples an outcome, possibily conditioned on the current state, though it could just be a value
		'''