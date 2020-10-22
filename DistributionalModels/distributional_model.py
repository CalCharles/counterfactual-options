import numpy as np
import os, cv2, time
from file_management import save_to_pickle, load_from_pickle


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
		self.num_params = kwargs["option_node"].num_params
		self.option_name = kwargs["option_node"].name

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
		return

class HypothesisModel():
	def __init__(self, **kwargs):
		self.option_name = kwargs["option_node"].name
		self.instanced = kwargs["instanced"]
		self.environment_model = kwargs["environment_model"]
        self.unflatten = kwargs["environment_model"].unflatten_state
		# self.num_objects = kwargs["environment_model"].get_num(instanced=self.instanced)

	def train(self, rollouts):
		'''
		trains the necessary components
		'''
		pass

	def predict(self, rollouts):
		'''
		predicts the next state from the factored state, as well as the passive and active components
		'''

	def sample(self, all=False):
		'''
		sample a hypothesis from the set of hypotheses, if @param all is true, returns all of the hypotheses
		'''

def load_factored_model(pth): 
    model = load_from_pickle(os.path.join(pth, "dataset_model.pkl"))
    model.load()
    # print(model.observed_differences, model.difference_counts)
    return model
