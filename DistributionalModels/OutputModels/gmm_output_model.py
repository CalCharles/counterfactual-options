import numpy as np
import os, cv2, time
import sklearn as sk
import sklearn.mixture as mix
import torch
from collections import Counter
from DistributionalModels.distributional_model import DistributionalModel

def initialize_DP_GMM(kwargs, cov_prior, mean_prior):
	return mix.BayesianGaussianMixture(n_components=kwargs['num_components'], max_iter=kwargs['max_iter'], 
                                        weight_concentration_prior=kwargs['weight_prior'], covariance_type=kwargs['cov_form'], 
                                        covariance_prior=cov_prior, mean_prior=mean_prior) # uses a dirichlet process GMM to cluster

class GaussianMixtureOutputModel():
	def __init__(self, **kwargs):
		'''
		fits a DP-GMM to the outcome distribution. Hopefully, the primary modes of the outcome distribution will be captured
		and the useless modes will not be captured. However, since the non-counterfactual states are not included, this might not be the case
		'''
		super().__init__(**kwargs)
        cov_prior = [kwargs['cov_prior'] for _ in range(kwargs['state_dim'])]
        mean_prior = [kwargs['mean_prior'] for i in range(kwargs['state_dim'])]
        self.model = initialize_DP_GMM(kwargs, cov_prior, mean_prior)

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		data = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		self.model.fit(data)		
		

	def predict(self, rollouts):
		data = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
		return self.model.predict(data)


class FactoredGaussianMixtureOutputModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		trains a GMM on each of the objects separately, and then assigns modes based on unique combinations
		TODO: non-counterfactual rollouts empty
		'''
		super().__init__(**kwargs)
		self.unflatten = kwargs["environment_model"].unflatten_state
		self.names = kwargs["environment_model"].object_names
        self.models = {n: initialize_DP_GMM(kwargs, cov_prior, mean_prior) for n in self.names}
        self.modes = dict()
        self.num_modes = 0

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, typed=False)
		out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
		factored_data = dict()
		for name in self.names:
			factored_data[name] = torch.cat((out_state[name], out_diff[name]), dim = 1)
			self.models[name].fit(factored_data[name])

	def predict(self, rollouts):
		out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, typed=False)
		out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
		factored_data = dict()
		factored_preds = dict()
		for name in self.names:
			factored_data[name] = torch.cat((out_state, out_diff), dim = 1)
			factored_preds[name] = self.models[name].predict(data)
		tuple_predictions = []
		for name in self.names:
			predictions.append(factored_preds[name])
		tuple_predictions = torch.cat(predictions, dim=1)
		predictions = []
		for tup in predictions:
			tup = tuple(tup.data.cpu().numpy().tolist())
			if tup in self.modes:
				predictions.append(self.modes[tup])
			else:
				self.modes[tup] = self.num_modes
				self.num_modes += 1
				predictions.append(self.modes[tup])
		return torch.tensor(predictions)

class FactoredCounterfactualGaussianMixtureOutputModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		constructs datasets where only the counterfactually affected component for each object is used, and builds a likelihood mask over counterfactual cases
		relies on the outcome rollouts being sequentially added
		'''
		super().__init__(**kwargs)
		self.unflatten = kwargs["environment_model"].unflatten_state
		self.names = kwargs["environment_model"].object_names
		self.num_options = kwargs["option_level"].num_options
        self.models = {n: initialize_DP_GMM(kwargs, cov_prior, mean_prior) for n in self.names}
        self.min_probability = kwargs["min_probability"]
        self.modes = dict()
        self.num_modes = 0
        self.training_probability = None
        self.probable_names = []

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, typed=False)
		out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
		factored_data = dict()
		name_counts = Counter()
		for name in self.names:
			factored_data[name] = torch.cat((out_state[name], out_diff[name]), dim = 1)
		counterfactual_factored_data = {n: [] for n in self.names}
		EPSILON = 1e-10 # TODO: move this outside?
		for i in [k*self.num_options for k in range(outcome_rollouts.length // self.num_options)]: # TODO: assert evenly divisible
			for n in self.names:
				s = factored_data[n][i]
				state_equals = lambda x,y: (x-y).norm(p=1) < EPSILON
				if sum([int(not state_equals(s, factored_data[n][i+j])) for j in range(1, self.num_options)]):
					counterfactual_factored_data[n] += [factored_data[n][i+j] for j in range(self.num_options)]
					name_counts[name] += 1
		self.training_probability = {n:name_counts[n] / sum(name_counts.values()) for n in self.names}
		for name in self.names:
			self.models[name].fit(torch.cat(counterfactual_factored_data[name], dim=0))
		self.probable_names = [n for n in self.names if self.training_probability[n] > self.min_probability]

	def predict(self, rollouts):
		out_state = self.unflatten(outcome_rollouts.get_values("state"), vec = True, typed=False)
		out_diff = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
		factored_data = dict()
		factored_preds = dict()
		for name in self.probable_names:
			factored_data[name] = torch.cat((out_state, out_diff), dim = 1)
			factored_preds[name] = self.models[name].predict(data)
		tuple_predictions = []
		for name in self.probable_names:
			predictions.append(factored_preds[name])
		tuple_predictions = torch.cat(predictions, dim=1)
		predictions = []
		for tup in predictions:
			tup = tuple(tup.data.cpu().numpy().tolist())
			if tup in self.modes:
				predictions.append(self.modes[tup])
			else:
				self.modes[tup] = self.num_modes
				self.num_modes += 1
				predictions.append(self.modes[tup])
		return torch.tensor(predictions)
