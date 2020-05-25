import numpy as np
import os, cv2, time
import sklearn as sk
import sklearn.mixture as mix
import torch

class ClassifiedOutputModel(DistributionalModel):
	def __init__(self, **kwargs):
		'''
		wraps around an output model and perform classification between the non-outcome states and the outcome states, to typify the single
		state outcomes when they occur. if single state outcomes look the same as a large number of outcome states, this will be a problem.
		TODO: non-counterfactual rollouts empty
		'''
		super().__init__(**kwargs)
        self.classification_model = kwargs["class_model"](**kwargs)
        self.model = kwargs["output_model"](**kwargs)

	def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
		non_cf_data = torch.cat((non_counterfactual_rollouts.vals["state"], non_counterfactual_rollouts.vals["state_diff"]), dim = 1)
		data = torch.cat((outcome_rollouts.vals["state"], outcome_rollouts.vals["state_diff"]), dim = 1)
		labels = torch.cat((torch.zeros(non_cf_data.shape[0]), torch.ones(data.shape[0])))
		class_data = torch.cat((non_cf_data, data), dim=0)
		self.classification_model.fit(class_data, labels)
		return self.model.fit(counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts)
		

	def predict(self, rollouts):
		data = torch.cat((outcome_rollouts.vals["state"], outcome_rollouts.vals["state_diff"]), dim = 1)
		labels = self.classification_model.classify(data)
		modes = self.model.predict(rollouts)
		modes += 1
		modes[labels==0] = 0
		return modes
