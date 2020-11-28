# Feature Search Function
import numpy as np
import os, cv2, time
import torch
from collections import Counter
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts
from DistributionalModels.InteractionModels.interaction_model import interaction_models

class FeatureExplorer():
	def __init__(self, graph, controllable_feature_selectors, environment_model, model_args):
		self.cfs = controllable_feature_selectors
		self.em = environment_model
		self.model_args = model_args

	def search(self, rollouts, train_args):
		# only search between entities (so that it's easier)
		gamma_size = 1
		delta_size = 1
		found = False
		gamma_tested = set()
		self.model_args["cuda"] = train_args.cuda
		while not found:
			for cfs in self.cfs:
				controllable_entity = cfs.feature_selector.get_entity()[0]
				if controllable_entity not in gamma_tested:
					delta_tested = set()
					for name in self.em.object_names:
						if name != controllable_entity and name not in delta_tested:
							entity_selection = self.em.create_entity_selector([controllable_entity, name])
							model, test, gamma_new, delta_new = self.train(cfs.object(), rollouts, train_args, self.em.create_entity_selector([controllable_entity]), name)
							if self.pass_criteria(model.assess_losses(test)):
								found = True
								gamma = gamma_new
								delta = delta_new
								break
							entity_selection = self.em.create_entity_selector([controllable_entity])
							model, gamma_new, delta_new = self.train(self.em.create_entity_selector([controllable_entity]), name)
							if self.pass_criteria(model.assess_losses(test)):
								found = True
								gamma = gamma_new
								delta = delta_new
								break
							delta.add(name)
					gamma.add(controllable_entity)
		return model, gamma, delta

	def train(self, control_name, rollouts, train_args, entity_selection, name):
		print("Training ", control_name, "-> ", name)
		self.model_args['gamma'] = entity_selection
		self.model_args['delta'] = self.em.create_entity_selector([name])
		self.model_args['num_inputs'] = self.model_args['gamma'].output_size()
		self.model_args['num_outputs'] = self.model_args['delta'].output_size()
		model = interaction_models[self.model_args['model_type']](**self.model_args)
		train, test = rollouts.split_train_test(train_args.ratio)
		model.train(train, train_args, control_name=control_name, target_name=name)
		return model, test, self.model_args['gamma'], self.model_args['delta']


				# if cfs.feature_selector in self.gamma_tested and gamma_size == 1: 
				# 	continue
				# else: # start picking elements

