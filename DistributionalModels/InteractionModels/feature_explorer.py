# Feature Search Function
import numpy as np
import os, cv2, time
import torch
from collections import Counter
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts

class FeatureExplorer():
	def __init__(self, graph, controllable_feature_selectors, environment_model, model_args):
		self.cfs = controllable_feature_selectors
		self.em = environment_model
		self.model_args = model_args

	def search(self, rollouts train_args):
		# only search between entities (so that it's easier)
		gamma_size = 1
		delta_size = 1
		found = False
		gamma_tested = set()
		while not found
			for cfs in self.cfs:
				controllable_entity = cfs.feature_selector.get_entity()[0]
				if controllable_entity not in gamma_tested:
					entity_selection = self.em.create_entity_selector([controllable_entity])
					delta_tested = set()
					for name in self.em.names:
						if name != controllable_entity and name not in delta_tested:
							self.model_args.gamma = entity_selection
							self.model_args.delta = self.em.create_entity_selector([name])
							model = interaction_models[self.model_args['model_type']](self.model_args)
							train, test = rollouts.split_train_test()
							model.train(rollouts, train_args)
							if self.pass_criteria(model.assess_losses(test)):
								found = True
								gamma = entity_selection
								delta = self.model_args.delta
								break
							delta.add(name)
					gamma.add(controllable_entity)
		return model, gamma, delta


				# if cfs.feature_selector in self.gamma_tested and gamma_size == 1: 
				# 	continue
				# else: # start picking elements

