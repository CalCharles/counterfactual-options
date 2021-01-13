# reward functions
import numpy as np
import os, cv2, time

class Reward():
	def __init__(self, **kwargs):
		self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome

	def get_reward(self, state, diff, param):
		return 1

class BinaryParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome
		# self.use_both = kwargs['use_both'] # supercedes use_diff
		self.epsilon = kwargs['epsilon']

	def get_reward(self, input_state, state, param):
		# if self.use_both:
		# 	if len(diff.shape) == 1 and len(param.shape) == 1:
		# 		s = torch.cat((state, diff), dim=0)
		# 		return ((s - param).norm(p=1) <= self.epsilon).float()
		# 	else:
		# 		s = torch.cat((state, diff), dim=1)
		# 		return ((s - param).norm(p=1, dim=1) <= self.epsilon).float()
		# elif self.use_diff:
		# 	if len(diff.shape) == 1 and len(param.shape) == 1:
		# 		return ((diff - param).norm(p=1) <= self.epsilon).float()
		# 	else:
		# 		return ((diff - param).norm(p=1, dim=1) <= self.epsilon).float()
		# else:
		if len(param.shape) == 1:
			return ((state - param).norm(p=1) <= self.epsilon).float()
		else:
			return ((state - param).norm(p=1, dim=1) <= self.epsilon).float()

class InteractionReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_model = kwargs["interaction_model"]

	def get_reward(self, input_state, state, param):
		return self.interaction_model(input_state) - 1

class CombinedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_reward = InteractionReward(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.lmbda = kwargs["parameterized_lambda"]
		self.interaction_probability = kwargs["interaction_minimum"]


	def get_reward(self, input_state, state, param):
		ireward = self.interaction_reward.get_reward(input_state, state, param)
		preward = self.parameterized_reward.get_reward(input_state, state, param)
		interaction_reward = 1 if ireward > self.interaction_probability - 1 else 0 # only give parameterized reward at interactions
		# print(ireward * self.lmbda, preward, interaction_reward, state, param)
		return ireward * self.lmbda + preward * interaction_reward

reward_forms = {'bin': BinaryParameterizedReward, 'int': InteractionReward, 'comb': CombinedReward}