# reward functions
import numpy as np
import os, cv2, time
from Networks.network import pytorch_model

class Reward():
	def __init__(self, **kwargs):
		pass
		
	def get_reward(self, input_state, state, param, true_reward=0):
		return 1

class BinaryParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome
		# self.use_both = kwargs['use_both'] # supercedes use_diff
		self.epsilon = kwargs['epsilon']

	def get_reward(self, input_state, state, param, true_reward=0):
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

	def get_reward(self, input_state, state, param, true_reward=0):
		# return (self.interaction_model(input_state) - 1).squeeze()
		print(pytorch_model.unwrap(self.interaction_model(input_state)), pytorch_model.unwrap(input_state))
		return (self.interaction_model(input_state)).squeeze()

class CombinedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_reward = InteractionReward(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.lmbda = kwargs["parameterized_lambda"]
		self.reward_constant = kwargs["reward_constant"]

	def get_reward(self, input_state, state, param, true_reward=0):
		ireward = self.interaction_reward.get_reward(input_state, state, param)
		preward = self.parameterized_reward.get_reward(input_state, state, param)
		interacting = self.interaction_reward.interaction_model(input_state)
		# interaction_reward = (ireward > self.interaction_probability).float() - 1 # only give parameterized reward at interactions
		
		# print(ireward * self.lmbda, preward, interaction_reward, )
		# print(ireward, preward, interacting)
		return ireward * self.lmbda + preward * interacting.squeeze() + self.reward_constant

class TrueReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_reward(self, input_state, state, param, true_reward=0):
		return true_reward



reward_forms = {'bin': BinaryParameterizedReward, 'int': InteractionReward, 'comb': CombinedReward, 'true': TrueReward}