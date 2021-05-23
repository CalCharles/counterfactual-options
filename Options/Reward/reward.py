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
		# NOTE: input state is from the current state, state, param are from the next state
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
		if len(state.shape) == 1:
			return (np.linalg.norm(state - param, ord = 1) <= self.epsilon).astype(float)
		else:
			return (np.linalg.norm(state - param, ord = 1, axis=1) <= self.epsilon).astype(float)

class InteractionReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_model = kwargs["interaction_model"]

	def get_reward(self, input_state, state, param, true_reward=0):
		# NOTE: input state is from the current state, state, param are from the next state
		# return (self.interaction_model(input_state) - 1).squeeze()
		# print(pytorch_model.unwrap(self.interaction_model(input_state)), pytorch_model.unwrap(input_state))
		return (pytorch_model.unwrap(self.interaction_model(input_state))).squeeze()

class TrueNegativeCombinedReward(Reward):
	def __init__(self, **kwargs):
		# only gives the negative component of the true reward
		super().__init__(**kwargs)
		self.combined = CombinedReward(**kwargs)
		self.rlambda = kwargs["true_reward_lambda"]

	def get_reward(self, input_state, state, param, true_reward=0):
		# print(self.rlambda * true_reward * float(true_reward < 0))
		if type(true_reward) == np.ndarray:
			new_true = true_reward.copy()
			new_true[true_reward > 0] = 0
		else:
			new_true = 0 if true_reward > 0 else 1
		return self.combined.get_reward(input_state, state, param, true_reward) + self.rlambda * np.squeeze(true_reward * new_true)

class CombinedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_reward = InteractionReward(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.plmbda = kwargs["parameterized_lambda"]
		self.lmbda = kwargs["interaction_lambda"]
		self.reward_constant = kwargs["reward_constant"]

	def get_reward(self, input_state, state, param, true_reward=0):
		# NOTE: input state is from the current state, state, param are from the next state
		ireward = self.interaction_reward.get_reward(input_state, state, param)
		preward = self.parameterized_reward.get_reward(input_state, state, param)
		interacting = pytorch_model.unwrap(self.interaction_reward.interaction_model(input_state))
		# interaction_reward = (ireward > self.interaction_probability).float() - 1 # only give parameterized reward at interactions
		
		# print(ireward * self.lmbda, preward, interaction_reward, )
		# print(ireward, preward, interacting)
		return pytorch_model.unwrap(ireward * self.lmbda + preward * interacting.squeeze() * self.plmbda + self.reward_constant)

class TrueReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_reward(self, input_state, state, param, true_reward=0):
		return true_reward

class EnvFnReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reward_fn = kwargs["env"].check_reward

	def get_reward(self, input_state, state, param, true_reward=0):
		return self.reward_fn(input_state, state, param)


reward_forms = {'bin': BinaryParameterizedReward, 'int': InteractionReward, 'comb': CombinedReward, 
				'true': TrueReward, 'env': EnvFnReward, 'tcomb': TrueNegativeCombinedReward}