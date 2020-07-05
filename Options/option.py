import numpy as np
import os, cv2, time

class Option():
	def __init__(self, policy, termination, next_level, model, object_names, temp_ext=False):
		self.policy=policy
		self.termination = termination
		self.reward = reward # the reward function for this option
		self.next_level = next_level
		self.model = model
		self.object_names = object_names
		# parameters for temporal extension
		self.temp_ext = temp_ext
		self.last_action = None
		self.terminated = True


	def sample_action_chain(self, state, diff, param):
		'''
		Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
		also returns whether the current state is a termination condition. The option will still return an action in a termination state
		The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
		'''
		if self.temp_ext and (self.next_level is not None and not self.next_level.terminated):
			action = self.last_action
		else:
			if type(state) != torch.Tensor:
				state = self.model.flatten_factored_state(state)
			if type(diff) != torch.Tensor:
				diff = self.model.flatten_factored_state(diff)
			action = self.policy.sample_action(state, param)
		chain = [action]
		if self.next_level is not None:
			rem_chain, done, reward = self.next_level.sample_action_chain(state, diff, action)
			chain = rem_chain + chain
		done = self.termination.check(state, diff, param)
		reward = self.reward.get_reward(state, diff, param)
		self.terminated = done
		self.last_action = action
		return chain, done, reward

	def get_action_distribution(self, state, diff, param):
		'''
		gets the action probabilities, Q values, and any other statistics from the policy (inside a PolicyRollout).
		Operates on batches
		'''
		policy_rollout = self.policy.forward(state, diff, param)
		done = self.termination.check(state, diff, action)
		return policy_rollout, done

	def forward(self, state, param):
		return self.policy(state, param)

	def get_action(self, action, *args):
		'''
		depending on the way the distribution works, returns the args with the action selected
		the args could be distributions (continuous), parameters of distributions (gaussian distributions), or selections (discrete)
		'''
		pass

	def set_parameters(self, model):
		'''
		performs necessary settings for the parameter selection
		'''
		pass


class PrimitiveOption(Option): # primative discrete actions
	def sample_action_chain(self, state, diff, param): # param is an int denoting the primitive action, not protected (could send a faulty param)
		done = True
		chain = [param.max(0)[1]]
		return chain, int(done), 0

class DiscreteCounterfactualOption(Option):
	def set_parameters(self, dataset_model):
		'''
		sets the discrete distribution of options which are all the different outcomes
		'''
		self.termination.set_parameters(dataset_model)

	def get_action(self, action, *args):
		vals = []
		idx = action.max(0)[1]
		for vec in args:
			vals.append(args[:, idx])
		return vals

class ContinuousParticleCounterfactualOption(Option):
	def set_parameters(self, dataset_model):
		pass

	def get_action(self, action, *args):
		pass

class ContinuousGaussianCounterfactualOption(Option):
	def get_action(self, action, *args):
		pass

option_forms = {"discrete": DiscreteCounterfactualOption, "continuousGaussian": ContinuousGaussianCounterfactualOption, "continuousParticle": ContinuousParticleCounterfactualOption}