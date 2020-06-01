import numpy as np
import os, cv2, time

class Option():
	def __init__(self, policy, termination, next_level, model):
		if policy is not None: # Otherwise, this is a primative level option
			if policy.initialized:
				self.policy=policy
			else:
				self.initialize_policy()
		self.termination = termination
		self.next_level = next_level
		self.model = model

	def sample_action_chain(self, state):
		'''
		Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
		also returns whether the current state is a termination condition. The option will still return an action in a termination state
		'''
		if type(state) == torch.Tensor:
			action = self.policy.sample_action(state)
		else:
			action = self.policy.sample_action(self.model.flatten_factored_state(state))
		chain = [action]
		if self.next_level is not None:
			rem_chain, done = self.next_level.get_option(action).get_action_chain(state)
			chain = rem_chain + chain
		done = self.termination.check(factored_state)
		return chain, done

	def get_action_distribution(self, states):
		'''
		gets the action probabilities, Q values, and any other statistics from the policy (inside a PolicyRollout).
		Operates on batches
		'''
		policy_rollout = self.policy.forward(states)
		dones = self.termination.check(factored_state)
		return policy_rollout, dones

class PrimativeOption(Option): # primative discrete actions
	def __init__(self, policy, termination, next_level, model, iden=0):
		super().__init__(policy, termination, next_level, model)
		self.id = iden

	def sample_action_chain(self, state):
		done = True
		chain = [self.id]
		return chain, done

class OptionLayer(): # this class might get subsumed by the option graph
	def __init__(self, options):
		self.options = options
		self.num_options = len(options)
		self.action_shape = (1,) # only handles discrete options for now, since the combination would be complicated. 

	def add_option(self, option):
		self.options.append(option)
		self.num_options += 1

	def get_option(self, idx):
		return self.options[idx]
