# reward functions
import numpy as np
import os, cv2, time
from Networks.network import pytorch_model
from Options.Termination.termination import invert_mask, compute_instance_indexes

class Reward():
	def __init__(self, **kwargs):
		pass
		
	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		return 1

class BinaryParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome
		# self.use_both = kwargs['use_both'] # supercedes use_diff
		self.epsilon_close = kwargs['epsilon_close']
		self.norm_p = kwargs['param_norm']

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
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
			return (np.linalg.norm((state - param) * mask, ord = 1) <= self.epsilon_close).astype(float)
		else:
			return (np.linalg.norm((state - param) * mask, ord = 1, axis=1) <= self.epsilon_close).astype(float)

class ConstantParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.plmbda = kwargs["parameterized_lambda"]
		self.reward_constant = kwargs["reward_constant"]
		self.epsilon_close = kwargs["epsilon_close"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# NOTE: input state is from the current state, state, param are from the next state
		self.parameterized_reward.epsilon_close = self.epsilon_close

		preward = self.parameterized_reward.get_reward(inter, state, param, mask)
		return pytorch_model.unwrap(self.reward_constant + preward * self.plmbda) # preward * interacting.squeeze() * self.plmbda

class NegativeParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.epsilon_close = kwargs['negative_epsilon_close']

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# hardcoded for robosuite obstacles based on naming
		i=0
		name = "Obstacle" + str(i)
		obstacles = list()
		while name in info["factored_state"]:
			obstacles.append(info["factored_state"][name])
			i += 1
			name = "Obstacle" + str(i)
		obstacles = np.array(obstacles)
		return float(np.any(np.max(np.abs(obstacles[:, :2] - state[:2]), axis=-1) <= self.epsilon_close)) # hardcoded for robosuite obstacles

class ConstantNegativePositiveParameterizedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.negative_reward = NegativeParameterizedReward(**kwargs)
		self.plmbda = kwargs["parameterized_lambda"]
		self.reward_constant = kwargs["reward_constant"]
		self.nlambda = kwargs["true_reward_lambda"]
		self.epsilon_close = kwargs["epsilon_close"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# NOTE: input state is from the current state, state, param are from the next state
		self.parameterized_reward.epsilon_close = self.epsilon_close

		preward = self.parameterized_reward.get_reward(inter, state, param, mask)
		nreward = self.negative_reward.get_reward(inter, state, param, mask, info=info)
		return self.reward_constant + preward * self.plmbda + nreward * self.nlambda # preward * interacting.squeeze() * self.plmbda


class InteractionReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self.interaction_model = kwargs["dataset_model"].interaction_model
		# self.interaction_model = kwargs["interaction_model"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# NOTE: input state is from the current state, state, param are from the next state
		# return (self.interaction_model(input_state) - 1).squeeze()
		# print(pytorch_model.unwrap(self.interaction_model(input_state)), pytorch_model.unwrap(input_state))
		return pytorch_model.unwrap(inter).squeeze()

class TrueConstantReward(Reward):
	def __init__(self, **kwargs):
		# only gives the negative component of the true reward
		super().__init__(**kwargs)
		self.reward_constant = kwargs["reward_constant"]
		self.rlambda = kwargs["true_reward_lambda"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# print(self.rlambda * true_reward * float(true_reward < 0))
		return self.reward_constant + self.rlambda * true_reward

class TrueNegativeCombinedReward(Reward):
	def __init__(self, **kwargs):
		# only gives the negative component of the true reward
		super().__init__(**kwargs)
		self.combined = CombinedReward(**kwargs)
		self.rlambda = kwargs["true_reward_lambda"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# print(self.rlambda * true_reward * float(true_reward < 0))
		if type(true_reward) == np.ndarray:
			new_true = true_reward.copy()
			new_true[true_reward > 0] = 0
		else:
			new_true = 0 if true_reward > 0 else true_reward
		return self.combined.get_reward(inter, state, param, mask, true_reward) + self.rlambda * np.squeeze(new_true)

class CombinedReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_reward = InteractionReward(**kwargs)
		self.parameterized_reward = BinaryParameterizedReward(**kwargs)
		self.plmbda = kwargs["parameterized_lambda"]
		self.interaction_prediction = kwargs["dataset_model"].interaction_prediction
		self.lmbda = kwargs["interaction_lambda"]
		self.reward_constant = kwargs["reward_constant"]
		self.epsilon_close = kwargs["epsilon_close"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# NOTE: input state is from the current state, state, param are from the next state
		self.parameterized_reward.epsilon_close = self.epsilon_close

		ireward = self.interaction_reward.get_reward(inter, state, param, mask)
		preward = self.parameterized_reward.get_reward(inter, state, param, mask)
		interacting = pytorch_model.unwrap(inter)
		preward = preward * (inter > self.interaction_prediction)
		# interaction_reward = (ireward > self.interaction_probability).float() - 1 # only give parameterized reward at interactions
		return pytorch_model.unwrap(ireward * self.lmbda + self.reward_constant + preward * self.plmbda) # preward * interacting.squeeze() * self.plmbda

class InstancedReward(Reward):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		# self.mask = kwargs["mask"]
		# self.inverse_mask = mask.copy()
		# self.inverse_mask[mask == 0] = -1
		# self.inverse_mask[mask == 1] = 0
		# self.inverse_mask *= -1
		self.dataset_model = kwargs["dataset_model"]
		self.rewarder = reward_forms[kwargs["reward_type"][4:]](**kwargs)

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		'''
		finds out of the desired instance now has the desired value,
		Finds the instance by matching the values of the instance NOT masked (inverse mask)
		'''
		instanced = self.dataset_model.split_instances(state)
		inverse_mask = invert_mask(mask)
		indexes = compute_instance_indexes(instanced, param, inverse_mask, multi=-1)
		return self.rewarder.get_reward(inter, instanced[indexes], param, mask, true_reward=true_reward)

class ParameterizedInstancedReward(Reward):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		self.dataset_model = kwargs["dataset_model"]
		self.reward_constant = kwargs["reward_constant"]
		self.plmbda = kwargs["parameterized_lambda"]
		self.epsilon_close = kwargs["epsilon_close"]

	def get_reward(self, inter, state, param, mask, true_done=0, info=None):
		instanced = self.dataset_model.split_instances(state)
		# print(instanced.shape, param, np.abs(instanced - param).sum(axis=-1), (np.abs(instanced - param).sum(axis=-1).min(axis=-1) <= self.epsilon_close).astype(float) * inter * self.plmbda + self.reward_constant)
		return (np.abs(instanced - param).sum(axis=-1).min(axis=-1) <= self.epsilon_close).astype(float) * inter * self.plmbda + self.reward_constant

class DistanceInstancedReward(Reward):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		self.dataset_model = kwargs["dataset_model"]
		self.max_distance_epsilon = kwargs["max_distance_epsilon"]
		self.plmbda = kwargs["parameterized_lambda"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		# info is full state
		if inter:
			instanced = self.dataset_model.split_instances(state)
			# TODO: hardcoded hit checking
			prev_state = self.dataset_model.split_instances(self.dataset_model.delta(info['factored_state']))
			hit_idx = np.nonzero(instanced[:,4] - prev_state[:,4])[0]
			param_pos = param[...,:2]
			rews = list()

			# super hacky way
			# print(param_pos, hit_idx)
			if len(hit_idx) == 0: # TODO: random bug where instanced=prev_state when the target is hit
				return self.plmbda
			for hi in hit_idx:
				hit = instanced[hi.squeeze(),:2].squeeze()
				dist = np.linalg.norm(param_pos-hit, ord=1) 
				# print(param_pos, hit, hi, dist)
				# print("rew", dist, (np.exp(-dist/ self.max_distance_epsilon) - .1) * self.plmbda)
				rews.append((np.exp(-dist) - .1) * self.plmbda)
			# print(rews, max(rews))
			return max(rews)
		return 0

class ProximityInstancedReward(Reward):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		# self.mask = kwargs["mask"]
		# self.inverse_mask = mask.copy()
		# self.inverse_mask[mask == 0] = -1
		# self.inverse_mask[mask == 1] = 0
		# self.inverse_mask *= -1
		self.dataset_model = kwargs["dataset_model"]
		self.max_distance_epsilon = kwargs["max_distance_epsilon"]
		self.epsilon = kwargs['epsilon']
		self.plmbda = kwargs["parameterized_lambda"]
		self.rlambda = kwargs["true_reward_lambda"]
		self.reward_constant = kwargs["reward_constant"]

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		'''
		finds out if the desired instance now has the desired value,
		Finds the instance by matching the values of the instance NOT masked (inverse mask)
		'''
		instanced = self.dataset_model.split_instances(state)
		inverse_mask = invert_mask(mask)
		# print(input_state.shape, state.shape, param.shape, mask.shape, instanced.shape, inverse_mask)
		batch_idx, inst_idx = compute_instance_indexes(instanced, param, inverse_mask, multi=self.max_distance_epsilon)
		inter_bin = inter # pytorch_model.unwrap(self.dataset_model.interaction_model.instance_labels(pytorch_model.wrap(input_state, cuda=self.dataset_model.iscuda)))
		inter_bin[inter_bin < self.dataset_model.interaction_prediction] = 0
		inter_bin[inter_bin >= self.dataset_model.interaction_prediction] = 1
		batched = False
		if len(instanced.shape) == 3:
			batched = True
		if batched:
			ctr = 0
			output =list()
			for i in range(instanced.shape[0]):
				new_true = 0 if true_reward[ctr] > 0 else 1
				neg_tr = new_true * true_reward[ctr]
				# if len(inst_idx) - 1 < ctr: # No close indexes left
				# 	output.append(self.reward_constant + neg_tr * self.rlambda)
				if np.sum(inter_bin[i]) == 0 or len(batch_idx) == 0: # there are no interactions for this instance in the batch
					output.append(self.reward_constant + neg_tr * self.rlambda)
				else: 
					bi, ii = batch_idx[ctr], inst_idx[ctr]
					if i < bi: # if no instance was found for this value of the batch, go past
						output.append(self.reward_constant + neg_tr * self.rlambda)
						continue
					curr_output = list()
					while bi == i:
						if inter_bin[i, ii] == 1:
							curr_output.append(np.exp(-np.linalg.norm((instanced[bi, ii] - param) * mask, ord =1)/2.5) * self.plmbda)
						if ctr + 1 >= len(batch_idx):
							break
						ctr += 1
						bi, ii = batch_idx[ctr], inst_idx[ctr]
					outf = np.max(curr_output) if len(curr_output) > 0 else 0
					output.append(outf * self.plmbda + self.reward_constant + neg_tr * self.rlambda)
			output = np.stack(output, axis=0)
		else:
			inter_bin = inter_bin.squeeze()
			output = list()
			# print (inst_idx)
			new_true = 0 if true_reward > 0 else 1
			neg_tr = new_true * true_reward
			# print (inst_idx)
			print(inter_bin, inter, batch_idx, inst_idx)
			if np.sum(inter_bin) == 0 or len(batch_idx) == 0: # no interactions
				output = self.reward_constant + neg_tr * self.rlambda
			else:
				for idx in inst_idx:
					if inter_bin[idx] == 1 and np.linalg.norm((instanced[idx] - param) * mask, ord =1) < self.epsilon:
						output.append(np.exp(-np.linalg.norm((instanced[idx] - param), ord =1)/2.5) * self.plmbda + neg_tr * self.rlambda)
						print("rew", instanced[idx], param, np.linalg.norm((instanced[idx] - param), ord =1), np.exp(-np.linalg.norm((instanced[idx] - param), ord =1)/2.5) * self.plmbda + neg_tr * self.rlambda)
				output = np.max(output) if len(output) > 0 else self.reward_constant + neg_tr * self.rlambda
		return output


class TrueReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		return true_reward

class TrueScaledReward(Reward):
	#scales positive rewards by "parameterized_lambda"
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.plmbda = kwargs["parameterized_lambda"]		

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		return true_reward * self.plmbda if true_reward > 0 else true_reward 


class EnvFnReward(Reward):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reward_fn = kwargs["environment"].check_reward

	def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
		return self.reward_fn(inter, state, param)


reward_forms = {'bin': BinaryParameterizedReward, 'param': ConstantParameterizedReward, 'negparam': ConstantNegativePositiveParameterizedReward, 'int': InteractionReward, 'comb': CombinedReward, 'inst': InstancedReward,
				'true': TrueReward, 'tscale': TrueScaledReward, 'env': EnvFnReward, 'tcomb': TrueNegativeCombinedReward, 'proxist': ProximityInstancedReward, 'paramist': ParameterizedInstancedReward, 'tconst': TrueConstantReward, 'dist': DistanceInstancedReward}