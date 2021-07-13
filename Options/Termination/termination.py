# termination conditions
import numpy as np
import os, cv2, time, torch
from ReinforcementLearning.Policy.policy import pytorch_model

def invert_mask(mask):
	inverse_mask = mask.copy()
	inverse_mask[mask == 0] = -1
	inverse_mask[mask == 1] = 0
	inverse_mask *= -1
	return inverse_mask

def compute_instance_indexes(instanced, param, inverse_mask, multi=-1):
	# multi returns a single state if -1, or any state within range multi otherwise
	if len(instanced.shape) == 3: # batch of values
		remasked = list()
		if len(param.shape) == 2:
			for par, ins in zip(param, instanced):
				remasked.append((ins - par) * inverse_mask)
		else:
			for ins in instanced:
				remasked.append((ins - param) * inverse_mask)
		remasked = np.stack(remasked, axis=0)
		diff = np.sum(np.abs(remasked), axis=2) # batch x obj
		if multi < 0:
			idxes = np.argmin(diff, axis=1)
			idxes = idxes
		else:
			# binarize diff
			diff[diff <= multi] = 1
			diff[diff > multi] = 0
			idxes = diff.nonzero()
		# print(diff, idxes, remasked.shape, instanced.shape, param.shape)
	else:
		remasked = (instanced - param) * inverse_mask
		diff = np.sum(np.abs(remasked), axis=1)
		if multi < 0:
			idxes = np.argmin(diff, axis=0)
			idxes = idxes
		else:
			# binarize diff
			diff[diff <= multi] = 1
			diff[diff > multi] = 0
			idxes = diff.nonzero()
			idxes = ([0], idxes[0])
	# print(idxes)
	return idxes
		# print(instanced[idxes], param)


class Termination():
	def __init__(self, **kwargs):
		pass

	def check(self, input_state, state, param, mask, true_done=0):
		return True

class ParameterizedStateTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.epsilon = kwargs['epsilon']
		self.name = kwargs['name']
		self.discrete = True #kwargs['discrete']
		# in the discrete parameter space case
		self.dataset_model = kwargs['dataset_model']
		self.min_use = kwargs['min_use']
		kwargs['use_diff'] = self.dataset_model.predict_dynamics
		kwargs['use_both'] = 1 if kwargs['use_diff'] else 0
		# self.assign_parameters(self.dataset_model) # this line is commented out because we aren't using a stored parameter list
		print(self.name)

	def assign_parameters(self, dataset_model):
		def assign_dict(observed, totals):
			new_observed = list()
			new_totals = list()
			for i in range(len(observed)):
				if totals[i] >= self.min_use and observed[i][1].sum() > 0: # should have a minimum number, and a nonzero mask
					new_observed.append(observed[i])
					new_totals.append(totals[i])
			return new_observed, new_totals
		if self.use_both:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_both[self.name], dataset_model.both_counts[self.name])
			dataset_model.observed_both[self.name], dataset_model.both_counts[self.name] = self.discrete_parameters, self.counts 
		elif self.use_diff:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name])
			# print(dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name], self.discrete_parameters)
			dataset_model.observed_differences[self.name], dataset_model.difference_counts[self.name] = self.discrete_parameters, self.counts 
		else:
			self.discrete_parameters, self.counts = assign_dict(dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name])
			dataset_model.observed_outcomes[self.name], dataset_model.outcome_counts[self.name] = self.discrete_parameters, self.counts 

	def check(self, input_state, state, param, mask, true_done=0): # handling diff/both outside
		# NOTE: input state is from the current state, state, param are from the next state
		# param = self.convert_param(param)
		# if self.use_both:
			# if len(diff.shape) == 1:
			# 	s = torch.cat((state, diff), dim=0)
			# 	return (s - param).norm(p=1) <= self.epsilon
			# else:
				# s = torch.cat((state, diff), dim=1)
				# return (s - param).norm(p=1, dim=1) <= self.epsilon
		# elif self.use_diff:
		# 	if len(diff.shape) == 1:
		# 		return (diff - param).norm(p=1) <= self.epsilon
		# 	else:
		# 		return (diff - param).norm(p=1, dim=1) <= self.epsilon
		# else:
		if len(state.shape) == 1:
			return np.linalg.norm((state - param) * mask, ord  = 1) <= self.epsilon
		else:
			return np.linalg.norm((state - param) * mask, ord =1, axis=1 ) <= self.epsilon

class InteractionTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.interaction_model = kwargs["dataset_model"]
		self.epsilon = kwargs["epsilon"]

	def check(self, input_state, state, param, mask, true_done=0):
		# NOTE: input state is from the current state, state, param are from the next state
		interaction_pred = self.interaction_model(pytorch_model.wrap(input_state))
		return interaction_pred > 1 - self.epsilon

class CombinedTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dataset_model = kwargs["dataset_model"]
		self.epsilon = self.dataset_model.interaction_prediction
		self.epsilon_close = kwargs["epsilon"]
		self.interaction_model = self.dataset_model.interaction_model
		self.parameterized_termination = ParameterizedStateTermination(**kwargs)
		self.interaction_probability = kwargs["interaction_probability"]
		self.param_interaction = kwargs["param_interaction"]
		self.inter = 0
		self.inter_pred = 0
		self.p_hit = 0 # if the param is hit

	def check(self, inter, state, param, mask, true_done=0):
		# NOTE: input state is from the current state, state, param are from the next state
		# terminates if the parameter matches and interaction is true
		# has some probability of terminating if interaction is true
		# print("second", param, state)
		# interaction_pred = pytorch_model.unwrap(self.interaction_model(pytorch_model.wrap(input_state, cuda=self.interaction_model.iscuda)).squeeze())
		# print(interaction_pred, input_state)
		self.inter_pred = inter
		inter = self.inter_pred > (1 - self.epsilon)
		self.inter = inter
		self.p_hit = 0
		param_term = self.parameterized_termination.check(inter, state, param, mask)
		# print(self.interaction_probability, param_term, pytorch_model.unwrap(inter), state, param)
		if self.interaction_probability > 0:
			chances = np.random.random(size=self.inter_pred.shape) > self.interaction_probability
			if not self.param_interaction: param_inter = True
			else: param_inter = inter
			self.p_hit = param_term * param_inter
			# if np.sum(inter):
			# 	print("checked", inter, self.epsilon, input_state)
			chosen = inter * chances + param_term * param_inter
			# print(chances, inter, param_term, chosen)
			if type(chosen) == np.ndarray:
				chosen[chosen > 1] = 1
			# print(pytorch_model.unwrap(chosen), pytorch_model.unwrap(inter), pytorch_model.unwrap(interaction_pred), pytorch_model.unwrap(chances), input_state)
			# error
			return pytorch_model.unwrap(chosen)
		# print(inter, param_term, state, (state - param), self.parameterized_termination.epsilon)
		return pytorch_model.unwrap(inter) * param_term

class CombinedTrueTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.comb_terminator = CombinedTermination(**kwargs)
		self.inter = 0
		self.inter_pred = 0
		self.p_hit = 0
		self.epsilon = self.comb_terminator.epsilon
		self.epsilon_close = self.comb_terminator.epsilon_close

	def check(self, input_state, state, param, mask, true_done=False):
		comb_term = self.comb_terminator.check(input_state, state, param, mask, true_done=true_done)
		self.inter = self.comb_terminator.inter
		self.inter_pred = self.comb_terminator.inter_pred
		self.p_hit = self.comb_terminator.p_hit
		vals = comb_term + true_done
		if type(vals) == np.ndarray:
			vals[vals > 1] = 1
		# print(vals, comb_term, self.inter, self.inter_pred, self.p_hit)
		return vals

class InstancedTermination(Termination):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		# self.mask = kwargs["mask"]
		# self.inverse_mask = mask.copy()
		# self.inverse_mask[mask == 0] = -1
		# self.inverse_mask[mask == 1] = 0
		# self.inverse_mask *= -1
		self.terminator = terminal_forms[kwargs["terminal_type"][4:]](**kwargs)
		self.dataset_model = kwargs["dataset_model"]
		self.inter = self.terminator.inter

	def check(self, input_state, state, param, mask, true_done=0):
		'''
		finds out of the desired instance now has the desired value,
		Finds the instance by matching the values of the instance NOT masked (inverse mask)
		'''
		instanced = self.dataset_model.split_instances(state)
		inverse_mask = invert_mask(mask)
		indexes = compute_instance_indexes(instanced, param, inverse_mask, multi=-1)
		output = self.terminator.check(input_state, instanced[idxes], param, mask, true_done)
		self.inter = self.terminator.inter
		return output

class ProximityInstancedTermination(Termination):
	def __init__(self, **kwargs): # TODO: for now, operates under fixed mask assumption
		# self.mask = kwargs["mask"]
		# self.inverse_mask = mask.copy()
		# self.inverse_mask[mask == 0] = -1
		# self.inverse_mask[mask == 1] = 0
		# self.inverse_mask *= -1
		self.terminator = ParameterizedStateTermination(**kwargs)
		self.dataset_model = kwargs["dataset_model"]
		self.inter = 0 # this method performs interaction checking, but does not use it
		self.max_distance_epsilon = max(kwargs["max_distance_epsilon"], 1)
		self.epsilon = kwargs['epsilon']


	def check(self, input_state, state, param, mask, true_done=0):
		instanced = self.dataset_model.split_instances(state)
		inverse_mask = invert_mask(mask)
		batch_idx, inst_idx = compute_instance_indexes(instanced, param, inverse_mask, multi=self.max_distance_epsilon)
		inter_bin = pytorch_model.unwrap(self.dataset_model.interaction_model.instance_labels(pytorch_model.wrap(input_state, cuda=self.dataset_model.iscuda)))
		inter_bin[inter_bin < self.dataset_model.interaction_prediction] = 0
		inter_bin[inter_bin >= self.dataset_model.interaction_prediction] = 1
		# inter_bin = inter_bin.nonzero()
		batched = False
		if len(instanced.shape) == 3:
			batched = True
		if batched:
			ctr = 0
			output =list()
			for i in range(instanced.shape[0]):
				# if len(batch_idx) - 1 < ctr: # No close indexes left, should not be possible with blocks since they don't move
				# 	output.append(False)
				if np.sum(inter_bin[i]) == 0 or len(batch_idx) == 0: # there are no interactions for this instance in the batch
					output.append(False)
				else: 
					bi, ii = batch_idx[ctr], inst_idx[ctr]
					if i < bi:
						output.append(False)
						continue
					curr_output = list()
					while bi == i:
						# print(inter_bin.shape, i, ii)
						if inter_bin[i, ii] == 1:
						# print(batch_idx, inst_idx, bi, ii, instanced[bi, ii], param, mask)
							curr_output.append(np.linalg.norm((instanced[bi, ii] - param) * mask, ord =1) <= self.epsilon)
						if ctr + 1 >= len(batch_idx):
							break
						ctr += 1
						bi, ii = batch_idx[ctr], inst_idx[ctr]
					outf = np.max(curr_output) if len(curr_output) > 0 else False
					output.append(outf)
			output = np.stack(output, axis=0)
		else:
			inter_bin = inter_bin.squeeze()
			output = list()
			if np.sum(inter_bin) == 0 or len(batch_idx) == 0: # no interactions
				output = False
			else:
				for idx in inst_idx:
					# print(inter_bin.shape, idx)
					if inter_bin[idx] == 1:
						print("done", mask, instanced[idx], param, np.linalg.norm((instanced[idx] - param) * mask, ord =1), self.epsilon, np.linalg.norm((instanced[idx] - param) * mask, ord =1) <= self.epsilon)
						output.append(np.linalg.norm((instanced[idx] - param) * mask, ord =1) <= self.epsilon)
				output = np.max(output) if len(output) > 0 else False
		self.inter = pytorch_model.unwrap(self.dataset_model.interaction_model(input_state) > self.dataset_model.interaction_prediction)
		self.p_hit = output
		# if not batched:
		# 	print("termination", output, self.inter)
		return output


class TrueTermination(Termination):
	def check(self, input_state, state, param, mask, true_done=0):
		return true_done

class EnvFnTermination(Termination):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.term_fn = kwargs["environment"].check_done

	def check(self, input_state, state, param, mask, true_done=0):
		return self.term_fn(input_state, state, param)

terminal_forms = {'param': ParameterizedStateTermination, 'comb': CombinedTermination, 'tcomb': CombinedTrueTermination, 'inst': InstancedTermination, 'proxist': ProximityInstancedTermination, 'true': TrueTermination, 'env': EnvFnTermination}