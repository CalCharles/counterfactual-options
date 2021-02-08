import numpy as np
import os, cv2, time
import torch
from Rollouts.rollouts import Rollouts, ObjDict
from ReinforcementLearning.Policy.policy import pytorch_model

def get_RL_shapes(option, environment_model):
    shapes = dict()
    if option.object_name == "Raw":
        # state_size = [4,84,84]
        state_size = [environment_model.state_size]
        shapes["state_diff"], shapes["object_state"], shapes["next_object_state"] = ([environment_model.object_sizes["Action"]], [environment_model.object_sizes["Action"]], [environment_model.object_sizes["Action"]])
        param_shape = [environment_model.param_size]
    else:
        state_size = [option.dataset_model.gamma.output_size()] #[environment_model.object_sizes[option.object_name] + environment_model.object_sizes[option.next_option.object_name]]
        obj_size = [option.dataset_model.delta.output_size()]#environment_model.object_sizes[option.object_name]
        obj_size = option.get_state(form=1,inp=1).size(0)
        diff_size = option.get_state(form=2,inp=1).size(0)
        param_shape = [option.get_state(form=1,inp=1).size(0)]
        # print("obj_size", obj_size, diff_size)
        shapes["state_diff"], shapes["object_state"], shapes["next_object_state"] = [diff_size], [obj_size], [obj_size]
    shapes["state"], shapes["next_state"] = state_size, state_size
    shapes["action"] = option.action_shape
    shapes["true_action"] = environment_model.environment.action_shape
    shapes["probs"], shapes["Q_vals"], shapes["std"] = option.action_prob_shape, option.action_prob_shape, option.action_prob_shape # assumed Q distribution, diagonal covariance matches action probs
    shapes["value"], shapes["reward"], shapes["max_reward"], shapes["done"], shapes["true_done"], shapes["true_reward"], shapes["returns"] = (1,), (1,), (1,), (1,), (1,), (1,), (1,)
    shapes["param"], shapes["mask"] = param_shape, param_shape if option.use_mask else (1,)
    return shapes


class RLRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is 1 for discrete, needs to be specified for continuous
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "next_state", "object_state", "next_object_state", "state_diff", "action", 'true_action', 'true_done', 'true_reward', 'probs', 'Q_vals', 'std', 'value', 'param', 'mask', 'reward', "max_reward", "returns", "done"]
        # print({n: self.shapes[n] for n in self.names})
        self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})
        self.wrap = False

    def append(self, **kwargs):
        if self.filled == self.length and self.at > 0:
            self.wrap = True
        super().append(**kwargs)

    def compute_return(self, gamma, start_at, num_update, next_value, return_max = 20, return_form="value"):
        '''
        Computes the return for the rollout
        num_update is number of returns to update starting at start_at
        return_max is the maximum number of returns to update backward
        TODO: handle dones.
        '''
        # the indexes that will be used for the update, but also correcting backwards by "return_max" amount
        indexes = [(start_at + i) % self.length for i in range(num_update)]
        if self.wrap: # with wraparound, ensure that the previous indexes are not evaluated
            self.values.returns[indexes] = 0
        else:
            return_max = min(return_max, self.length)
        rewards = self.values.reward[indexes].clone()
        # dones = self.values.done[indexes].clone()

        # creates a vector which when multiplied with the reward, will give the effect of the reward return_max values into the past
        last_values = pytorch_model.wrap(torch.pow(gamma, torch.tensor(np.flip(np.arange(return_max)[::-1])).float()).detach(), cuda = self.iscuda)

        rmax = np.arange(return_max)
        for k, (i, rew) in enumerate(zip(indexes, rewards)):

            cutoff = 0
            if self.wrap:
                add_idxes = [(i - j) % self.length for j in range(return_max)] # count backwards, but loop to the beginning
            else:
                add_idxes = [(i - j) for j in range(min(i+1, return_max))]
            # stop computing returns at first dones
            nearest_done = torch.nonzero(self.values.done[add_idxes].squeeze())
            if nearest_done.shape[0] > 0:
                # print(nearest_done)
                add_idxes = add_idxes[:nearest_done[0]]
            # print(cutoff, self.at, self.filled, start_at, num_update, indexes, add_idxes, last_values[:len(add_idxes)], self.wrap)
            # self.values.returns[start_at:start_at + i + 1] += (torch.pow(gamma,last_values[-i-1:]) * rew).unsqueeze(1).detach()
            
            if len(add_idxes) > 0:
                # print(last_values, rew, add_idxes, self.values.returns)
                self.values.returns[add_idxes] += (last_values[:len(add_idxes)] * rew).unsqueeze(1).detach()
        # print(rewards, self.values.returns)
        # print(self.values.returns[start_at:start_at + rewards.size(0)].shape, torch.pow(gamma,last_values[-rewards.size(0):]).shape, next_value.squeeze(), (torch.pow(gamma,last_values[-rewards.size(0):]) * next_value.squeeze()).detach().unsqueeze(0).shape)
        # the same logic to insert the last value
        if return_form == "value":
            if self.wrap:
                add_idxes = [(start_at + num_update - j - 1) % self.length for j in range(return_max)] # count backwards, but loop to the beginning
            else:
                add_idxes = [(start_at + num_update - j - 1) for j in range(min(i+1, return_max))]

            # print(self.wrap, add_idxes, cutoff, return_max, self.length)
            # self.values.returns[add_idxes] += (torch.pow(gamma,last_values[-len(add_idxes):]) * next_value.squeeze()).detach().unsqueeze(1)
            if len(add_idxes) > 0:
                self.values.returns[add_idxes] += (last_values[:len(add_idxes)] * next_value.squeeze()).detach().unsqueeze(1)
        # if rewards.sum() > 0:
        #     print (return_form, self.values.returns, next_value, rewards, last_values[:len(add_idxes)])