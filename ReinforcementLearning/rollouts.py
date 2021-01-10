import numpy as np
import os, cv2, time
import torch
from Rollouts.rollouts import Rollouts, ObjDict


def get_RL_shapes(option, environment_model):
    shapes = dict()
    if option.object_name == "Raw":
        state_size = [4,84,84]
        shapes["state_diff"], shapes["object_state"], shapes["next_object_state"] = ([environment_model.object_sizes["Action"]], [environment_model.object_sizes["Action"]], [environment_model.object_sizes["Action"]])
    else:
        state_size = [environment_model.object_sizes[option.object_name] + environment_model.object_sizes[option.next_option.object_name]]
        obj_size = environment_model.object_sizes[option.object_name]
        obj_size = option.get_state(form=1,inp=1).size(0)
        diff_size = option.get_state(form=2,inp=1).size(0)
        shapes["state_diff"], shapes["object_state"], shapes["next_object_state"] = [diff_size], [obj_size], [obj_size]
    shapes["state"], shapes["next_state"] = state_size, state_size
    shapes["action"] = option.action_shape
    shapes["true_action"] = environment_model.environment.action_shape
    shapes["probs"], shapes["Q_vals"], shapes["std"] = option.action_prob_shape, option.action_prob_shape, option.action_prob_shape # assumed Q distribution, diagonal covariance matches action probs
    shapes["value"], shapes["reward"], shapes["max_reward"], shapes["done"], shapes["returns"] = (1,), (1,), (1,), (1,), (1,)
    param_shape = [option.get_state(form=1,inp=1).size(0)]
    shapes["param"], shapes["mask"] = param_shape, param_shape 
    return shapes


class RLRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is 1 for discrete, needs to be specified for continuous
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "next_state", "object_state", "next_object_state", "state_diff", "action", 'true_action', 'probs', 'Q_vals', 'std', 'value', 'param', 'mask', 'reward', "max_reward", "returns", "done"]
        print({n: self.shapes[n] for n in self.names})
        self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})
        self.wrap = False

    def append(self, **kwargs):
        if self.filled == self.length and self.at > 0:
            self.wrap = True
        super().append(**kwargs)

    def compute_return(self, gamma, start_at, num_update, next_value, return_max = 20, return_form="value"):
        '''
        Computes the return for the rollout
        '''
        indexes = [(start_at + i) % self.length for i in range(num_update)]
        if self.wrap:
            self.values.returns[indexes] = 0
        # if start_at + num_update > self.length:
            # roll_num = max(start_at - self.length + num_update, 0)
            # self.values.returns = self.values.returns.roll(-roll_num, 0)
            # self.values.returns[-roll_num:] = 0
            # start_at = self.length - num_update
        # rewards = self.values.reward[start_at:start_at+num_update]
        rewards = self.values.reward[indexes]
        # must call reset_lists afterwards
        # last_values = (torch.arange(start=return_max-1, end = -1, step=-1).float() * torch.ones(return_max)).detach()
        last_values = torch.pow(gamma, torch.tensor(np.flip(np.arange(return_max)[::-1])).float()).detach()
        if self.iscuda:
            last_values = last_values.cuda()
        rewards = rewards.clone()
        rmax = np.arange(return_max)
        for k, (i, rew) in enumerate(zip(indexes, rewards)):
        # for i, rew in enumerate(rewards):
            # print(self.values.returns[start_at:start_at + i], last_values[-i:], rew)
            # update the last ten returns
            cutoff = 0
            if self.wrap:
                add_idxes = [(i - j) % self.length for j in range(return_max)]
                cutoff = max(0, return_max + num_update - k - self.length - 1)
                add_idxes = add_idxes[:len(add_idxes) - cutoff]
            else:
                add_idxes = [(i - j) for j in range(min(i+1, return_max))]
            # print(cutoff, self.at, self.filled, start_at, num_update, indexes, add_idxes, last_values[:len(add_idxes)], self.wrap)
            # self.values.returns[start_at:start_at + i + 1] += (torch.pow(gamma,last_values[-i-1:]) * rew).unsqueeze(1).detach()
            if len(add_idxes) > 0:
                self.values.returns[add_idxes] += (last_values[:len(add_idxes)] * rew).unsqueeze(1).detach()
        # print(rewards, self.values.returns)
        # print(self.values.returns[start_at:start_at + rewards.size(0)].shape, torch.pow(gamma,last_values[-rewards.size(0):]).shape, next_value.squeeze(), (torch.pow(gamma,last_values[-rewards.size(0):]) * next_value.squeeze()).detach().unsqueeze(0).shape)
        if return_form == "value":
            cutoff = 0
            if self.wrap:
                add_idxes = [(start_at + num_update - j - 1) % self.length for j in range(return_max)]
                cutoff = max(0, return_max - self.length)
                add_idxes = add_idxes[:len(add_idxes) - cutoff]
            else:
                add_idxes = [(start_at + num_update - j - 1) for j in range(min(start_at + num_update, return_max))]
            # print(self.wrap, add_idxes, cutoff, return_max, self.length)
            # self.values.returns[add_idxes] += (torch.pow(gamma,last_values[-len(add_idxes):]) * next_value.squeeze()).detach().unsqueeze(1)
            if len(add_idxes) > 0:
                self.values.returns[add_idxes] += (last_values[:len(add_idxes)] * next_value.squeeze()).detach().unsqueeze(1)
        # if rewards.sum() > 0:
        #     print (return_form, self.values.returns, next_value, rewards, last_values[:len(add_idxes)])