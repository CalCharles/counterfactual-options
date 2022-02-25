import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import numpy as np
from tianshou.data import Batch
import gym

from ReinforcementLearning.Policy.policy import TSPolicy
from tianshou.exploration import GaussianNoise, OUNoise
from Rollouts.rollouts import ObjDict
from Networks.network import pytorch_model


class HACPolicy(TSPolicy):
    '''
    wraps around a TianShao Base policy, but supports most of the same functions to interface with Option.option.py
    Note that TianShao Policies also support learning
    @param input shape is the shape of the input to the network
    @param paction space is a gym.space corresponding to the ENVIRONMENT action space
    @param action space is the action space of the agent
    @param max action is the maximum action for continuous
    @param discrete actions is when the action space is discrete
    kwargs includes:
        learning type, lookahead, input norm, sample merged, option, epsilon_schedule, epsilon, object dim, first_obj_dim, parameterized, grad epoch
        network args: max critic, cuda, policy type, gpu (device), hidden sizes, actor_lr, critic_lr, aggregate final, post dim, 
    '''
    def __init__(self, input_shape, paction_space, action_space, max_action, discrete_actions, **kwargs):
        args = ObjDict(kwargs)
        args.pretrain_iters = 0
        args.parameterized = True
        self.paction_space = paction_space
        super().__init__(input_shape, paction_space, action_space, max_action, discrete_actions, **args)
        print(self)
        # learning type, lookahead, input_norm
        # self.algo_name = args.learning_type # the algorithm being used
        # self.lookahead = args.lookahead # lookahead for RL, computes \sum^lookahead R + Q(s^lookahead+1)
        # self.use_input_norm = args.input_norm
        # self.input_var = np.ones(input_shape) # only if input variance and mean are used
        # self.input_mean = np.zeros(input_shape)
        # self.action_space = action_space # policy action space
        # self.epsilon_schedule = args.epsilon_schedule # if > 0, adjusts epsilon from 1->args.epsilon by exp(-steps/epsilon schedule)
        # self.epsilon_timer = 0 # timer to record steps
        # self.epsilon = 1 if self.epsilon_schedule > 0 else args.epsilon
        # self.epsilon_base = args.epsilon

    def cpu(self):
        super().cpu()
        self.assign_device("cpu")

    def cuda(self, device=None):
        super().cuda()
        if device is not None:
            self.assign_device(device)

    def select_action(self, obs):
        batch = Batch(obs=np.expand_dims(obs, 0), info={})
        policy_batch = self.algo_policy.forward(batch, None)
        act = pytorch_model.unwrap(policy_batch.act)
        mapped_act = self.map_action(act)
        # act = self.actor(batch).detach().cpu().data.numpy().flatten()
        # mapped = self.map_action(act)
        return act[0], mapped_act[0]

    def map_action(self, act):
        """COPIED FROM BASE: Map raw network output to action range in gym's env.action_space.
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.algo_policy.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)  # type: ignore
            elif self.algo_policy.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.algo_policy.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.paction_space.low, self.paction_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def reverse_map_action(self, mapped_act):
        # COPIED FROMBASE reverse the effect of map_action, not one to one because information might be lost (ignores clipping)
        act = mapped_act
        if self.algo_policy.action_scaling:
            low, high = self.paction_space.low, self.paction_space.high
            act = ((mapped_act - low) / (high - low)) * 2 - 1
        if self.algo_policy.action_bound_method == "tanh":
            act = np.arctanh(mapped_act)
        return act


    def update(self, buffer, batch_size, **kwargs):
        for i in range(self.grad_epoch):
            batch, indice = buffer.sample(batch_size)
            self.algo_policy.updating = True
            batch = self.algo_policy.process_fn(batch, buffer, indice)
            kwargs["batch_size"] = batch_size
            kwargs["repeat"] = 2
            result = self.algo_policy.learn(batch, **kwargs)
            if i == 0: cumul_losses = result
            else: 
                for k in result.keys():
                    cumul_losses[k] += result[k]
            self.algo_policy.post_process_fn(batch, buffer, indice)                
            self.algo_policy.updating = False
        return {k: cumul_losses[k] / self.grad_epoch for k in cumul_losses.keys()}