import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import copy, os, cv2
from file_management import default_value_arg
from Networks.network import Network, pytorch_model
from Networks.tianshou_networks import networks
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
cActor, cCritic = Actor, Critic
from tianshou.utils.net.discrete import Actor, Critic
dActor, dCritic = Actor, Critic
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
import tianshou as ts
import gym
from typing import Any, Dict, Tuple, Union, Optional, Callable
from ReinforcementLearning.LearningAlgorithm.iterated_supervised_learner import IteratedSupervisedPolicy
from ReinforcementLearning.LearningAlgorithm.HER import HER
from Rollouts.rollouts import ObjDict


class LoadedPolicy(nn.Module):
    '''
    loads a network from test_network
    '''
    def __init__(self, **kwargs):
        super().__init__()
        args = ObjDict(kwargs)

        self.network = torch.load(args.load_network)
        self.network_dir = args.load_network
        self.dist_fn = torch.distributions.Categorical

    def cpu(self):
        super().cpu()
        self.assign_device("cpu")

    def cuda(self, device=None):
        super().cuda()
        if device is not None:
            self.assign_device(device)


    def assign_device(self, device):
        '''
        Tianshou stores the device on a variable inside the internal models. This must be pudated when changing CUDA/CPU devices
        '''
        if type(device) == int:
            device = 'cuda:' + str(device)
        self.network.device = device

    def set_eps(self, epsilon): # not all algo policies have set eps
        self.epsilon = epsilon

    def save(self, pth, name):
        torch.save(self, os.path.join(pth, name + ".pt"))

    def compute_Q(
        self, batch: Batch, nxt: bool
    ) -> torch.Tensor:
        comp = batch.obs_next if nxt else batch.obs
        comp = pytorch_model.wrap(comp)
        Q_val = pytorch_model.unwrap(self.network(comp))
        return Q_val

    def add_param(self, batch, indices = None):
        orig_obs, orig_next = None, None
        if self.parameterized:
            orig_obs, orig_next = batch.obs, batch.obs_next
            if self.param_process is None:
                param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
            else:
                param_process = self.param_process
            if indices is None:
                batch['obs'] = param_process(batch['obs'], batch['param'])
                if type(batch['obs_next']) == np.ndarray: batch['obs_next'] = param_process(batch['obs_next'], batch['param']) # relies on batch defaulting to Batch, and np.ndarray for all other state representations
            else: # indices indicates that it is handling a buffer
                batch.obs[indices] = param_process(batch.obs[indices], batch.param[indices])
                if type(batch.obs_next[indices]) == np.ndarray: batch.obs_next[indices] = param_process(batch.obs_next[indices], batch.param[indices])                
                # print(batch.obs[indices].shape, batch.obs_next.shape)
        return orig_obs, orig_next

    def restore_obs(self, batch, orig_obs, orig_next):
        if self.parameterized:
            batch['obs'], batch['obs_next'] = orig_obs, orig_next

    def restore_buffer(self, buffer, orig_obs, orig_next, rew, done, idices):
        if self.parameterized:
            buffer.obs[idices], buffer.obs_next[idices], buffer.rew[idices], buffer.done[idices] = orig_obs, orig_next, rew, done

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """COPIED FROM BASE: Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
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
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def reverse_map_action(self, mapped_act):
        # reverse the effect of map_action, not one to one because information might be lost (ignores clipping)
        if self.algo_policy.action_scaling:
            low, high = self.action_space.low, self.action_space.high
            act = ((mapped_act - low) / (high - low)) * 2 - 1
        if self.algo_policy.action_bound_method == "tanh":
            act = np.arctanh(act)
        return act

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        return act

    def reverse_map_action(self, mapped_act):
        return mapped_act

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return act

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs", **kwargs: Any):
        '''
        Matches the call for the forward of another algorithm method. Calls 
        '''
            # not cloning batch could result in issues
        # print(batch.obs.shape, batch.obs_next.shape)
        # print("input: ", batch.obs, self.use_input_norm, self.input_mean, self.input_var)
        # print("forward call")
        batch = copy.deepcopy(batch) # make sure input norm does not alter the input batch
        comp = batch.obs_next if input == "obs_next" else batch.obs
        comp = pytorch_model.wrap(comp)
        # self.apply_input_norm(batch)
        logits= self.network(comp)
        dist = self.dist_fn(logits)
        act = dist.sample()
        print("logits", logits, act, comp[0,:10])
        vals = Batch(logits=pytorch_model.unwrap(logits), act=act, state=None)
        return vals

    def compute_input_norm(self, buffer):
        if len(buffer) > 0:
            error
            avail = buffer.sample(0)[0]
            # print("trying compute", len(buffer), avail.obs.shape)
            # print(len(avail))
            if len (avail) >= 500: # need at least 500 values before applying input variance, typically this is the number of random actions
                if len(avail) > 20000: # only use the last 20k states
                    avail = avail[len(avail) - 20000:]
                self.input_var = np.sqrt(np.var(avail.obs, axis=0))
                self.input_var[self.input_var < .0001] = .0001 # to prevent divide by zero errors
                self.input_mean = np.mean(avail.obs, axis=0)
                # print("computing input norm", self.input_mean, self.input_var)
                if self.algo_name in _actor_critic + ['ppo']:
                    self.algo_policy.actor.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['sac']: 
                    self.algo_policy.critic1.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic1_old.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['ppo', 'ddpg']:
                    self.algo_policy.critic.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic_old.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['dqn']:
                    self.algo_policy.model.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.model_old.update_norm(self.input_mean, self.input_var)
                if self.algo_name in _double_critic:
                    self.algo_policy.critic2.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic2_old.preprocess.update_norm(self.input_mean, self.input_var)
                return True
        return False

    def apply_input_norm(self, batch):
        if self.use_input_norm:
            batch.update(obs=(batch.obs - self.input_mean) / self.input_var)

    def update(
            self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
        ) -> Dict[str, Any]: # does nothing and returns dictionary
        return dict() 

    def collect(self, full_batch, single_batch, skipped, added):
        return 