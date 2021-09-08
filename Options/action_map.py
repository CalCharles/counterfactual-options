# action handler
import numpy as np
import copy
import gym
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from EnvironmentModels.environment_model import discretize_space

# Base action space handling
class PrimitiveActionMap():
    def __init__(self, args):
        self.action_space = args.environment.action_space
        self.discrete_actions = args.environment.discrete_actions
        self.discrete_control = np.arange(args.environment.action_space.n) if args.environment.discrete_actions else None
        self.control_max = args.environment.action_space.high if not args.environment.discrete_actions else args.environment.action_space.n
        self.control_min = args.environment.action_space.low if not args.environment.discrete_actions else 0

    def sample(self):
        self.action_space.sample()

    def map_action(self, act, batch):
        ''' 
        maps a policy space to the action space used by a parameter
        '''
        act = to_numpy(act)
        act = self._exploration_noise(act, batch)
        mapped_act = self._policy_map_action(act) # usually converts from policy space to environment space (even for options)
        return act, mapped_act

    def assign_policy_map(self, policy_map_action, policy_reverse_map_action, policy_exploration_noise):
        self._policy_map_action = policy_map_action # a function from option.policy
        self._policy_reverse_map_action = policy_reverse_map_action # a function from option.policy
        self._exploration_noise = policy_exploration_noise

class ActionMap():
    def __init__(self, args, next_act_map):
        # input variables
        self.discrete_actions = args.discrete_actions # 0 if continuous, also 0 if discretize_acts is used
        self.discretize_acts = args.discretize_acts# none if not used
        self.discrete_control = args.discrete_params # if the parameter is sampled from a discrete dict
        self.num_actions = args.num_actions 
        self.control_min, self.control_max = np.array(args.control_min), np.array(args.control_max) # from dataset model
        self.control_shape = self.control_min.shape
        self.relative_actions = args.relative_action # uses actions relative to the current values
        self._state_extractor = args.state_extractor

        # set policy space
        if self.discrete_actions:
            self.policy_action_shape = (1,)
            self.policy_action_space = gym.spaces.Discrete(args.num_actions)
        else:
            self.policy_action_shape = next_act_map.control_max.shape
            self.policy_min = -1 * np.ones(self.policy_action_shape) # policy space is always the same
            self.policy_max = 1 * np.ones(self.policy_action_shape)
            self.policy_action_space = gym.spaces.Box(self.policy_min, self.policy_max)
        if self.discretize_acts is not None:
            if type(self.discretize_acts) == dict: # use the hardcoded discrete dict
                self.discrete_dict = self.discretize_acts
            else:
                self.discrete_dict = discretize_space(next_act_map.control_max.shape)
            self.policy_action_shape = (1,)
            self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))

        # set mapped space
        if self.discrete_actions:
            self.mapped_action_shape = (1,)
            self.action_space = gym.spaces.Discrete(args.num_actions)
        else:
            self.mapped_action_shape = next_act_map.control_max.shape
            self.mapped_action_max = np.array(next_act_map.control_max)
            self.mapped_action_min = np.array(next_act_map.control_min)
            self.action_space = gym.spaces.Box(self.mapped_action_min, self.mapped_action_max)

    def assign_policy_map(self, policy_map_action, policy_reverse_map_action, policy_exploration_noise):
        self._policy_map_action = policy_map_action # a function from option.policy
        self._policy_reverse_map_action = policy_reverse_map_action # a function from option.policy
        self._exploration_noise = policy_exploration_noise


    def _convert_relative_action(self, state, act):
        act = copy.deepcopy(act)
        base = self._state_extractor.get_action(state)
        base += act
        act = np.clip(base, self.mapped_action_min, self.mapped_action_max)
        return act

    def _reverse_relative_action(self, state, act):
        act = copy.deepcopy(act)
        base = self._state_extractor.get_action(state)
        return base - act

    def map_action(self, act, batch):
        ''' 
        maps a policy space to the action space used by a parameter
        '''
        act = to_numpy(act)
        if len(act.shape) == 2:
            act = act[0]
        if len(act.shape) == 0: # exploration noise requires "len"  
            act = np.array([act])
            act = self._exploration_noise(act, batch)
            act = act.squeeze()
        else: 
            act = self._exploration_noise(act, batch)
        mapped_act = act
        if self.discretize_acts: # if actions are discretized, then converts discrete action to continuous
            mapped_act = self._get_cont(mapped_act)
        if self.relative_actions > 0:
            mapped_act = act * self.relative_actions
            mapped_act = self._convert_relative_action(batch.full_state, mapped_act)
        else:
            mapped_act = self._policy_map_action(mapped_act) # usually converts from policy space to environment space (even for options)
        return act, mapped_act

    def reverse_map_action(self, mapped_act, batch):
        '''
        gets a policy space action from a mapped action
        '''
        if self.relative_actions > 0: # converts relative actions maintaining value
            mapped_act = self._reverse_relative_action(batch.full_state, mapped_act)
        act = self._policy_reverse_map_action(mapped_act) # usually converts from policy space to environment space (even for options)
        if self.discretize_acts: # if actions are discretized, then converts discrete action to continuous
            act = self._get_discrete(act)
        return act


    def sample_policy_space(self):
        '''
        samples the policy action space
        '''
        return self.policy_action_space.sample()

    def sample(self):
        '''
        sample the mapped action space
        '''
        return self.action_space.sample() # maybe we should map a policy action space

    def _get_cont(self, act):
        if self.discrete_dict:
            if type(act) == np.ndarray:
                return np.array([self.discrete_dict[a].copy() for a in act])
            return self.discrete_dict[act].copy()

    def _get_discrete(self, act):
        def find_closest(a):
            closest = (-1, 99999999)
            for i in range(len(list(self.discrete_dict.keys()))):
                dist = np.linalg.norm(a - np.array(self.discrete_dict[i]))
                if dist < closest[1]:
                    closest = (i,dist)
            return closest[0]
        if self.discrete_dict:
            if type(act) == np.ndarray and len(act.shape) > 1:
                return np.array([find_closest(a) for a in act])
            return find_closest(act)

    def get_action_prob(self, action, mean, variance):
        if self.discrete_actions:
            return mean[torch.arange(mean.size(0)), action.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), action.squeeze().long()])
        idx = action
        dist = torch.distributions.normal.Normal # TODO: hardcoded action distribution as diagonal gaussian
        log_probs = dist(mean, variance).log_probs(action)
        return torch.exp(log_probs), log_probs
