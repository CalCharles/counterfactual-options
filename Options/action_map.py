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


class ActionMap():
    def __init__(self, args):
        # input variables
        self.discrete_actions = args.discrete_actions # 0 if continuous, also 0 if discretize_acts is used
        self.discretize_acts = args.discretize_acts # none if not used
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
            self.policy_action_shape = self.next_option.control_max.shape
            self.policy_min = -1 * np.ones(self.policy_action_shape) # policy space is always the same
            self.policy_max = 1 * np.ones(self.policy_action_shape)
            self.policy_action_space = gym.spaces.Box(self.policy_min, self.policy_max)
        if self.discretize_acts:
            self.discrete_dict = discretize_space(self.next_option.control_max.shape)
            self.policy_action_shape = (1,)
            self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))

        # set mapped space
        if self.discrete_actions:
            self.mapped_action_shape = (1,)
            self.action_space = gym.spaces.Discrete(args.num_actions)
        else:
            self.mapped_action_shape = self.next_option.control_max.shape
            self.mapped_action_max = np.array(self.next_option.control_max)
            self.mapped_action_min = np.array(self.next_option.control_min)
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
        # print("act", act)
        act = copy.deepcopy(act)
        base = self.state_extractor.get_action(state)
        return base - act

    def map_action(self, act, batch):
        ''' 
        maps a policy space to the action space used by a parameter
        '''
        act = to_numpy(act)
        if len(act.shape) == 0: # exploration noise requires "len"  
            act = np.array([act])
            act = self._exploration_noise(act, batch)
            act = act.squeeze()
        else: 
            act = self._exploration_noise(act, batch)            
        mapped_act = act
        if self.discretize_acts: # if actions are discretized, then converts discrete action to continuous
            mapped_act = self._get_cont(mapped_act)
        mapped_act = self._policy_map_action(mapped_act) # usually converts from policy space to environment space (even for options)
        if self.relative_actions > 0:
            mapped_act = self._convert_relative_action(batch.full_state, mapped_act)
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







#         # old action space stuff below
#         # if policy_reward is not None:
#         #     self.assign_policy_reward(policy_reward)
#         # else:
#         #     self.discrete_actions = False
#         # if discretize_acts: # forces discrete actions
#         #     self.discrete_actions = True 
#         # self.object_name = object_name
#         # print("init option", self.object_name)
#         # self.action_shape = (1,) # should be set in subclass
#         # self.action_prob_shape = (1,) # should be set in subclass
#         # self.output_prob_shape = (1,) # set in subclass
#         # self.control_max = None # set in subclass, maximum values for the parameter
#         # self.action_max = None # set in subclass, the limits for actions that can be taken
#         # self.action_space = None # set in subclass, the space object corresponding to all of the above information
#         # self.relative_action_space = None # set in subclass, space object used to set actions relative to current position
#         # self.relative_actions = relative_actions > 0
#         # self.relative_state = relative_state
#         # self.relative_param = relative_param
#         # self.range_limiter = relative_actions
#         # self.discretize_actions = discretize_acts # convert a continuous state into a discrete one

#             act = self.exploration_noise(act, self.data)
#             if self.option.discrete_actions: act = act.squeeze()


# primitive action information
#         self.num_params = models[1].environment.num_actions
#         self.object_name = "Action"
#         self.action_featurizer = action_featurizer
#         environment = models[1].environment
#         self.action_space = models[1].environment.action_space
#         self.action_shape = environment.action_space.shape or (1,)
#         self.action_prob_shape = environment.action_space.shape or (1,)
#         self.output_prob_shape = environment.action_space.shape or environment.action_space.n# (models[1].environment.num_actions, )
#         print(self.action_shape[0])
#         self.action_mask = np.ones(self.action_shape)
#         self.discrete = self.action_shape[0] == 1
#         self.discrete_actions = models[1].environment.discrete_actions
#         self.control_max = environment.action_space.n if self.discrete_actions else environment.action_space.high
#         self.control_min = None if self.discrete_actions else environment.action_space.low
#         self.action_max = environment.action_space.shape or environment.action_space.n


#     raw option action information
#         self.object_name = "Raw"
#         self.action_shape = (1,)
#         self.action_prob_shape = (self.environment_model.environment.num_actions,)
#         self.discrete_actions = self.environment_model.environment.discrete_actions
#         self.action_max = self.environment_model.environment.action_space.n if self.discrete_actions else self.environment_model.environment.action_space.high
#         self.action_space = self.environment_model.environment.action_space
#         self.control_max = 0 # could put in "true" parameter, unused otherwise
#         self.discrete = False # This should not be used, since rawoption is not performing parameterized RL
#         self.use_mask = False
#         self.stack = torch.zeros((4,84,84))
#         # print("frame", self.environment_model.environment.get_state()['Frame'].shape)
#         # self.param = self.environment_model.get_param(self.environment_model.environment.get_state()[1])
#         self.param = self.environment_model.get_param(self.environment_model.environment.get_state())

#     # The definition of this function has changed
#     def get_action(self, action, mean, variance):
#         idx = action
#         return mean[torch.arange(mean.size(0)), idx.squeeze().long()], None#torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

# Model option action space handling
# self.action_prob_shape = self.next_option.output_prob_shape
#         if self.discrete_actions:
#             self.action_shape = (1,)
#         else:
#             self.action_shape = self.next_option.output_prob_shape
#         print(self.next_option.control_max)
#         self.action_max = np.array(self.next_option.control_max)
#         self.action_min = np.array(self.next_option.control_min)
#         self.control_max = np.array(self.dataset_model.control_max)
#         self.control_min = np.array(self.dataset_model.control_min)
#         self.policy_min = -1 * np.ones(self.action_min.shape) # policy space is always the same
#         self.policy_max = 1 * np.ones(self.action_min.shape)
#         self.expand_policy_space = False
#         if self.next_option.object_name != "Action":
#             self.expand_policy_space = True

#         # if we are converting the space to be discrete. If discretize_acts is a dict, use it directly
#         if type(discretize_acts) == dict:
#             self.discrete_dict = discretize_acts 
#         elif discretize_acts:
#             self.discrete_dict = discretize_actions(self.action_min.shape)

#         if type(discretize_acts) == dict:
#             acts = np.stack([v for v in self.discrete_dict.values()], axis = 0)
#             self.action_min = np.min(acts, axis=0)
#             self.action_max = np.max(acts, axis=0)
#             self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))
#             self.policy_action_shape = (1,)
#             self.action_shape = self.action_min.shape
#             self.action_space = gym.spaces.Box(self.action_min, self.action_max)
#             self.relative_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys()))) # This should not be used
#         else:
#             if self.discrete_actions and discretize_acts:
#                 self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))
#             if self.discrete_actions and not discretize_acts:
#                 self.policy_action_space = gym.spaces.Discrete(self.next_option.control_max)
#                 self.action_space = gym.spaces.Discrete(self.next_option.control_max)
#                 self.relative_action_space = gym.spaces.Discrete(self.next_option.control_max) # no relative actions for discrete
#             if (self.discrete_actions and discretize_acts) or not self.discrete_actions:
#                 self.action_space = gym.spaces.Box(self.action_min, self.action_max)
#                 rng = self.action_max - self.action_min
#                 self.relative_action_space = gym.spaces.Box(-rng / self.range_limiter, rng / self.range_limiter)
#                 print(self.action_min, self.action_max)
#             if not self.discrete_actions:
#                 self.policy_action_space = gym.spaces.Box(self.policy_min, self.policy_max)
#             self.policy_action_shape = self.policy_min.shape
#         self.last_action = np.zeros(self.action_shape)
#         if self.discrete_actions:
#             self.last_action = np.zeros(self.action_shape)[0]
#         self.last_act = np.zeros(self.policy_action_shape)
#         print(self.last_action, self.last_act, self.discrete_actions, self.action_shape, self.action_min, self.control_min, self.policy_min)
        
#                 self.output_prob_shape = (self.dataset_model.delta.output_size(), ) # continuous, so the size will match
#         # TODO: fix this so that the output size is equal to the nonzero elements of the self.dataset_model.selection_binary() at each level

#     # action manager handles all action conversions
#     def get_cont(self, act):
#         if self.discrete_dict:
#             if type(act) == np.ndarray:
#                 return np.array([self.discrete_dict[a].copy() for a in act])
#             return self.discrete_dict[act].copy()

#     def get_discrete(self, act):
#         def find_closest(a):
#             closest = (-1, 99999999)
#             for i in range(len(list(self.discrete_dict.keys()))):
#                 dist = np.linalg.norm(a - np.array(self.discrete_dict[i]))
#                 if dist < closest[1]:
#                     closest = (i,dist)
#             return closest[0]
#         if self.discretize_actions:
#             if type(act) == np.ndarray and len(act.shape) > 1:
#                 return np.array([find_closest(a) for a in act])
#             return find_closest(act)

#     def get_action(self, action, mean, variance):
#         if self.discrete_actions:
#             return mean[torch.arange(mean.size(0)), action.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), action.squeeze().long()])
#         idx = action
#         dist = torch.distributions.normal.Normal # TODO: hardcoded action distribution as diagonal gaussian
#         log_probs = dist(mean, variance).log_probs(action)
#         return torch.exp(log_probs), log_probs