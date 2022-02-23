import torch
import numpy as np
from Baselines.HAC.HAC_param_buffer import ParamPriorityReplayBuffer
from Baselines.HAC.HAC_policy import HACPolicy
from gym import spaces

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, args, k_level, H, threshold, epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state, state_space, reduced_state_space, object_dim=0):
        '''
        k_level is number of HAC levels
        H is the time horizon
        threshold is epsilon close
        epsilon is the action epsilon
        flat_state is a full flat state
        reduced flat state is the flattened state if reduced
        '''
        # adding lowest level
        no_param_input_shape = flat_state.shape if args.keep_instanced else reduced_flat_state.shape
        param_input_shape = (no_param_input_shape[0] * 2, )
        primitive_action_space = environment.action_space
        action_space = primitive_action_space
        paction_space = primitive_action_space
        max_action = primitive_action_space.n if environment.discrete_actions else primitive_action_space.high[0]
        args.object_dim = 0
        args.first_obj_dim = 0
        args.policy_type = "basic" # must use MLP architecture except at last layer with instanced
        lt = args.learning_type
        if environment.discrete_actions: args.learning_type = "dqn" 
        self.HAC = [HACPolicy(param_input_shape[0], primitive_action_space, primitive_action_space, max_action, environment.discrete_actions, **vars(args))]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        args.learning_type = lt
        
        # adding intermediate levels
        paction_space = state_space if args.keep_instanced else reduced_state_space
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)
        max_action = 1
        print(max_action)
        for _ in range(k_level-2):
            self.HAC.append(HACPolicy(param_input_shape[0], paction_space, action_space, max_action, False, **vars(args)))
            self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        args.object_dim = object_dim
        args.first_obj_dim = reduced_flat_state.shape[0]
        print(reduced_flat_state.shape)
        args.post_dim = 0
        args.policy_type = "pair" # must use MLP architecture except at last layer with instanced
        input_shape = (flat_state.shape[0], )
        # adding last level (might not be parametereized so state dim changes)
        print(input_shape, args.object_dim, args.first_obj_dim)
        self.HAC.append(HACPolicy(input_shape[0], paction_space, action_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        self.buffer_at = [0 for i in range(k_level)]
        # set some parameters
        self.keep_instanced = not args.no_keep_instanced
        self.goal_final = goal_final
        self.k_level = k_level
        self.H = H
        self.threshold = threshold
        self.epsilon = epsilon
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0
        
    def set_parameters(self, lamda, gamma):
        self.lamda = lamda
        self.gamma = gamma

    def get_obs(self, level, full_state, param, environment_model):
        use_instanced = self.keep_instanced or level == self.k_level - 1
        inp = environment_model.get_HAC_flattened_state(full_state, use_instanced=use_instanced)
        if level != self.k_level - 1 or self.goal_final is not None:
            inp = np.concatenate([param, inp], axis=-1)
        return inp

    def get_target(self, level, full_state, environment_model):
        use_instanced = self.keep_instanced or level == self.k_level - 1
        return environment_model.get_HAC_flattened_state(full_state, use_instanced=use_instanced)

    def set_epsilon_below(self, level, value):
        if level == 0:
            self.HAC[0].set_eps(value)
        for i in range(level): 
            self.HAC[i].set_eps(value)
    
    def check_goal(self, state, goal, threshold):
        assert state.shape == goal.shape
        for i in range(state.shape[0]):
            if type(threshold) == np.ndarray:
                if abs(state[i]-goal[i]) > threshold[i]:
                    return False
            else: # it's a single value
                if abs(state[i]-goal[i]) > threshold:
                    return False
        return True
    
    def update(self, batch_size):
        losses_dict = {}
        for i in range(self.k_level):
            losses = self.HAC[i].update(self.replay_buffer[i], batch_size)
            losses_dict[i] = losses
        return losses_dict
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}.pt'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}.pt'.format(i))
