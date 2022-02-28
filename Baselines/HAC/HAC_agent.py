import torch
import numpy as np
from Baselines.HAC.HAC_param_buffer import ParamPriorityReplayBuffer
from Baselines.HAC.HAC_policy import HACPolicy
from gym import spaces

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, args, k_level, H, threshold, epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state, state_space, reduced_state_space, object_dim=0, sampler=None):
        '''
        k_level is number of HAC levels
        H is the time horizon
        threshold is epsilon close
        epsilon is the action epsilon
        flat_state is a full flat state
        reduced flat state is the flattened state if reduced
        '''
        # adding lowest level
        self.top_level_random = False
        args.epsilon = 0 # we have separate exploration noise
        no_param_input_shape = flat_state.shape if args.keep_instanced else reduced_flat_state.shape
        param_input_shape = (no_param_input_shape[0] * 2, )
        primitive_action_space = environment.action_space
        action_space = primitive_action_space
        paction_space = primitive_action_space
        
        r_space = state_space if args.keep_instanced else reduced_state_space
        low = np.concatenate([r_space.low.copy(), r_space.low.copy()])
        high = np.concatenate([r_space.high.copy(), r_space.high.copy()])
        obs_space = spaces.Box(low=low, high=high)
        

        self.primitive_action_discrete = environment.discrete_actions 
        self.max_action = primitive_action_space.n if environment.discrete_actions else primitive_action_space.high[0]
        max_action = self.max_action
        args.object_dim = 0
        args.first_obj_dim = 0
        args.policy_type = "basic" # must use MLP architecture except at last layer with instanced
        lt = args.learning_type
        if environment.discrete_actions: args.learning_type = "dqn" 
        self.HAC = [HACPolicy(param_input_shape[0], primitive_action_space, primitive_action_space, obs_space, max_action, environment.discrete_actions, **vars(args))]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        args.learning_type = lt
        
        # adding intermediate levels
        paction_space = state_space if args.keep_instanced else reduced_state_space
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)
        max_action = 1
        print(max_action)
        for _ in range(k_level-2):
            self.HAC.append(HACPolicy(param_input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
            self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        args.object_dim = object_dim
        goal_final_size = goal_final.shape[0] if goal_final is not None else 0
        args.first_obj_dim = reduced_flat_state.shape[0] + goal_final_size
        print(reduced_flat_state.shape, flat_state.shape)
        args.post_dim = 0
        if args.final_instanced:
            args.policy_type = "pair" # must use MLP architecture except at last layer with instanced
            input_shape = (int(flat_state.shape[0] + goal_final_size), )
        else:
            args.policy_type = "basic"
            input_shape = flat_state.shape[0]
            input_shape = (int(input_shape + goal_final_size), )
        if goal_final is not None:
            low = np.concatenate([r_space.low.copy(), state_space.low.copy()])
            high = np.concatenate([r_space.high.copy(), state_space.high.copy()])

            obs_space = spaces.Box(low=low, high=high)
        else:
            obs_space = state_space
        # adding last level (might not be parametereized so state dim changes)
        print(input_shape, args.object_dim, args.first_obj_dim)
        self.HAC.append(HACPolicy(input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        self.buffer_at = [0 for i in range(k_level)]
        # set some parameters
        self.keep_instanced = not args.reduced_state
        self.goal_final = goal_final
        self.k_level = k_level
        self.H = H
        self.threshold = threshold
        self.epsilon = epsilon
        args.epsilon = epsilon
        self.iscuda=False
        self.sampler=sampler
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

    def cuda(self, device=None, actor_lr=1e-5,critic_lr=1e-5):
        self.iscuda=True
        for hac in self.HAC:
            hac.cuda(device=device, actor_lr=actor_lr, critic_lr=critic_lr)
            print("atcuda", hac)

    def cpu(self):
        self.iscuda=False
        for hac in self.HAC:
            hac.cpu()
        
    def set_parameters(self, lamda, gamma):
        self.lamda = lamda
        self.gamma = gamma

    def get_obs(self, level, full_state, param, environment_model):
        use_instanced = self.keep_instanced or level == self.k_level - 1
        inp = environment_model.get_HAC_flattened_state(full_state, instanced=True, use_instanced=use_instanced)
        if level != self.k_level - 1 or self.goal_final is not None:
            inp = np.concatenate([param, inp], axis=-1)
        n_inp = self.HAC[level].normalize(inp)
        # print(n_inp, inp)
        return n_inp

    def get_target(self, level, full_state, environment_model):
        use_instanced = self.keep_instanced #or level == self.k_level - 1
        if level == self.k_level and self.sampler is not None:
            return sampler.param.copy()
        return environment_model.get_HAC_flattened_state(full_state, instanced=True, use_instanced=use_instanced)

    def set_epsilon_below(self, level, value):
        if level == 0:
            self.HAC[0].set_eps(value)
        for i in range(level): 
            self.HAC[i].set_eps(value)
    
    def check_goal(self, state, goal, threshold):
        assert state.shape == goal.shape
        for i in range(state.shape[0]):
            if type(threshold) == np.ndarray:
                # print("testing", state[i], state[i] - goal[i], threshold[i])
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
            self.HAC[i] = torch.load(os.path.join(directory, name+'_level_{}.pt.pt'.format(i)))

class RobopushingHAC(HAC):
    def __init__(self, args, k_level, H, threshold, epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state, state_space, reduced_state_space, object_dim=0, sampler = None):
        '''
        k_level is number of HAC levels
        H is the time horizon
        threshold is epsilon close
        epsilon is the action epsilon
        flat_state is a full flat state
        reduced flat state is the flattened state if reduced
        '''
        # adding lowest level
        self.top_level_random = False
        self.ctrl_type = args.ctrl_type
        args.relative_action = -1
        args.epsilon = 0 # we have separate exploration noise
        no_param_input_shape = flat_state.shape if args.keep_instanced else reduced_flat_state.shape
        param_input_shape = (no_param_input_shape[0] * 2, )
        primitive_action_space = environment.action_space
        action_space = primitive_action_space
        paction_space = primitive_action_space
        
        # possible ctrl types: 3 layer gripper-block-reward, gripper-block-delta, 
        # 2 layer: block-reward, gripper-reward, block-delta, 
        r_space = state_space if args.keep_instanced else reduced_state_space
        if args.ctrl_type in ["gripper2", "block2"]:
            k_level = 2
        elif args.ctrl_type in ["limit"]:
            k_level = 3
        if args.ctrl_type in ["gripper2", "limit"]:
            low = np.array([-.2, -.21,.83,])
            high = np.array([.2, .26, .925,])
        elif args.ctrl_type in ["block2", "blockdelta"]: 
            low = np.array([-.2,-.21,.8239])
            high = np.array([.2, .26,.8241])
        else:
            low = r_space.low
            high = r_space.high
        olow = np.concatenate([low.copy(), r_space.low.copy()])
        ohigh = np.concatenate([high.copy(), r_space.high.copy()])
        obs_space = spaces.Box(low=olow, high=ohigh)
        


        self.primitive_action_discrete = environment.discrete_actions 
        self.max_action = primitive_action_space.n if environment.discrete_actions else primitive_action_space.high[0]
        max_action = self.max_action
        args.object_dim = 0
        args.first_obj_dim = 0
        args.policy_type = "basic" # must use MLP architecture except at last layer with instanced
        lt = args.learning_type
        if environment.discrete_actions: args.learning_type = "dqn" 
        self.HAC = [HACPolicy(obs_space.low.shape[0], primitive_action_space, primitive_action_space, obs_space, max_action, environment.discrete_actions, **vars(args))]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        args.learning_type = lt

        args.relative_action = .05 if len(args.ctrl_type) != 0 else -1 # all relative actions within 5 cm?
        # adding intermediate levels
        if len(args.ctrl_type) > 0:
            paction_space = spaces.Box(low=low, high=high)
        else:
            paction_space = state_space if args.keep_instanced else reduced_state_space
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)

        if args.ctrl_type in ["limit", "blockdelta"]: 
            low = np.array([-.2,-.21,.8239])
            high = np.array([.2, .26,.8241])
        else:
            low = r_space.low
            high =r_space.high

        olow = np.concatenate([low.copy(), r_space.low.copy()])
        ohigh = np.concatenate([high.copy(), r_space.high.copy()])
        obs_space = spaces.Box(low=olow, high=ohigh)
        max_action = 1
        print(max_action)
        for _ in range(k_level-2):
            self.HAC.append(HACPolicy(obs_space.low.shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
            self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        if args.ctrl_type in ["limit", "blockdelta"]: 
            paction_space = spaces.Box(low=low, high=high)
            action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)
        else:
            r_space = state_space if args.keep_instanced else reduced_state_space
            low = np.concatenate([r_space.low.copy(), r_space.low.copy()])
            high = np.concatenate([r_space.high.copy(), r_space.high.copy()])


        args.object_dim = object_dim
        goal_final_size = goal_final.shape[0] if goal_final is not None else 0
        args.first_obj_dim = reduced_flat_state.shape[0]
        print(reduced_flat_state.shape, flat_state.shape)
        args.post_dim = 0
        if args.final_instanced:
            args.policy_type = "pair" # must use MLP architecture except at last layer with instanced
            input_shape = (int(flat_state.shape[0]) + goal_final_size, )
        else:
            args.policy_type = "basic"
            input_shape = flat_state.shape[0]
            input_shape = (int(input_shape + goal_final_size), )
        
        low = np.concatenate([np.array([-.15,-.1,.8289]), state_space.low.copy()])
        high = np.concatenate([np.array([.15,.2,.829  ]), state_space.high.copy()])
        obs_space = spaces.Box(low=low, high=high)
        # adding last level (might not be parametereized so state dim changes)
        print(input_shape, args.object_dim, args.first_obj_dim, paction_space, action_space)
        self.HAC.append(HACPolicy(input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        self.buffer_at = [0 for i in range(k_level)]
        # set some parameters
        self.keep_instanced = not args.reduced_state
        self.goal_final = goal_final
        self.k_level = k_level
        self.H = H
        self.threshold = threshold
        self.epsilon = epsilon
        args.epsilon = epsilon
        self.iscuda=False
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

    def get_target(self, level, full_state, environment_model):
        use_instanced = self.keep_instanced #or level == self.k_level - 1
        if level == 0 and self.ctrl_type in ["limit", "gripper2"]:
            return full_state['factored_state']['Gripper']
        elif level == 1 and self.ctrl_type in ["limit"] or level == 0 and self.ctrl_type in ["block2"]:
            return full_state['factored_state']['Block']
        if level == self.k_level - 1:
            return full_state['factored_state']['Block']
        return environment_model.get_HAC_flattened_state(full_state, instanced=True, use_instanced=use_instanced)



class BreakoutHAC(HAC):
    def __init__(self, args, k_level, H, threshold, epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state, state_space, reduced_state_space, object_dim=0, sampler=None):
        '''
        k_level is number of HAC levels, and it will be 3
        H is the time horizon
        threshold is epsilon close
        epsilon is the action epsilon
        flat_state is a full flat state
        reduced flat state is the flattened state if reduced
        '''
        # adding lowest level
        self.top_level_random = False
        args.hidden_sizes = [128,128]
        k_level = 3
        args.k_level = k_level
        no_param_input_shape = flat_state.shape if args.keep_instanced else reduced_flat_state.shape
        param_input_shape = (no_param_input_shape[0] * 2, )
        args.epsilon = 0 # we have separate exploration noise
        primitive_action_space = environment.action_space
        action_space = primitive_action_space
        paction_space = primitive_action_space
        self.primitive_action_discrete = environment.discrete_actions 
        self.max_action = primitive_action_space.n if environment.discrete_actions else primitive_action_space.high[0]
        max_action = self.max_action
        args.object_dim = 0
        args.first_obj_dim = 0
        args.policy_type = "basic" # must use MLP architecture except at last layer with instanced
        lt = args.learning_type
        if environment.discrete_actions: args.learning_type = "dqn" 
        
        self.ctrl_type = args.ctrl_type
        args.relative_action = -1.0
        if self.ctrl_type in ["limit", "lrel"]:
            low = np.array([7])
            high = np.array([76])
        else:
            low = np.array([72,7,0,0,0.9])
            high = np.array([72.1,76,0.1,0.1,1])
        paddle_input_shape = (no_param_input_shape[0] + low.shape[0], )


        obs_space = spaces.Box(low=np.concatenate([low, reduced_state_space.low]), high=np.concatenate([high, reduced_state_space.high]))
        self.HAC = [HACPolicy(paddle_input_shape[0], primitive_action_space, primitive_action_space, obs_space, max_action, environment.discrete_actions, **vars(args))]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        args.learning_type = lt
        
        # adding intermediate levels
        # self.reduced_flat_rel_space = spaces.Box(low=np.array([-73,-8,-2,-1,0,72,11,0,0,0.9,10,11,-2,-1,1]), 
        #                             high=np.array([73,74,2,1,0.1,72.1,79,0.1,0.1,1,80,79,2,1,1.1]), 
        args.hidden_sizes = [128,128,128,128,128]

        if self.ctrl_type in ["lrel"]:
            args.relative_action = 8.0 # take actions 8 steps away from target
        paction_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)

        if self.ctrl_type == "bvel3" or self.ctrl_type == "limit":
            low = np.array([-2,-1])
            high = np.array([2,1])
        elif self.ctrl_type == "bpos3":
            low = np.array([10,11])
            high = np.array([80,79])
        elif self.ctrl_type == "lrel":
            low = np.array([-73])
            high = np.array([73])
        else:
            low = np.array([10,11,-2,-1,1])
            high = np.array([80,79,2,1,1.1])

        obs_space = spaces.Box(low=np.concatenate([low, reduced_state_space.low]), high=np.concatenate([high, reduced_state_space.high]))
        max_action = 1
        ball_input_shape = (no_param_input_shape[0] + low.shape[0], )
        print(max_action)
        self.HAC.append(HACPolicy(ball_input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        if self.ctrl_type == "lrel":
            args.relative_action = 4.0 # reach a relative position within a certain margin
        else:
            args.relative_action = -1.0
        args.hidden_sizes = [128,128,512,1024]
        paction_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)
        if goal_final is not None:
            blow = np.array([20,11,0,0,0])
            bhigh = np.array([40,79,0.1,0.1,1])
            obs_space = spaces.Box(low=np.concatenate([blow, state_space.low]), high = np.concatenate([bhigh, state_space.high]))
        else:
            obs_space = state_space
        args.object_dim = object_dim
        goal_final_size = goal_final.shape[0] if goal_final is not None else 0
        args.first_obj_dim = reduced_flat_state.shape[0] + goal_final_size
        print(reduced_flat_state.shape, flat_state.shape)
        args.post_dim = 0
        input_shape = (int(flat_state.shape[0] + goal_final_size), )
        # adding last level (might not be parametereized so state dim changes)
        print(input_shape, args.object_dim, args.first_obj_dim)
        self.HAC.append(HACPolicy(input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        self.buffer_at = [0 for i in range(k_level)]
        # set some parameters
        self.keep_instanced = not args.reduced_state
        self.goal_final = goal_final
        self.k_level = 3
        self.H = H
        self.threshold = threshold
        self.epsilon = epsilon
        self.sampler = sampler
        args.epsilon = epsilon
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

    def get_target(self, level, full_state, environment_model):
        if level == 0:
            if self.ctrl_type in ["limit", "lrel"]:
                return np.array(full_state['factored_state']['Paddle'])[1:2]
            return np.array(full_state['factored_state']['Paddle'])
        if level == 1:
            if self.ctrl_type == "bvel3" or self.ctrl_type == "limit":
                return np.array(full_state['factored_state']['Ball'])[2:4]
            if self.ctrl_type == "bpos3":
                return np.array(full_state['factored_state']['Ball'])[0:2]
            if self.ctrl_type == "lrel":
                return np.array(full_state['factored_state']['Paddle'])[1:2] - np.array(full_state['factored_state']['Ball'])[1:2]
            return np.array(full_state['factored_state']['Ball'])
        if level == 2: # technically, this level has no target
            if self.sampler is not None:
                return sampler.param.copy()
            return np.array(full_state['factored_state']['Ball'])

class Breakout2HAC(HAC):
    def __init__(self, args, k_level, H, threshold, epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state, state_space, reduced_state_space, object_dim=0, sampler=None):
        '''
        k_level is number of HAC levels, and it will be 2
        two level versions of breakout
        H is the time horizon
        threshold is epsilon close
        epsilon is the action epsilon
        flat_state is a full flat state
        reduced flat state is the flattened state if reduced
        '''
        # adding lowest level
        self.top_level_random = False
        args.relative_action = -1.0
        args.hidden_sizes = [128,128,128,128,128]
        k_level = 2
        args.k_level = k_level
        no_param_input_shape = flat_state.shape if args.keep_instanced else reduced_flat_state.shape
        param_input_shape = (no_param_input_shape[0] * 2, )
        args.epsilon = 0 # we have separate exploration noise
        primitive_action_space = environment.action_space
        action_space = primitive_action_space
        paction_space = primitive_action_space
        self.primitive_action_discrete = environment.discrete_actions 
        self.max_action = primitive_action_space.n if environment.discrete_actions else primitive_action_space.high[0]
        max_action = self.max_action
        args.object_dim = 0
        args.first_obj_dim = 0
        args.policy_type = "basic" # must use MLP architecture except at last layer with instanced
        lt = args.learning_type
        if environment.discrete_actions: args.learning_type = "dqn" 

        self.ctrl_type =  args.ctrl_type
        if self.ctrl_type == "paddle":
            low = np.array([7])
            high = np.array([76])
        elif self.ctrl_type == "brel":
            low = np.array([-73.0])
            high = np.array([73])
        elif self.ctrl_type == "bvel":
            low = np.array([-2, -1])
            high = np.array([2,1])
        elif self.ctrl_type == "bpos":
            low = np.array([10,11])
            high = np.array([80,79])
        paddle_input_shape = (no_param_input_shape[0] + low.shape[0], )

        obs_space = spaces.Box(low=np.concatenate([low, reduced_state_space.low]), high=np.concatenate([high, reduced_state_space.high]))
        self.HAC = [HACPolicy(paddle_input_shape[0], primitive_action_space, primitive_action_space, obs_space, max_action, environment.discrete_actions, **vars(args))]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        args.learning_type = lt
        
        if self.ctrl_type in ["paddle"] and args.relative_action:
            args.relative_action = 8.0 # take actions 8 steps away from target
        if self.ctrl_type in ["brel"] and args.relative_action:
            args.relative_action = 6.0
        args.hidden_sizes = [128,128,128,256,512]
        paction_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=-1, high=1, shape= paction_space.shape)
        
        if goal_final is not None:
            blow = np.array([20,11,0,0,0])
            bhigh = np.array([40,79,0.1,0.1,1])
            obs_space = spaces.Box(low=np.concatenate([blow, state_space.low]), high = np.concatenate([bhigh, state_space.high]))
        else:
            obs_space = state_space
        args.object_dim = object_dim
        goal_final_size = goal_final.shape[0] if goal_final is not None else 0
        args.first_obj_dim = reduced_flat_state.shape[0] + goal_final_size
        print(reduced_flat_state.shape, flat_state.shape)
        args.post_dim = 0
        input_shape = (int(flat_state.shape[0] + goal_final_size), )
        # adding last level (might not be parametereized so state dim changes)
        print(input_shape, args.object_dim, args.first_obj_dim)
        self.HAC.append(HACPolicy(input_shape[0], paction_space, action_space, obs_space, max_action, False, **vars(args)))
        self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))

        self.buffer_at = [0 for i in range(k_level)]
        # set some parameters
        self.keep_instanced = not args.reduced_state
        self.goal_final = goal_final
        self.k_level = 2
        self.H = H
        self.threshold = threshold
        self.epsilon = epsilon
        args.epsilon = epsilon
        self.sampler = sampler
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0

    def get_target(self, level, full_state, environment_model):
        if self.ctrl_type == "paddle":
            return np.array(full_state['factored_state']['Paddle'])[1:2]
        elif self.ctrl_type == "brel":
            return np.array(full_state['factored_state']['Paddle'])[1:2] - np.array(full_state['factored_state']['Ball'])[1:2]
        elif self.ctrl_type == "bvel":
            return np.array(full_state['factored_state']['Ball'])[2:4]
        elif self.ctrl_type == "bpos":
            return np.array(full_state['factored_state']['Ball'])[0:2]
        if level == 1: # not used in a two layer structure
            if sampler is not None:
                return sampler.param.copy()
            return np.array(full_state['factored_state']['Ball'])

