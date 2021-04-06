import numpy as np
import os, cv2, time
import torch
import gym
from ReinforcementLearning.Policy.policy import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from EnvironmentModels.environment_model import FeatureSelector
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


INPUT_STATE = 0
OUTPUT_STATE = 1

FULL = 0
FEATURIZED = 1
DIFF = 2


class Option():
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        '''
        policy_reward is a PolicyReward object, which contains the necessary components to run a policy
        models is a tuple of dataset model and environment model
        featurizers is Featurizers object, which contains the object name, the gamma features, the delta features, and a vector representing which features are contingent
        '''
        if policy_reward is not None:
            self.assign_policy_reward(policy_reward)
        else:
            self.discrete_actions = False
        self.assign_models(models)
        self.assign_featurizers()
        self.object_name = object_name
        self.action_shape = (1,) # should be set in subclass
        self.action_prob_shape = (1,) # should be set in subclass
        self.output_prob_shape = (1,) # set in subclass
        self.control_max = None # set in subclass, maximum values for the parameter
        self.action_max = None # set in subclass, the limits for actions that can be taken
        self.action_space = None # set in subclass, the space object corresponding to all of the above information
        self.discrete = False
        self.iscuda = False
        self.use_mask = True
        self.last_factor = self.get_state(form=1, inp=1)
        print(self.get_state(form=0))
        self.param, self.mask = self.dataset_model.sample(self.environment_model.environment.get_state()) # sample should use full state as input
        self.input_shape = self.get_state(form=FEATURIZED, inp=INPUT_STATE).shape
        print("last_factor", self.last_factor)
        # parameters for temporal extension TODO: move this to a single function
        self.temp_ext = temp_ext 
        self.last_action = None
        self.terminated = True
        self.time_cutoff = -1
        self.timer = 0
        self.reward_timer = 0 # a hack to reduce the frequency of rewards
        self.reward_freq = 13

    def assign_models(self, models):
        self.dataset_model, self.environment_model, self.sampler = models

    def assign_policy_reward(self, policy_reward):
        self.policy = policy_reward.policy
        self.termination = policy_reward.termination
        self.reward = policy_reward.reward # the reward function for this option
        self.next_option = policy_reward.next_option
        if self.next_option is not None:
            self.discrete_actions = self.next_option.discrete # the action space might be discrete even if the parameter space is continuous

    def assign_featurizers(self):
        self.gamma_featurizer = self.dataset_model.gamma
        self.delta_featurizer = self.dataset_model.delta
        self.contingent_input = self.dataset_model.controllable

    def cuda(self):
        self.iscuda = True
        if self.policy is not None:
            self.policy.cuda()
        if self.next_option is not None:
            self.next_option.cuda()

    def cpu(self):
        self.iscuda = False
        if self.policy is not None:
            self.policy.cpu()
        if self.next_option is not None:
            self.next_option.cpu()


    # def get_flattened_input_state(self, factored_state):
    #     return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)
    def get_state(self, full_state=None, form=1, inp=0):
        if full_state is None:
            full_state = self.environment_model.get_state()
        elif type(full_state) is list or type(full_state) is np.ndarray:
            return np.array([self.get_single_state(f, form=form, inp=inp) for f in full_state])
        else: # assume it is a dict
            return self.get_single_state(f, form=form, inp=inp)
        # form is an enumerator, 0 is full state, 1 is gamma/delta state, 2 is diff using last_factor
    def get_single_state(full_state, form=1, inp=0):
        factored_state = full_state['factored_state']
        featurize = self.gamma_featurizer if inp == 0 else self.delta_featurizer
        if form == 0:
            return pytorch_model.wrap(self.environment_model.flatten_factored_state(factored_state), cuda=self.iscuda)
        elif form == 1:
            return pytorch_model.wrap(featurize(self.environment_model.flatten_factored_state(factored_state)), cuda=self.iscuda)
        else:
            return pytorch_model.wrap(self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state)), cuda=self.iscuda) - self.last_factor

    def get_param(self, full_state, last_done):
        # move this into option file
        # print(self.timer, last_done)
        if last_done or self.timer == 0:
            # print("resample")
            if self.object_name == 'Raw':
                self.param, self.mask = torch.tensor([1]), torch.tensor([1])
            else: # commented out is old version
                # param, mask = self.dataset_model.sample(full_state, 1, both=self.use_both==2, diff=self.use_both==1, name=self.object_name)
                if self.timer == 0:
                    self.param, self.mask = self.sampler.sample(self.get_state(full_state, form=0))  
                    # print(self.param, self.get_state(full_state, form=FEATURIZED, inp=OUTPUT_STATE))
            self.param, self.mask = pytorch_model.unwrap(self.param), pytorch_model.unwrap(self.mask)
        # print(self.timer, self.time_cutoff, self.param, self.get_state(full_state, form=FEATURIZED, inp=OUTPUT_STATE))
        return self.param, self.mask

    def convert_param(self, param):
        if self.discrete:
            return self.get_possible_parameters()[param.squeeze().long()][0]
        return param

    def sample_action_chain(self, batch, state_chain, random=False): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        '''
        # compute policy information for next state
        # input_state = self.get_state(state, form=FEATURIZED, inp=INPUT_STATE)

        # rl_output = policy.forward(input_state.unsqueeze(0), param.unsqueeze(0)) # REMOVE THIS LINE LATER
        if random:
            act = self.action_space.sample()
            policy_batch = None
        # get the action from the behavior policy, baction is integer for discrete
        if self.temp_ext and (self.next_option is not None and not self.next_option.terminated):
            act = self.last_action # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
        else:
            batch['obs'] = self.get_state(batch['full_state'], form=FEATURIZED, inp=INPUT_STATE)
            batch['next_obs'] = self.get_state(batch['next_full_state'], form=FEATURIZED, inp=INPUT_STATE)
            policy_batch, state = self.policy.forward(batch, state_chain[-1]) # uncomment this
            act = policy_batch.act
            act = to_numpy(act)
            act = self.policy.exploration_noise(act, self.data)

        # print(self.iscuda, param, baction, action)
        chain = [act.squeeze()]
        param = batch['param']
        batch['param'] = act.squeeze()
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            rem_chain, result, rem_state_chain = self.next_option.sample_action_chain(batch, state_chain[:-1])
            chain = rem_chain + chain
            state = rem_state_chain + [state]
        batch['param'] = param
        return chain, policy_batch, state

    def step(self, last_state, chain):
        # This can only be called once per time step because the state diffs are managed here
        if self.next_option is not None:
            self.next_option.step(last_state, chain[:len(chain)-1])
        self.last_action = chain[-1]
        self.last_factor = self.get_state(last_state, form=FEATURIZED, inp=OUTPUT_STATE)

    def step_timer(self, done): # TODO: does this need to handle done chains?
        self.timer += 1
        # print(done, self.timer)
        if done or (self.timer == self.time_cutoff and self.time_cutoff > 0):
            self.timer = 0

    def terminate_reward(self, state, param, chain, needs_reward=True):
        # recursively get all of the dones and rewards
        dones, rewards = list(), list()
        if self.next_option is not None:
            last_dones, last_rewards = self.next_option.terminate_reward(state, self.next_option.convert_param(chain[-1]), chain[:len(chain)-1], needs_reward=False)
        # get the state to be entered into the termination check
        input_state = self.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
        # object_state = self.get_state(state, form = FEATURIZED, inp=OUTPUT_STATE)
        object_state = self.get_state(state, form = DIFF if self.reward.use_diff else FEATURIZED, inp=OUTPUT_STATE)
        mask = self.dataset_model.get_active_mask()

        # assign done and reward
        done = self.termination.check(input_state, object_state * mask, param * mask)
        if needs_reward:
            reward = self.reward.get_reward(input_state, object_state * mask, param * mask)
        else:
            reward = 0
        self.terminated = done

        # environment termination overrides
        if self.environment_model.get_done(state):
            done = 1

        # manage a maximum time duration to run an option, NOT used, quietly switches option
        if self.time_cutoff > 0:
            if self.timer == self.time_cutoff - 1:
                done = 1
        dones = last_dones + [done]
        rewards = last_rewards + [reward]
        return dones, rewards

    def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        if self.next_option is not None:
            self.next_option.record_state(state, next_state, action_chain[:-1], rl_outputs[:-1], action_chain[-1], rewards[:-1], dones[:-1])
        self.rollouts.append(**self.get_state_dict(state, next_state, action_chain, rl_outputs, param, rewards, dones))

    def get_state_dict(self, state, next_state, action_chain, rl_outputs, param, rewards, dones): # also used in HER
            return {'state': self.get_state(state, form=FEATURIZED, inp=INPUT_STATE),
                'next_state': self.get_state(next_state, form=FEATURIZED, inp=INPUT_STATE),
                'object_state': self.get_state(state, form=FEATURIZED, inp=OUTPUT_STATE),
                'next_object_state': self.get_state(next_state, form=FEATURIZED, inp=OUTPUT_STATE),
                'state_diff': self.get_state(state, form=DIFF, inp=OUTPUT_STATE), 
                'true_action': action_chain[0],
                'true_reward': rewards[0],
                'true_done': dones[0],
                'action': action_chain[-1],
                'probs': rl_outputs[-1].probs[0],
                'Q_vals': rl_outputs[-1].Q_vals[0],
                'param': param, 
                'mask': self.dataset_model.get_active_mask(), 
                'reward': rewards[-1], 
                'done': dones[-1]}


    def get_input_state(self, state = None): # gets the state used for the forward model/policy
        if state is not None:
            input_state = self.gamma_featurizer(self.pytorch_model.wrap(environment_model.get_flattened_state(), cuda=args.cuda))
        else:
            input_state = self.gamma_featurizer(self.pytorch_model.wrap(environment_model.flatten_factored_state(state)))
        return input_state

    def forward(self, state, param): # runs the policy and gets the RL output
        return self.policy(state, param)

    def compute_return(self, gamma, start_at, num_update, next_value, return_max = 20, return_form="value"):
        return self.rollouts.compute_return(gamma, start_at, num_update, next_value, return_max = 20, return_form="value")

    # def set_behavior_epsilon(self, epsilon):
    #     self.behavior_policy.epsilon = epsilon


    def save(self, save_dir, clear=False):
        # checks and prepares for saving option as a pickle
        policy, rollouts = self.policy, self.rollouts
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.policy.cpu() 
            self.rollouts.cpu()
            self.last_factor = self.last_factor.cpu()
            self.policy.save(save_dir, self.object_name +"_policy")
            print(self.iscuda)
            if self.iscuda:
                self.policy.cuda()
            if clear:
                self.policy, self.rollouts = None, None # removes the policy and rollouts for saving
            return policy, rollouts
        return None, None

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.object_name +"_policy.pt"))
            print(self.policy)



class PrimitiveOption(Option): # primative discrete actions
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        self.num_params = models[1].environment.num_actions
        self.object_name = "Action"
        environment = models[1].environment
        self.action_shape = environment.action_space.shape or (1,)
        self.action_prob_shape = environment.action_space.shape or (1,)
        self.output_prob_shape = environment.action_space.shape or environment.action_space.n# (models[1].environment.num_actions, )
        self.control_max = environment.action_space.high[0]
        self.action_max = environment.action_space.shape or environment.action_space.n
        self.discrete = environment.action_space.shape is not None
        self.discrete_actions = models[1].environment.discrete_actions
        self.next_option = None
        self.iscuda = False
        self.policy = None
        self.dataset_model = None
        self.time_cutoff = 1
        self.rollouts = None

    def save(self, save_dir, clear=False):
        return None, None

    def load_policy(self, load_dir):
        pass

    def set_behavior_epsilon(self, epsilon):
        pass

    def step(self, last_state, chain):
        pass

    def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        pass

    def cpu(self):
        self.iscuda = False

    def cuda(self):
        self.iscuda = True

    def get_possible_parameters(self):
        if self.iscuda:
            return [(torch.tensor([i]).cuda(), torch.tensor([1]).cuda()) for i in range(self.num_params)]
        return [(torch.tensor([i]), torch.tensor([1])) for i in range(self.num_params)]

    def sample_action_chain(self, state, param, random=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        done = True
        chain = [param.squeeze().long()]
        return chain, list()

    def terminate_reward(self, state, param, chain, needs_reward=False):
        return [1], [0]


class RawOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext)
        self.object_name = "Raw"
        self.action_shape = (1,)
        self.action_prob_shape = (self.environment_model.environment.num_actions,)
        self.action_max = self.environment_model.environment.action_space.high
        self.action_space = self.environment_model.environment.action_space
        self.control_max = 0 # could put in "true" parameter, unused otherwise
        self.discrete = False # This should not be used, since rawoption is not performing parameterized RL
        self.use_mask = False
        self.stack = torch.zeros((4,84,84))
        # print("frame", self.environment_model.environment.get_state()['Frame'].shape)
        # self.param = self.environment_model.get_param(self.environment_model.environment.get_state()[1])
        self.param = self.environment_model.get_param(self.environment_model.environment.get_state())
        self.discrete_actions = self.environment_model.environment.discrete_actions

    def get_state_dict(self, state, next_state, action_chain, rl_outputs, param, rewards, dones): # also used in HER
            return {'state': self.get_state(state, form=FEATURIZED, inp=INPUT_STATE),
                'next_state': self.get_state(next_state, form=FEATURIZED, inp=INPUT_STATE),
                'object_state': state["Object"],
                'next_object_state': next_state["Object"],
                'state_diff': state["Action"], # storing some dummy information
                'true_action': action_chain[0],
                'true_reward': rewards[0],
                'true_done': dones[0],
                'action': action_chain[-1],
                'probs': None if rl_outputs[-1].probs is None else rl_outputs[-1].probs[0],
                'Q_vals': None if rl_outputs[-1].Q_vals is None else rl_outputs[-1].Q_vals[0],
                'param': param, 
                'mask': self.dataset_model.get_active_mask(), 
                'reward': rewards[-1], 
                'done': dones[-1]}


    def get_param(self, full_state, done):
        if done:
            self.param = self.environment_model.get_param(full_state)
            return pytorch_model.wrap(self.param, cuda=self.iscuda), pytorch_model.wrap([1,], cuda=self.iscuda)
        return pytorch_model.wrap(self.param, cuda=self.iscuda), pytorch_model.wrap([1,], cuda=self.iscuda)

    def get_possible_parameters(self):
        if self.iscuda:
            return [(torch.tensor([1]).cuda(), torch.tensor([1]).cuda())]
        return [(torch.tensor([1]), torch.tensor([1]))]

    def cuda(self):
        super().cuda()
        # self.stack = self.stack.cuda()

    def get_state(self, full_state = None, form=0, inp=1):
        if not full_state: return self.environment_model.get_state()['raw_state']
        if type(full_state) is list or type(full_state) is np.ndarray: return np.array([f['raw_state'] for f in full_state])
        else: return full_state['raw_state']

    def get_input_state(self):
        # stack = stack.roll(-1,0)
        # stack[-1] = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
        # input_state = stack.clone().detach()

        input_state = self.get_state(self.environment_model.get_state())
        return input_state


    def sample_action_chain(self, batch, state_chain, random=False):
        '''
        Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
        also returns whether the current state is a termination condition. The option will still return an action in a termination state
        The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
        '''
        # input_state = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
        # self.stack = self.stack.roll(-1,0)
        # self.stack[-1] = input_state
        # input_state = self.stack.clone()
        if random:
            act = self.action_space.sample()
            policy_batch = None
            state = None
        else:
            batch['obs'] = self.get_state(batch['full_state'])
            # batch['next_obs'] = self.get_state(batch['next_full_state'])
            state = None if state_chain is None else state_chain[-1]
            policy_batch = self.policy.forward(batch, state) # uncomment this
            act = policy_batch.act
            state = [policy_batch.state] if state is not None else None
            # get the action from the behavior policy, baction is integer for discrete
            act = to_numpy(act)
            act = self.policy.exploration_noise(act, batch)

        # input_state = pytorch_model.wrap(self.environment_model.get_raw_state(state), cuda=self.iscuda)
        # print("raw_state", self.environment_model.get_raw_state(state), input_state)
        # if len(param.shape) == 1:
        #     param = param.unsqueeze(0)
        # rl_output = self.policy.forward(input_state.unsqueeze(0), param) # uncomment later
        # rl_output = policy.forward(input_state.unsqueeze(0), param)
        # print("forwarded")
        # baction = self.behavior_policy.get_action(rl_output)
        chain = [act.squeeze()]
        # print(chain)
        return chain, policy_batch, state

    def terminate_reward(self, state, param, chain, needs_reward=False):
        return [int(self.environment_model.environment.done)], [self.environment_model.environment.reward]#, torch.tensor([self.environment_model.environment.reward]), None, 1

    # The definition of this function has changed
    def get_action(self, action, mean, variance):
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()], None#torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

class ModelCounterfactualOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext)
        self.action_prob_shape = self.next_option.output_prob_shape
        if self.discrete_actions:
            self.action_shape = (1,)
        else:
            self.action_shape = self.next_option.output_prob_shape
        self.action_max = self.next_option.control_max
        self.action_min = self.next_option.control_min
        self.control_max = self.dataset_model.control_max
        self.control_min = self.dataset_model.control_min
        self.action_space = gym.spaces.Box(self.action_min, self.action_max)
        self.output_prob_shape = (self.dataset_model.delta.output_size(), ) # continuous, so the size will match
        
        # TODO: fix this so that the output size is equal to the nonzero elements of the self.dataset_model.selection_binary() at each level

    def get_action(self, action, mean, variance):
        if self.discrete_actions:
            return mean[torch.arange(mean.size(0)), action.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), action.squeeze().long()])
        idx = action
        dist = torch.distributions.normal.Normal # TODO: hardcoded action distribution as diagonal gaussian
        log_probs = dist(mean, variance).log_probs(action)
        return torch.exp(log_probs), log_probs

    def get_critic(self, state, action, mean):
        return self.policy.compute_Q(state, action)

option_forms = {'model': ModelCounterfactualOption, "raw": RawOption}