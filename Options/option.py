import numpy as np
import os, cv2, time
import torch
from ReinforcementLearning.Policy.policy import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian

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
        self.discrete = False
        self.iscuda = False
        self.last_factor = self.get_state(form=1, inp=1)
        self.param, self.mask = self.dataset_model.sample(self.get_state(form=0))
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
        self.dataset_model, self.environment_model = models

    def assign_policy_reward(self, policy_reward):
        self.policy = policy_reward.policy
        self.behavior_policy = policy_reward.behavior_policy
        self.termination = policy_reward.termination
        self.reward = policy_reward.reward # the reward function for this option
        self.next_option = policy_reward.next_option
        self.discrete_actions = self.next_option.discrete # the action space might be discrete even if the parameter space is continuous
        self.rollouts = policy_reward.rollouts

    def assign_featurizers(self):
        self.gamma_featurizer = self.dataset_model.gamma
        self.delta_featurizer = self.dataset_model.delta
        self.contingent_input = self.dataset_model.controllable

    def cuda(self):
        self.iscuda = True
        self.last_factor = self.last_factor.cuda()
        if self.policy is not None:
            self.policy.cuda()
        if self.rollouts is not None:
            self.rollouts.cuda()

    def cpu(self):
        self.iscuda = False
        self.last_factor = self.last_factor.cpu()
        if self.policy is not None:
            self.policy.cpu()
        if self.rollouts is not None:
            self.rollouts.cpu()

    # def get_flattened_input_state(self, factored_state):
    #     return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)
    def get_state(self, factored_state=None, form=1, inp=0):
        if factored_state is None:
            factored_state = self.environment_model.get_flattened_state()
        # form is an enumerator, 0 is full state, 1 is gamma/delta state, 2 is diff using last_factor
        featurize = self.gamma_featurizer if inp == 0 else self.delta_featurizer
        if form == 0:
            return pytorch_model.wrap(self.environment_model.flatten_factored_state(factored_state), cuda=self.iscuda)
        elif form == 1:
            return pytorch_model.wrap(featurize(self.environment_model.flatten_factored_state(factored_state)), cuda=self.iscuda)
        else:
            return pytorch_model.wrap(self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state)), cuda=self.iscuda) - self.last_factor

    def get_param(self, full_state, last_done):
        # move this into option file
        if last_done:
            # print("resample")
            if self.object_name == 'Raw':
                self.param, self.mask = torch.tensor([1]), torch.tensor([1])
            else: # commented out is old version
                # param, mask = self.dataset_model.sample(full_state, 1, both=self.use_both==2, diff=self.use_both==1, name=self.object_name)
                if self.timer == 0:
                    self.param, self.mask = self.dataset_model.sample(self.get_state(full_state, form=0))  
            self.param, self.mask = self.param.squeeze(), self.mask.squeeze()
            if self.cuda:
                self.param, self.mask = self.param.cuda(), self.mask.cuda()
        # print(self.timer, self.time_cutoff, self.param, self.get_state(full_state, form=FEATURIZED, inp=OUTPUT_STATE))
        return self.param, self.mask

    def convert_param(self, param):
        if self.discrete:
            return self.get_possible_parameters()[param.squeeze().long()][0]
        return param

    def sample_action_chain(self, state, param):
        '''
        Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
        also returns whether the current state is a termination condition. The option will still return an action in a termination state
        The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
        '''
        # compute policy information for next state
        input_state = self.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
        rl_output = self.policy.forward(input_state.unsqueeze(0), param.unsqueeze(0))

        # get the action from the behavior policy, baction is integer for discrete
        if self.temp_ext and (self.next_option is not None and not self.next_option.terminated):
            baction = self.last_action # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
        else:
            baction = self.behavior_policy.get_action(rl_output)
        action = self.next_option.convert_param(baction.squeeze())
        chain = [baction.squeeze()]
        rl_outputs = [rl_output]
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            rem_chain, rem_rl_outputs = self.next_option.sample_action_chain(state, action)
            chain = rem_chain + chain
            rl_outputs = rem_rl_outputs + rl_outputs
        return chain, rl_outputs

    def step(self, last_state, chain):
        # This can only be called once per time step because the state diffs are managed here
        if self.next_option is not None:
            self.next_option.step(last_state, chain[:len(chain)-1])
        self.last_action = chain[-1]
        self.last_factor = self.get_state(last_state, form=FEATURIZED, inp=OUTPUT_STATE)
    
    def step_timer(self, done):
        self.timer += 1
        if self.time_cutoff > 0:
            if done or self.timer == self.time_cutoff:
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


        # manage a maximum time duration to run an option
        if self.time_cutoff > 0:
            if self.timer == self.time_cutoff - 1:
                done = True
        if done:
            print("Terminated: ", self.timer)
        dones = last_dones + [done]
        rewards = last_rewards + [reward]
        return dones, rewards

    def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        if self.next_option is not None:
            self.next_option.record_state(state, next_state, action_chain[:-1], rl_outputs[:-1], action_chain[-1], rewards[:-1], dones[:-1])
        self.rollouts.append(**{'state': self.get_state(state, form=FEATURIZED, inp=INPUT_STATE),
                'next_state': self.get_state(next_state, form=FEATURIZED, inp=INPUT_STATE),
                'object_state': self.get_state(state, form=FEATURIZED, inp=OUTPUT_STATE),
                'next_object_state': self.get_state(next_state, form=FEATURIZED, inp=OUTPUT_STATE),
                'state_diff': self.get_state(state, form=DIFF, inp=OUTPUT_STATE), 
                'true_action': action_chain[0],
                'action': action_chain[-1],
                'probs': rl_outputs[-1].probs[0],
                'Q_vals': rl_outputs[-1].Q_vals[0],
                'param': param, 
                'mask': self.dataset_model.get_active_mask(), 
                'reward': rewards[-1], 
                'done': dones[-1]})


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

    def set_behavior_epsilon(self, epsilon):
        self.behavior_policy.epsilon = epsilon


    def save(self, save_dir):
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.policy.cpu() 
            self.rollouts.cpu()
            self.policy.save(save_dir, self.object_name +"_policy")
            print(self.iscuda)
            if self.iscuda:
                self.policy.cuda()

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.object_name +"_policy.pt"))
            print(self.policy)



class PrimitiveOption(Option): # primative discrete actions
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        self.num_params = models[1].environment.num_actions
        self.object_name = "Action"
        self.action_shape = (1,)
        self.action_prob_shape = (1,)
        self.output_prob_shape = (models[1].environment.num_actions, )
        self.discrete = True
        self.next_option = None
        self.iscuda = False
        self.policy = None
        self.dataset_model = None
        self.time_cutoff = 1

    def save(self, args):
        pass

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

    def sample_action_chain(self, state, param): # param is an int denoting the primitive action, not protected (could send a faulty param)
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
        self.discrete = True
        self.stack = torch.zeros((4,84,84))

    def get_possible_parameters(self):
        if self.iscuda:
            return [(torch.tensor([1]).cuda(), torch.tensor([1]).cuda())]
        return [(torch.tensor([1]), torch.tensor([1]))]

    def cuda(self):
        super().cuda()
        self.stack = self.stack.cuda()

    def get_input_state(self):
        stack = stack.roll(-1,0)
        stack[-1] = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
        input_state = stack.clone().detach()
        return input_state


    def sample_action_chain(self, state, param):
        '''
        Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
        also returns whether the current state is a termination condition. The option will still return an action in a termination state
        The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
        '''
        input_state = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
        self.stack = self.stack.roll(-1,0)
        self.stack[-1] = input_state
        input_state = self.stack.clone()
        rl_output = self.policy.forward(input_state.unsqueeze(0), param)
        baction = self.behavior_policy.get_action(rl_output)
        chain = [baction.squeeze()]
        return chain, rl_output

    def terminate_reward(self, state, param, chain, needs_reward=False):
        return [self.environment_model.environment.done], [self.environment_model.environment.reward]#, torch.tensor([self.environment_model.environment.reward]), None, 1

    # The definition of this function has changed
    def get_action(self, action, mean, variance):
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()]

class DiscreteCounterfactualOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        super().__init__(self, policy_reward, models, object_name, temp_ext=temp_ext)
        self.action_shape = (1,)
        self.action_prob_shape = self.next_option.output_prob_shape()
        self.output_prob_shape = (len(self.get_possible_parameters()),)

    def set_parameters(self, dataset_model):
        '''
        sets the discrete distribution of options which are all the different outcomes
        '''
        self.termination.set_parameters(dataset_model)

    def convert_param(self, baction):
        return self.get_possible_parameters()[baction.long()][0]

    def get_param_shape(self):
        return self.get_possible_parameters()[0][0].shape

    def get_mask(self, param): # param is a raw parameter without a mask
        full_params = self.get_possible_parameters()
        params = torch.stack([p[0] for p in full_params], dim=0)
        minv, minidx = (params - param).norm(dim=1).min(dim=0)
        # print(params, param, (params - param).norm(dim=1), minidx, minv, full_params[minidx][1])
        return full_params[minidx][1]

    def get_possible_parameters(self):
        '''
        gets all the possible actions in order as a list
        '''
        params = self.termination.discrete_parameters
        cloned = []
        for i in range(len(params)):
            cloned.append((params[i][0].clone(), params[i][1].clone()))
        if self.iscuda:
            cuda_params = []
            for i in range(len(params)):
                cuda_params.append((params[i][0].cuda(), params[i][1].cuda()))
            params = cuda_params
        return params

    def get_action(self, action, mean, variance): # mean is probability
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

    def get_critic(self, action, mean):
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

class ModelCounterfactualOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext)
        self.action_prob_shape = self.next_option.output_prob_shape
        if self.discrete_actions:
            self.action_shape = (1,)
        else:
            self.action_shape = self.next_option.output_prob_shape
        self.output_prob_shape = (self.dataset_model.gamma.output_size, ) # continuous, so the size will match


    def get_action(self, action, mean, variance):
        if self.discrete_actions:
            return mean[torch.arange(mean.size(0)), action.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), action.squeeze().long()])
        idx = action
        dist = torch.distributions.normal.Normal # TODO: hardcoded action distribution as diagonal gaussian
        log_probs = dist(mean, variance).log_probs(action)
        return torch.exp(log_probs), log_probs

    def get_critic(self, state, action, mean):
        return self.policy.compute_Q(state, action)

class HackedStateCounterfactualOption(DiscreteCounterfactualOption): # eventually, we will have non-hacked StateCounterfactualOption
    def __init__(self, policy_reward, models, object_name, temp_ext=False):
        super().__init__(self, policy_reward, models, object_name, temp_ext=temp_ext)
        self.action_shape = (1,)
        self.discrete = True

    # def cuda(self):
    #     super().cuda()
    #     self.stack = self.stack.cuda()
    def get_flattened_diff_state(self, factored_state):
        object_state = pytorch_model.wrap(self.environment_model.get_flattened_state(names=[self.object_name]), cuda=self.iscuda)
        return torch.cat((object_state, object_state - self.last_factor), dim=0)


    def sample_action_chain(self, state, param):
        '''
        In practice, this should be the only step that should be significantly different from a classic option
        forward, however, might have some bugs
        '''
        input_state = self.get_flattened_input_state(state)
        target_state = self.dataset_model.reverse_model(param)
        rl_output = self.policy.forward(input_state.unsqueeze(0), param.unsqueeze(0))
        if self.temp_ext and (self.next_option is not None and not self.next_option.terminated):
            action = self.last_action # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
        else:
            dist = 10000
            bestaction = None
            for action in self.next_option.get_possible_parameters():
                temp = self.environment_model.get_factored_state(instanced=False)
                if type(self.next_option) is not PrimitiveOption:
                    temp[self.next_option.object_name] += action[0] # the action should be a masked expected change in state of the last object
                else:
                    temp[self.next_option.object_name][-1] = action[0]
                features = self.dataset_model.featurize(np.expand_dims(self.environment_model.flatten_factored_state(temp, instanced = False), axis=0))
                newdist = ((features - target_state[1]) * self.dataset_model.input_mask).norm()
                if newdist < dist:
                    newdist = dist
                    bestaction = action
            action = bestaction[0]
        rem_chain, last_rl_output = self.next_option.sample_action_chain(state, action)
        chain = rem_chain + [action]
        self.last_action = action
        return chain, rl_output

class ContinuousParticleCounterfactualOption(Option): # TODO: write this code
    def set_parameters(self, dataset_model):
        pass

    def get_action(self, action, *args):
        pass

class ContinuousGaussianCounterfactualOption(Option):
    def get_action(self, action, *args):
        pass

option_forms = {"discrete": DiscreteCounterfactualOption, "continuousGaussian": ContinuousGaussianCounterfactualOption, 'model': ModelCounterfactualOption,
"continuousParticle": ContinuousParticleCounterfactualOption, "raw": RawOption, "hacked": HackedStateCounterfactualOption}