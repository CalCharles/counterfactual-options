import numpy as np
import os, cv2, time
import torch
from ReinforcementLearning.Policy.policy import pytorch_model

class Option():
    def __init__(self, policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False):
        self.policy=policy
        self.behavior_policy = behavior_policy
        self.termination = termination
        self.reward = reward # the reward function for this option
        self.next_option = next_option
        self.dataset_model = dataset_model
        self.environment_model = environment_model
        self.object_name = object_name
        self.names = names # names for the input state
        self.action_shape = (1,) # should be set in subclass
        self.action_prob_shape = (1,) # should be set in subclass
        self.discrete = self.termination.discrete
        self.iscuda = False
        self.last_factor = self.get_flattened_object_state(self.environment_model.get_factored_zero_state())
        print("last_factor", self.last_factor)
        # parameters for temporal extension
        self.temp_ext = temp_ext 
        self.last_action = None
        self.terminated = True
        self.time_cutoff = -1
        self.timer = 0
        self.reward_timer = 0 # a hack to reduce the frequency of rewards
        self.reward_freq = 13

    def cuda(self):
        self.iscuda = True
        self.last_factor = self.last_factor.cuda()
        if self.policy is not None:
            self.policy.cuda()

    def cpu(self):
        self.iscuda = False
        self.last_factor = self.last_factor.cpu()
        if self.policy is not None:
            self.policy.cpu()

    # def freeze_past(self, top=True): 
    #     if self.next_option is not None:
    #         if self.policy is not None:


    def get_flattened_input_state(self, factored_state):
        return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)

    def get_flattened_object_state(self, factored_state):
        return pytorch_model.wrap(self.environment_model.get_flattened_state(names=[self.object_name]), cuda=self.iscuda)

    def get_flattened_diff_state(self, factored_state):
        return self.get_flattened_object_state(factored_state) - self.last_factor

    def sample_action_chain(self, state, param):
        '''
        Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
        also returns whether the current state is a termination condition. The option will still return an action in a termination state
        The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
        '''
        # if self.object_name == "Paddle":
        #     print(self.policy.action_eval.weight)
        input_state = self.get_flattened_input_state(state)
        rl_output = self.policy.forward(input_state.unsqueeze(0), param.unsqueeze(0))
        if self.temp_ext and (self.next_option is not None and not self.next_option.terminated):
            baction = self.last_action # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
        else:
            baction = self.behavior_policy.get_action(rl_output)
        if self.next_option.discrete:
            action = self.next_option.convert_param(baction.squeeze())
            # print(self.object_name, baction, action, param, self.next_option.get_possible_parameters()[baction.squeeze().long()][0], self.next_option.convert_param(baction.squeeze()), self.next_option.discrete)
        chain = [baction.squeeze()]
        if self.next_option is not None:
            rem_chain, last_rl_output = self.next_option.sample_action_chain(state, action)
            chain = rem_chain + chain
        self.last_action = baction
        return chain, rl_output

    def get_mask(self, param):
        pass # gets a mask based on the parameter

    def terminate_reward(self, state, param, chain, needs_reward=True):
        if self.next_option is not None:
            last_done, last_reward, last_all_reward, diff, last_last_done = self.next_option.terminate_reward(state, self.next_option.convert_param(chain[-1]), chain[:len(chain)-1], needs_reward=False)
        object_state, diff = self.get_flattened_object_state(state), self.get_flattened_diff_state(state)
        mask = self.get_mask(param)
        if mask.size(0) == diff.size(0) != object_state.size(0):
            corrected_object_state = diff.clone()
        else:
            corrected_object_state = object_state
        # print(corrected_object_state, diff, last_done, self.last_factor)
        done = self.termination.check(corrected_object_state * mask, diff * mask, param)
        if needs_reward:
            reward = self.reward.get_reward(corrected_object_state * mask, diff * mask, param)
            all_reward = []
            for param, mask in self.get_possible_parameters(): # TODO: unless there are a ton of parameters, this shouldn't be too expensive
                r = self.reward.get_reward(corrected_object_state * mask, diff * mask, param)
                if self.reward_timer > 0:
                    r = 0
                all_reward.append(r)
            ## hacked, comment out in most cases
            all_reward = pytorch_model.wrap(all_reward, cuda=self.iscuda)
            if self.reward_timer > 0:
                # if self.object_name == "Ball":
                #     print("suspended reward", self.object_name, self.reward_timer, reward)
                reward = 0
                self.reward_timer -= 1
            if all_reward.max() > 0:
                # if self.object_name == "Ball":
                #     print(corrected_object_state, mask, param)
                self.reward_timer = self.reward_freq
        else:
            reward = 0
            all_reward = [0 for i in range(len(self.get_possible_parameters()))]
            all_reward = pytorch_model.wrap(all_reward, cuda=self.iscuda)
        # print(self.object_name, all_reward, self.get_possible_parameters())
        # print ("rewarding", corrected_object_state * mask, param * mask, reward)
        # if reward > 0:
        if last_done:
            self.last_factor = object_state
        # print(self.timer, self.time_cutoff)
        self.terminated = done
        if self.time_cutoff > 0:
            self.timer += 1
            if self.timer == self.time_cutoff:
                done = True
            if done or self.timer == self.time_cutoff:
                self.timer = 0
        return done, reward, all_reward, diff, last_done


    def get_action_distribution(self, state, diff, param):
        '''
        gets the action probabilities, Q values, and any other statistics from the policy (inside a PolicyRollout).
        Operates on batches
        '''
        policy_rollout = self.policy.forward(state, diff, param)
        done = self.termination.check(state, diff, action)
        return policy_rollout, done

    def forward(self, state, param):
        return self.policy(state, param)

    def get_action(self, action, *args):
        '''
        depending on the way the distribution works, returns the args with the action selected
        the args could be distributions (continuous), parameters of distributions (gaussian distributions), or selections (discrete)
        '''
        pass

    def set_parameters(self, model):
        '''
        performs necessary settings for the parameter selection
        '''
        pass

    def set_behavior_epsilon(self, epsilon):
        self.behavior_policy.epsilon = epsilon

    def get_param_shape(self):
        '''
        gets the shape of the parameter
        '''
        return (1,)

    def convert_param(self, baction):
        if self.discrete:
            # idx = param.max(0)[1]
            return self.get_possible_parameters()[baction.long()][0]
        return param

    def save(self, save_dir):
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.policy.cpu() 
            self.policy.save(save_dir, self.object_name +"_policy")
            print(self.iscuda)
            if self.iscuda:
                self.policy.cuda()

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.object_name +"_policy.pt"))
            print(self.policy)


class PrimitiveOption(Option): # primative discrete actions
    def __init__(self, policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, num_params=0, temp_ext=False):
        self.num_params = num_params
        self.object_name = "Action"
        self.action_shape = (1,)
        self.action_prob_shape = (1,)
        self.discrete = True
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
        return chain, None

    def terminate_reward(self, state, param, chain, needs_reward=False):
        return 1, 0, torch.tensor([0]), None, 1

    # def convert_param(self, param):
    #     if self.discrete:
    #         return self.get_possible_parameters()[param.squeeze().long()][0]
    #     return param

class RawOption(Option):
    def __init__(self, policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False):
        super().__init__(policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False)
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
        return self.environment_model.environment.done, self.environment_model.environment.reward, torch.tensor([self.environment_model.environment.reward]), None, 1

    def get_action(self, action, *args):
        vals = []
        idx = action
        for vec in args:
            vals.append(vec[torch.arange(vec.size(0)), idx.squeeze().long()])
        return vals

class DiscreteCounterfactualOption(Option):
    def __init__(self, policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False):
        super().__init__(policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False)
        self.action_shape = (1,)
        self.action_prob_shape = (len(self.next_option.get_possible_parameters()),)

    def set_parameters(self, dataset_model):
        '''
        sets the discrete distribution of options which are all the different outcomes
        '''
        self.termination.set_parameters(dataset_model)

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

    def get_action(self, action, *args):
        vals = []
        idx = action
        for vec in args:
            vals.append(vec[torch.arange(vec.size(0)), idx.squeeze().long()])
        return vals

class HackedStateCounterfactualOption(DiscreteCounterfactualOption): # eventually, we will have non-hacked StateCounterfactualOption
    def __init__(self, policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False):
        super().__init__(policy, behavior_policy, termination, next_option, dataset_model, environment_model, reward, object_name, names, temp_ext=False)
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
                temp = self.environment_model.get_factored_state(typed=False)
                if type(self.next_option) is not PrimitiveOption:
                    temp[self.next_option.object_name] += action[0] # the action should be a masked expected change in state of the last object
                else:
                    temp[self.next_option.object_name][-1] = action[0]
                features = self.dataset_model.featurize(np.expand_dims(self.environment_model.flatten_factored_state(temp, typed = False), axis=0))
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

option_forms = {"discrete": DiscreteCounterfactualOption, "continuousGaussian": ContinuousGaussianCounterfactualOption, 
"continuousParticle": ContinuousParticleCounterfactualOption, "raw": RawOption, "hacked": HackedStateCounterfactualOption}