import numpy as np
import os, cv2, time, copy
import torch
import gym
from Networks.network import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from EnvironmentModels.environment_model import FeatureSelector, discretize_actions
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from file_management import suppress_stdout_stderr

INPUT_STATE = 0
OUTPUT_STATE = 1
PARAM_STATE = 2
PARAM_NOINPUT_STATE = 3
NO_PARAM_STATE = 4
ONLY_RELATIVE_STATE = 5

FULL = 0
FEATURIZED = 1
DIFF = 2


class Option():
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, relative_param=0, 
        discretize_acts=False, device=-1, param_first=False, no_input=False):
        '''
        policy_reward is a PolicyReward object, which contains the necessary components to run a policy
        models is a tuple of dataset model and environment model
        featurizers is Featurizers object, which contains the object name, the gamma features, the delta features, and a vector representing which features are contingent
        '''
        if policy_reward is not None:
            self.assign_policy_reward(policy_reward)
        else:
            self.discrete_actions = False
        if discretize_acts: # forces discrete actions
            self.discrete_actions = True 
        self.assign_models(models)
        self.assign_featurizers()
        self.object_name = object_name
        print("init option", self.object_name)
        self.action_shape = (1,) # should be set in subclass
        self.action_prob_shape = (1,) # should be set in subclass
        self.output_prob_shape = (1,) # set in subclass
        self.control_max = None # set in subclass, maximum values for the parameter
        self.action_max = None # set in subclass, the limits for actions that can be taken
        self.action_space = None # set in subclass, the space object corresponding to all of the above information
        self.relative_action_space = None # set in subclass, space object used to set actions relative to current position
        self.relative_actions = relative_actions > 0
        self.relative_state = relative_state
        self.relative_param = relative_param
        self.range_limiter = relative_actions
        self.discrete = False
        self.iscuda = False
        self.device = device
        self.use_mask = True
        self.param_process = None # must be assigned externally
        inp_set = PARAM_NOINPUT_STATE if no_input == 1 else (NO_PARAM_STATE if no_input == 2 else (ONLY_RELATIVE_STATE if no_input == 3 else PARAM_STATE))
        print(no_input, inp_set)
        # define different settings for get_state, a tuple of (form, inp, rel, param_rel)
        print(self.relative_state, self.relative_param)
        self.input_setting = (FEATURIZED, inp_set, int(self.relative_state), int(self.relative_param))
        self.inter_setting = (FEATURIZED, INPUT_STATE, 0, 0)
        self.output_setting = (FEATURIZED, OUTPUT_STATE, 0, 0)
        self.full_flat_setting = (FULL, 0,0,0)

        self.last_factor = self.get_state(setting=self.output_setting)
        self.last_act = None
        self.policy_batch = None # the policy batch for previous

        self.discretize_actions = discretize_acts # convert a continuous state into a discrete one
        # print(self.get_state(form=0))
        self.param_first = param_first
        self.inter_shape = self.get_state(setting=self.inter_setting).shape if not no_input else (0,)
        self.output_shape = self.get_state(setting=self.output_setting).shape
        self.object_shape = self.dataset_model.object_dim
        self.rel_shape = (0,) if not self.relative_state else self.get_relative(inp=INPUT_STATE).shape
        self.param_rel_shape = (0,) if not self.relative_param else self.output_shape
        print(self.get_state(setting=self.full_flat_setting))
        if self.sampler:
            self.param, self.mask = self.sampler.sample(self.get_state(setting=self.full_flat_setting))
            print(self.param)
            self.param_shape = self.param.shape
        else:
            self.param_shape = self.inter_shape
            self.param, self.mask = np.array([1]), np.array([1])
        print(self.inter_shape, self.output_shape, self.param_shape)
        self.input_shape = self.get_state(setting=self.input_setting).shape
        self.first_obj_shape = self.inter_shape[0] - self.output_shape[0] + self.param_shape[0]  # TODO: assumes the output as part of the input
        


        # print("last_factor", self.last_factor)
        # parameters for temporal extension TODO: move this to a single function
        self.temp_ext = temp_ext 
        self.last_action = None
        self.terminated = True
        self.time_cutoff = -1 # time for ending the episode
        self.terminate_cutoff = -1 # cutoff for termination
        self.terminate_timer = 0 # tracking for terminating the current option parameter
        self.timer = 0 # tracking for end of episode
        self.resample_timer = 0 # tracking when the last resample was
        self.reward_timer = 0 # a hack to reduce the frequency of rewards
        self.reward_freq = 13

    def assign_models(self, models):
        self.dataset_model, self.environment_model, self.sampler = models

    def assign_policy_reward(self, policy_reward):
        self.policy = policy_reward.policy
        self.done_model = policy_reward.done_model
        self.termination = policy_reward.termination
        self.reward = policy_reward.reward # the reward function for this option
        self.next_option = policy_reward.next_option
        if self.next_option is not None:
            print("next option", self.next_option.object_name, self.next_option.discrete)
            self.discrete_actions = self.next_option.discrete # the action space might be discrete even if the parameter space is continuous

    def assign_featurizers(self):
        self.gamma_featurizer = self.dataset_model.gamma
        self.delta_featurizer = self.dataset_model.delta
        self.contingent_input = self.dataset_model.controllable

    def set_device(self, device_no):
        device = 'cpu' if not self.iscuda else 'cuda:' + str(device_no)
        if self.policy is not None:
            self.policy.to(device)
        if self.dataset_model is not None:
            self.dataset_model.to(device)
        if self.next_option is not None:
            self.next_option.set_device(device_no)

    def cuda(self):
        self.iscuda = True
        if self.policy is not None:
            self.policy.cuda()
        if self.dataset_model is not None:
            self.dataset_model.cuda()
        if self.next_option is not None:
            self.next_option.cuda()

    def cpu(self):
        self.iscuda = False
        if self.policy is not None:
            self.policy.cpu()
        if self.dataset_model is not None:
            self.dataset_model.cpu()
        if self.next_option is not None:
            self.next_option.cpu()


    def get_relative(self, flat = None, factored=None, full_state=None, inp=0):
        state = flat
        if flat is None:
            if factored is None:
                if full_state is None:
                    full_state = self.environment_model.get_state()
                factored_state = full_state['factored_state']
                flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
                base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
                n_unsqueeze = len(base_shape) - 1
                for _ in range(n_unsqueeze):
                    flat = np.expand_dims(flat, 0)
                state = flat
            else:
                state = factored
        featurize = self.gamma_featurizer if (inp == 0 or inp == 2 or inp == 3) else self.delta_featurizer
        rel = featurize.get_relative(state) # there might be some difficulty managing concatenate for relative state
        return rel
    
    # def get_flattened_input_state(self, factored_state):
    #     return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)
    def get_state(self, full_state=None, setting = (1,0,0,0), param=None, factored=False): # param is expected 
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # param can either be None (add a dummy param), a list of the same length as full_state, or a param dimensional numpy array
        # factored indicates if the state should NOT be flattened because only the factored components are given
        form, inp, rel, param_rel = setting
        if full_state is None:
            full_state = self.environment_model.get_state()
        if type(full_state) is list or type(full_state) is np.ndarray:
            # if type(param) is list or type(param) is np.ndarray:
            #     return np.array([self.get_single_state(f, form=form, inp=inp, param=p) for f,p in zip(full_state, param)])
            if param is not None:
                return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param.copy(), factored=factored) for f in full_state])
            return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param, factored=factored) for f in full_state])
        else: # assume it is a dict
            return self.get_single_state(full_state, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param, factored=factored)

    def get_single_state(self, full_state, form=1, inp=0, rel= 0, param_rel=0, param=None, factored=False):
        # print(full_state)
        factored_state = full_state['factored_state']
        featurize = self.gamma_featurizer if (inp == 0 or inp == 2) else self.delta_featurizer
        comb_param = lambda x: self.add_param(x, param) if (inp == 2 or inp == 3) else x

        if form == 0:
            return self.environment_model.flatten_factored_state(factored_state, instanced=True)
        elif form == 1:
            '''
            concatenation occurs in the order: relative, param, gamma, param
            '''
            # if inp == 2:
                # print(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True)), param)
                # print("getting combined", comb_param(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True))))
            base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
            n_unsqueeze = len(base_shape) - 1
            if not factored:
                flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
                # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
                for _ in range(n_unsqueeze):
                    flat = np.expand_dims(flat, 0)
            if inp == 3 or inp == 5:
                if len(base_shape) == 1:
                    state = comb_param(np.zeros((0,)))
                else:
                    state = comb_param(np.zeros((base_shape[0],0,)))
            else:
                if factored:
                    state = comb_param(featurize(factored_state))
                else:
                    # flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
                    # # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
                    # for _ in range(n_unsqueeze):
                    #     flat = np.expand_dims(flat, 0)
                    state = comb_param(featurize(flat))
            # print(state.shape)
            # print("cat param", state.shape, rel)
            if rel == 1:
                if factored:
                    rel = self.get_relative(factored=factored_state, inp=inp)
                else:
                    rel = self.get_relative(flat=flat, inp=inp) # there might be some difficulty managing concatenate for relative state
                if len(state.shape) == 1:
                    state = np.concatenate((rel, state), axis=0)
                else:
                    state = np.concatenate((rel, state), axis=1)
            if param_rel > 0: # TODO: add mask to relative param
                # print(factored_state)
                if factored:
                    os = self.delta_featurizer(factored_state)
                else:
                    os = self.delta_featurizer(flat)
                # print(os, param)
                param = self.handle_param(os, param)
                param_rel =  os - param
                if len(state.shape) == 1:
                    # print(state.shape, param_rel.shape)
                    state = np.concatenate((param_rel, state), axis=0)
                else:
                    state = np.concatenate((param_rel, state), axis=1)
                # print("cat rel", rel.shape, state.shape)
            return state
        else:
            return self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state, instanced=True)) - self.last_factor

    def strip_param(self, combined):
        '''
        TODO: only handles stripping concatenated state with one dimension
        TODO: name is slightly confusing, strips BOTH param and relative states
        '''
        if self.param_process is None: # param process is not none would mean that the environment handles things like this
            if len(combined.shape) > 1:
                if self.param_first:
                    return combined[:, self.param_shape[0]:]
                return combined[:,  self.rel_shape[0] + self.param_rel_shape[0]:self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]]
            if self.param_first:
                return combined[self.param_shape[0]:]
            return combined[self.rel_shape[0] + self.param_rel_shape[0]:self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]]
        return combined

    def assign_param(self, state, param, obj_state=None):
        '''
        similar to add_param, but for a state which is already added
        TODO: assumes that param_process is an inplace operation
        '''
        if len(param.shape) != len(state.shape): # assume that state is batch and param is single
            param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
        if self.param_process is None:
            # print(state.shape, param.shape, self.inter_shape, self.rel_shape, self.relative_state)
            inter_rel = self.inter_shape[0] + self.rel_shape[0]
            prel = self.param_rel_shape[0]
            if len(state.shape) == 1:
                if self.param_first: # param is at the beginning
                    state[:self.param_shape[0]] = param
                    inter_rel += self.param_shape[0]
                else: # param is at the end
                    state[inter_rel + prel:] = param
                if self.param_rel_shape[0] > 0 and obj_state is not None:
                    state[inter_rel:inter_rel + prel] = obj_state - param 
            else:
                if self.param_first:
                    state[:, :self.param_shape[0]] = param
                    inter_rel += self.param_shape[0]
                else:
                    state[:, self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]:] = param # default to was concatenated
                if self.param_rel_shape[0] > 0 and obj_state is not None:
                    state[:, inter_rel:inter_rel + prel] = obj_state - param 
        else:
            state = self.param_process(state, param)
        return state

    def handle_param(self, state, param):
        if param is None: # insert a dummy param
            param = np.zeros(self.param_shape)
        if len(param.shape) != len(state.shape): # assume that state is batch and param is single
            param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
        return param        

    def add_param(self, state, param):
        '''
        only handles single states and batches
        '''
        param = self.handle_param(state, param)
        # if param is None: # insert a dummy param
        #     param = np.zeros(self.param_shape)
        # if len(param.shape) != len(state.shape): # assume that state is batch and param is single
        #     param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
        if self.param_process is None:
            if len(state.shape) == 1:
                param_process = lambda x,y: np.concatenate((x,y), axis=0)
            else:
                param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
        else:
            param_process = self.param_process
        if self.param_first:
            return param_process(param, state)
        return param_process(state, param)

    def get_param(self, full_state, last_terminate, force=False):
        # print(self.timer, last_terminate)
        new_param = False
        if last_terminate or self.timer == 0 or force or self.terminate_timer == self.terminate_cutoff:
            # print("resample")
            self.terminate_timer = 0
            if self.object_name == 'Raw': # raw param handling is questionable at best
                self.param, self.mask = np.array([1]), np.array([1])
            else: 
                self.param, self.mask = self.sampler.sample(self.get_state(full_state, setting=self.full_flat_setting))
                self.terminate_timer = 0
                new_param = True
                # print(self.param, self.get_state(full_state, form=FEATURIZED, inp=OUTPUT_STATE))
            self.param, self.mask = pytorch_model.unwrap(self.param), pytorch_model.unwrap(self.mask)
        # print(self.timer, self.time_cutoff, self.param, self.get_state(full_state, form=FEATURIZED, inp=OUTPUT_STATE))
        return self.param, self.mask, new_param

    def convert_param(self, param): # TODO: only handles single params at a time
        if self.discrete:
            if type(param) == np.ndarray: param = int(param.squeeze())
            if type(param) == torch.tensor: param = param.squeeze().long()
            return self.get_possible_parameters()[param][0]
        else:
            if self.object_name == "Action":
                mask = self.action_mask
            else:
                mask = self.dataset_model.get_active_mask()
            new_param = (mask.copy())
            param = param.squeeze()
            # print(mask, new_param, param)
            new_param[new_param == 1] = param
            param = new_param
        return param

    def convert_relative_action(self, state, act):
        # print("act", act)
        if self.relative_actions:
            new_act = list()
            for a, cfs in zip(act, self.next_option.dataset_model.cfselectors):
                cfs.feature_selector(state) + a
                new_act.append(min(cfs.feature_range[1], max(cfs.feature_range[0], (cfs.feature_selector(state) + a).squeeze()))) # add the action, then constrain to the range
                # print("new_act", new_act, cfs.feature_selector(state), a, cfs.feature_range)
            return np.array(new_act)
        return new_act

    def reverse_relative_action(self, state, act):
        # print("act", act)
        new_act = list()
        for a, cfs in zip(act, self.next_option.dataset_model.cfselectors):
            new_act.append(cfs.feature_selector(state) - a)
            # new_act.append(min(cfs.feature_range[1], max(cfs.feature_range[0], (cfs.feature_selector(state) + a).squeeze()))) # add the action, then constrain to the range
            # print("new_act", new_act, cfs.feature_selector(state), a, cfs.feature_range)
        return np.array(new_act)


    def map_action(self, act, resampled, batch):
        if self.discretize_actions: # if actions are discretized, then converts discrete action to continuous
            act = self.get_cont(act)
        act = self.policy.map_action(act) # usually converts from policy space to environment space (even for options)
        if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # converts relative actions maintaining value
            if resampled:
                act = self.convert_relative_action(self.next_option.get_state(batch["full_state"], setting=self.full_flat_setting), act)
                self.last_mapped_act = act
            else:
                act = self.last_mapped_act # otherwise we get a moving target problem
        return act

    def reverse_map_action(self, mapped_act, batch):
        if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # converts relative actions maintaining value
            mapped_act = self.reverse_relative_action(self.next_option.get_state(batch["full_state"], setting=self.full_flat_setting), mapped_act)
        act = self.policy.reverse_map_action(mapped_act) # usually converts from policy space to environment space (even for options)
        if self.discretize_actions: # if actions are discretized, then converts discrete action to continuous
            act = self.get_discrete(act)
        return act


    def sample_action_chain(self, batch, state_chain, random=False, force=False, use_model=False, preserve=False): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
        use_model is only for model based search
        if the batch object contains a partial flag (key with PARTIAL=1), then treat the state as a partial
        '''
        # compute policy information for next state
        # input_state = self.get_state(state, form=FEATURIZED, inp=INPUT_STATE)

        resampled = True
        # if self.object_name == "Ball":
        #     print("resample_check", not (self.timer % self.temp_ext == 0 and self.timer != 0), # using temporal extension
        #     (self.timer), # temporal extension timer ellapsed
        #     (self.next_option is not None and not self.next_option.terminated), # waiting for next option termination 
        #     not force) # forces a new action

        # print(self.object_name, self.next_option.terminated)
        factored = False # This is really only used for the model based method
        # print(batch)
        if "PARTIAL" in batch["full_state"] and batch["full_state"]["PARTIAL"] == 1:
            # print("using factored")
            factored = True
        if preserve or (self.temp_ext > 0 and # just force it to preserve the action, and using temporal extension
            (not (self.resample_timer == self.temp_ext)) and # temporal extension timer ellapsed
            (self.next_option is not None and not self.next_option.terminated) # waiting for next option termination 
            and not force): # forces a new action
            # if self.object_name == "Block":
            #     print("last action", self.object_name, type(self.last_act), self.last_action)
            act = self.last_act # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
            mapped_act = self.last_action
            if state_chain is None: state = None
            else: state = state_chain[-1] 
            policy_batch = self.policy_batch
            resampled = False
        elif random:
            # if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # only on higher level options
            #     act = self.relative_action_space.sample()
            #     act = self.convert_relative_action(self.next_option.get_state(batch["full_state"], form=0, inp=1), act)
            # else:
            act = self.policy_action_space.sample()
            if hasattr(self, "expand_policy_space") and self.expand_policy_space and (type(act) == np.int64 or type(act) == int):
                act = np.array([act])
            # if self.policy
            policy_batch = None
            state = None
            mapped_act = self.map_action(act, resampled, batch)
            # if self.object_name == "Block":
            #     print("random", self.object_name, type(act), mapped_act)
            # print("random action", mapped_act)
        else:
            # batch['obs'] = self.get_state(batch['full_state'], form=FEATURIZED, inp=INPUT_STATE)
            # batch['next_obs'] = self.get_state(batch['next_full_state'], form=FEATURIZED, inp=INPUT_STATE) if 'next_full_state' in batch else None
            # print(self.object_name, self.iscuda)
            # print(batch)
            if state_chain is None: policy_batch = self.policy.forward(batch, None)
            else: policy_batch = self.policy.forward(batch, state_chain[-1]) # uncomment this
            state = policy_batch['state']
            act = policy_batch.act
            act = to_numpy(act)
            # print("prenoise", act)
            act = self.policy.exploration_noise(act, batch)
            act = act[0]
            # print(self.object_name, batch.obs, batch.obs_next, batch.param, act, mapped_act, self.convert_relative_action(self.next_option.get_state(batch["full_state"], form=0, inp=1), act) if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action" else 0)
            # print(self.object_name, batch.obs, act)
            # print("relative", act)
            if hasattr(self, "expand_policy_space") and self.expand_policy_space and (type(act) == np.int64 or type(act) == int):
                # print("expanding")
                act = np.array([act])
            # print(type(act), act)
            mapped_act = self.map_action(act, resampled, batch)
            if self.object_name == "Block":
                print(act, mapped_act)
            # if self.object_name == "Block":
            #     print("policy action", self.object_name, type(act), mapped_act)
            if use_model: 
                self.last_act = act # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
                self.last_action = mapped_act
                self.policy_batch = policy_batch
                act, mapped_act = self.search(batch, state_chain, act, mapped_act) # only this line differs from the main
        if resampled: self.resample_timer = 0
        # print(self.iscuda, param, baction, action)
        # print(act, random, mapped_act)
        # print("output actions", act, mapped_act)
        chain = [mapped_act]
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            param = batch['param']
            obs = batch['obs']
            param_act = self.next_option.convert_param(mapped_act)
            batch['param'] = [param_act]
            if self.next_option.object_name != "Action": batch['obs'] = self.next_option.get_state(batch["full_state"], setting=self.next_option.input_setting, param=param_act, factored=factored) # will always concatenate param
            # print("params", self.next_option.object_name, batch.param, param_act, param, batch.obs)
            # print(self.next_option.object_name, batch.obs)
            if state_chain is None: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, None) # , random=random # TODO: only sample top level randomly
            else: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, state_chain[:-1], force=resampled) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain = rem_chain + chain
            state = rem_state_chain + [state]
            batch['param'] = param
            batch['obs'] = obs
        self.policy_batch = policy_batch
        self.last_act = act
        # print(act, self.last_act, self.last_action)
        return act, chain, policy_batch, state, resampled

    def step(self, last_state, chain):
        # This can only be called once per time step because the state diffs are managed here
        if self.next_option is not None:
            self.next_option.step(last_state, chain[:len(chain)-1])
        self.last_action = chain[-1]
        self.last_factor = self.get_state(last_state, setting=self.output_setting)

    def step_timer(self, done): # TODO: does this need to handle done chains?
        # return true if the timer ellapsed
        self.timer += 1
        self.resample_timer += 1
        self.terminate_timer += 1
        # print(done, self.timer)
        if done or (self.timer == self.time_cutoff and self.time_cutoff > 0): # all timers reset if end of episode
            self.timer = 0
            self.terminate_timer = 0
            self.resample_timer = 0
            return self.timer == self.time_cutoff and self.time_cutoff > 0
        return False

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=True):
        # recursively get all of the dones and rewards
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            last_dones, last_rewards, last_termination = self.next_option.terminate_reward(state, next_state, self.next_option.convert_param(chain[-1]), chain[:len(chain)-1], needs_reward=False)
        # get the state to be entered into the termination check
        input_state = self.get_state(state, setting=self.inter_setting)
        object_state = self.get_state(next_state, setting=self.output_setting)
        # object_state = self.get_state(state, form = DIFF if self.data.use_diff else FEATURIZED, inp=OUTPUT_STATE) # don't predict diff state
        # if mask is None:
        #     mask = self.dataset_model.get_active_mask()

        # assign terminated, done and reward ( might need to unwrap)
        # print("first", object_state.shape, self.mask, param, input_state.shape)
        # print(self.environment_model.get_done(state), self.environment_model.get_done(next_state))
        # print(self.mask)
        termination = self.termination.check(input_state, object_state, param, self.mask, self.environment_model.get_done(next_state))
        # print("reward fn", self.reward)
        if needs_reward:
            reward = self.reward.get_reward(input_state, object_state, param, self.mask, self.environment_model.get_reward(next_state))
        else:
            reward = 0
        self.terminated = termination
        self.done = self.done_model.check(termination, self.timer, self.environment_model.get_done(next_state))
        # print("checking", termination, self.timer, self.done, self.termination.epsilon, self.object_name, self.termination.inter, self.termination.inter_pred)
        # done = op_done
        # # environment termination overrides
        # if self.environment_model.get_done(next_state):
        #     # print("true_termination")
        #     done = True
        # # print(self.environment_model.get_done(state), done)
        # # manage a maximum time duration to run an option, NOT used, quietly switches option
        # if self.time_cutoff > 0:
        #     if self.timer == self.time_cutoff - 1:
        #         # print("timer termination")
        #         done = True
        dones = last_dones + [self.done]
        rewards = last_rewards + [reward]
        terminations = last_termination + [termination]
        # print(self.object_name, done, op_done, reward, object_state, param)
        return dones, rewards, terminations

    def tensor_state(self, factored_state): #TODO: this doesn't really belong here
        # might need to copy factored state
        for k in factored_state.keys():
            factored_state[k] = pytorch_model.wrap(factored_state[k], cuda = self.iscuda)
            if len(factored_state[k].shape) > 1: # flattens only up to one extra dimension
                factored_state[k] = factored_state[k][0]
        return factored_state

    def np_state(self, factored_state): #TODO: this doesn't really belong here
        for k in factored_state.keys():
            factored_state[k] = pytorch_model.unwrap(factored_state[k])
            if len(factored_state[k].shape) > 1: # flattens only up to one extra dimension
                factored_state[k] = factored_state[k][0]
        return factored_state


    def predict_state(self, factored_state, raw_action):
        # predict the next factored state given the action chain
        # This is different only at the primitive-option level, where the state of the Action is used from the environment model
        factored_state = self.tensor_state(factored_state)
        inters, new_factored_state = self.next_option.predict_state(factored_state, raw_action)
        if self.next_option.object_name == "Action": # special handling of actions, which are evaluated IN BETWEEN states
            factored_state["Action"] = new_factored_state["Action"] 
        inter, next_state = self.dataset_model.predict_next_state(factored_state) # uses the mean, no variance
        return inters + [inter], {**new_factored_state, **{self.object_name: next_state}}


    # def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
    #     if self.next_option is not None:
    #         self.next_option.record_state(state, next_state, action_chain[:-1], rl_outputs[:-1], action_chain[-1], rewards[:-1], dones[:-1])
    #     self.rollouts.append(**self.get_state_dict(state, next_state, action_chain, rl_outputs, param, rewards, dones))

    # def get_state_dict(self, state, next_state, action_chain, rl_outputs, param, rewards, dones, terminations): # also used in HER
    #         return {'state': self.get_state(state, form=FEATURIZED, inp=INPUT_STATE),
    #             'next_state': self.get_state(next_state, form=FEATURIZED, inp=INPUT_STATE),
    #             'object_state': self.get_state(state, form=FEATURIZED, inp=OUTPUT_STATE),
    #             'next_object_state': self.get_state(next_state, form=FEATURIZED, inp=OUTPUT_STATE),
    #             'state_diff': self.get_state(state, form=DIFF, inp=OUTPUT_STATE), 
    #             'true_action': action_chain[0],
    #             'true_reward': rewards[0],
    #             'true_done': dones[0],
    #             'action': action_chain[-1],
    #             'probs': rl_outputs[-1].probs[0],
    #             'Q_vals': rl_outputs[-1].Q_vals[0],
    #             'param': param, 
    #             'mask': self.dataset_model.get_active_mask(), 
    #             'termination': terminations[-1], 
    #             'reward': rewards[-1], 
    #             'done': dones[-1]}


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
        policy = self.policy
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            env = self.environment_model.environment
            self.environment_model.environment = None
            self.policy.cpu() 
            self.policy.save(save_dir, self.object_name +"_policy")
            print(self.iscuda)
            if self.iscuda:
                self.policy.cuda()
            if clear:
                self.policy = None# removes the policy and rollouts for saving
            self.environment_model.environment = env
            return policy
        return None, None

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.object_name +"_policy.pt"))
            print(self.policy)



class PrimitiveOption(Option): # primative discrete actions
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, relative_param=0, 
        discretize_acts=False, device=-1, action_featurizer=None, param_first=False, no_input=False):
        self.num_params = models[1].environment.num_actions
        self.object_name = "Action"
        self.action_featurizer = action_featurizer
        environment = models[1].environment
        self.action_space = models[1].environment.action_space
        self.action_shape = environment.action_space.shape or (1,)
        self.action_prob_shape = environment.action_space.shape or (1,)
        self.output_prob_shape = environment.action_space.shape or environment.action_space.n# (models[1].environment.num_actions, )
        print(self.action_shape[0])
        self.action_mask = np.ones(self.action_shape)
        self.discrete = self.action_shape[0] == 1
        self.discrete_actions = models[1].environment.discrete_actions
        self.control_max = environment.action_space.n if self.discrete_actions else environment.action_space.high
        self.control_min = None if self.discrete_actions else environment.action_space.low
        self.action_max = environment.action_space.shape or environment.action_space.n
        self.next_option = None
        self.iscuda = False
        self.policy = None
        self.dataset_model = None
        self.time_cutoff = 1
        self.rollouts = None
        self.terminated = True


    def save(self, save_dir, clear=False):
        return None

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

    def sample_action_chain(self, batch, state, random=False, force=False, preserve=False, use_model=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        param = batch['param']
        if self.discrete_actions:
            sq_param = int(param.squeeze())
        else:
            sq_param = param.squeeze()
        if random:
            sq_param = self.action_space.sample()
        chain = [sq_param]
        return sq_param, chain, None, list(), True # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return [1], [0], [1]

    def predict_state(self, factored_state, raw_action):
        new_action = copy.deepcopy(factored_state["Action"])
        new_action = self.action_featurizer.assign_feature({"Action": new_action}, raw_action, factored=True)
        return [1], new_action


class RawOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, relative_param=0, 
        discretize_acts=False, device = -1, param_first=False, no_input=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions=relative_actions, relative_state=relative_state,
         relative_param=relative_param, discretize_acts=discretize_acts, param_first=param_first, no_input=no_input)
        self.object_name = "Raw"
        self.action_shape = (1,)
        self.action_prob_shape = (self.environment_model.environment.num_actions,)
        self.discrete_actions = self.environment_model.environment.discrete_actions
        self.action_max = self.environment_model.environment.action_space.n if self.discrete_actions else self.environment_model.environment.action_space.high
        self.action_space = self.environment_model.environment.action_space
        self.control_max = 0 # could put in "true" parameter, unused otherwise
        self.discrete = False # This should not be used, since rawoption is not performing parameterized RL
        self.use_mask = False
        self.stack = torch.zeros((4,84,84))
        # print("frame", self.environment_model.environment.get_state()['Frame'].shape)
        # self.param = self.environment_model.get_param(self.environment_model.environment.get_state()[1])
        self.param = self.environment_model.get_param(self.environment_model.environment.get_state())

    # def get_state_dict(self, state, next_state, action_chain, rl_outputs, param, rewards, dones, termination): # also used in HER
    #         return {'state': self.get_state(state, form=FEATURIZED, inp=INPUT_STATE),
    #             'next_state': self.get_state(next_state, form=FEATURIZED, inp=INPUT_STATE),
    #             'object_state': state["Object"],
    #             'next_object_state': next_state["Object"],
    #             'state_diff': state["Action"], # storing some dummy information
    #             'true_action': action_chain[0],
    #             'true_reward': rewards[0],
    #             'true_done': dones[0],
    #             'action': action_chain[-1],
    #             'probs': None if rl_outputs[-1].probs is None else rl_outputs[-1].probs[0],
    #             'Q_vals': None if rl_outputs[-1].Q_vals is None else rl_outputs[-1].Q_vals[0],
    #             'param': param, 
    #             'mask': self.dataset_model.get_active_mask(), 
    #             'termination': termination[-1],
    #             'reward': rewards[-1], 
    #             'done': dones[-1]}

    def assign_param(self, state, param):
        return self.param_process(state, param)

    def get_param(self, full_state, terminate, force=False):
        if terminate or force:
            self.param = self.environment_model.get_param(full_state)
            return self.param, [1], True
        return self.param, [1], False

    def get_possible_parameters(self):
        if self.iscuda:
            return [(torch.tensor([1]).cuda(), torch.tensor([1]).cuda())]
        return [(torch.tensor([1]), torch.tensor([1]))]

    def cuda(self):
        super().cuda()
        # self.stack = self.stack.cuda()

    def get_state(self, full_state = None, form=1, inp=0, rel=0, param=None):
        if not full_state: return self.environment_model.get_state()['raw_state']
        if type(full_state) is list or type(full_state) is np.ndarray: 
            if inp == 1:
                return np.array([self.environment_model.get_object(f) for f in full_state])
            return np.array([f['raw_state'] for f in full_state])
        else:
            if inp == 1:
                return self.environment_model.get_object(full_state)
            return full_state['raw_state']

    def get_input_state(self):
        # stack = stack.roll(-1,0)
        # stack[-1] = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
        # input_state = stack.clone().detach()

        input_state = self.get_state(self.environment_model.get_state())
        return input_state


    def sample_action_chain(self, batch, state_chain, random=False, force=False, preserve=False, use_model=False):
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
            act = act.squeeze()

        # input_state = pytorch_model.wrap(self.environment_model.get_raw_state(state), cuda=self.iscuda)
        # print("raw_state", self.environment_model.get_raw_state(state), input_state)
        # if len(param.shape) == 1:
        #     param = param.unsqueeze(0)
        # rl_output = self.policy.forward(input_state.unsqueeze(0), param) # uncomment later
        # rl_output = policy.forward(input_state.unsqueeze(0), param)
        # print("forwarded")
        # baction = self.behavior_policy.get_action(rl_output)
        chain = [act]
        # print(chain)
        return act, chain, policy_batch, state, True # resampled is always true since there is no temporal extension

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=False):
        # print(state)
        return state["factored_state"]["Done"], state["factored_state"]["Reward"], state["factored_state"]["Done"]
        # return [int(self.environment_model.environment.done or self.timer == (self.time_cutoff - 1))], [self.environment_model.environment.reward]#, torch.tensor([self.environment_model.environment.reward]), None, 1

    # The definition of this function has changed
    def get_action(self, action, mean, variance):
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()], None#torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

class ModelCounterfactualOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, relative_param=0, 
        discretize_acts=False, device=-1, param_first=False, no_input=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions=relative_actions, relative_state=relative_state,
         relative_param=relative_param, discretize_acts=discretize_acts, param_first=param_first, no_input=no_input)
        self.action_prob_shape = self.next_option.output_prob_shape
        if self.discrete_actions:
            self.action_shape = (1,)
        else:
            self.action_shape = self.next_option.output_prob_shape
        print(self.next_option.control_max)
        self.action_max = np.array(self.next_option.control_max)
        self.action_min = np.array(self.next_option.control_min)
        self.control_max = np.array(self.dataset_model.control_max)
        self.control_min = np.array(self.dataset_model.control_min)
        self.policy_min = -1 * np.ones(self.action_min.shape) # policy space is always the same
        self.policy_max = 1 * np.ones(self.action_min.shape)
        self.expand_policy_space = False
        if self.next_option.object_name != "Action":
            self.expand_policy_space = True

        # if we are converting the space to be discrete. If discretize_acts is a dict, use it directly
        if type(discretize_acts) == dict:
            self.discrete_dict = discretize_acts 
        elif discretize_acts:
            self.discrete_dict = discretize_actions(self.action_min.shape)

        if type(discretize_acts) == dict:
            acts = np.stack([v for v in self.discrete_dict.values()], axis = 0)
            self.action_min = np.min(acts, axis=0)
            self.action_max = np.max(acts, axis=0)
            self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))
            self.policy_action_shape = (1,)
            self.action_shape = self.action_min.shape
            self.action_space = gym.spaces.Box(self.action_min, self.action_max)
            self.relative_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys()))) # This should not be used
        else:
            if self.discrete_actions and discretize_acts:
                self.policy_action_space = gym.spaces.Discrete(len(list(self.discrete_dict.keys())))
            if self.discrete_actions and not discretize_acts:
                self.policy_action_space = gym.spaces.Discrete(self.next_option.control_max)
                self.action_space = gym.spaces.Discrete(self.next_option.control_max)
                self.relative_action_space = gym.spaces.Discrete(self.next_option.control_max) # no relative actions for discrete
            if (self.discrete_actions and discretize_acts) or not self.discrete_actions:
                self.action_space = gym.spaces.Box(self.action_min, self.action_max)
                rng = self.action_max - self.action_min
                self.relative_action_space = gym.spaces.Box(-rng / self.range_limiter, rng / self.range_limiter)
                print(self.action_min, self.action_max)
            if not self.discrete_actions:
                self.policy_action_space = gym.spaces.Box(self.policy_min, self.policy_max)
            self.policy_action_shape = self.policy_min.shape
        self.last_action = np.zeros(self.action_shape)
        if self.discrete_actions:
            self.last_action = np.zeros(self.action_shape)[0]
        self.last_act = np.zeros(self.policy_action_shape)
        print(self.last_action, self.last_act, self.discrete_actions, self.action_shape, self.action_min, self.control_min, self.policy_min)
        
        
        self.output_prob_shape = (self.dataset_model.delta.output_size(), ) # continuous, so the size will match
        # TODO: fix this so that the output size is equal to the nonzero elements of the self.dataset_model.selection_binary() at each level

        # force all previous options to have zero 
        self.next_option.set_behavior_epsilon(0)

    def set_behavior_epsilon(self, epsilon):
        if self.policy is not None:
            self.policy.set_eps(epsilon)
        if self.next_option is not None and self.next_option.policy is not None:
            self.next_option.set_behavior_epsilon(epsilon)

    def get_cont(self, act):
        if self.discretize_actions:
            if type(act) == np.ndarray:
                return np.array([self.discrete_dict[a].copy() for a in act])
            return self.discrete_dict[act].copy()

    def get_discrete(self, act):
        def find_closest(a):
            closest = (-1, 99999999)
            for i in range(len(list(self.discrete_dict.keys()))):
                dist = np.linalg.norm(a - np.array(self.discrete_dict[i]))
                if dist < closest[1]:
                    closest = (i,dist)
            return closest[0]
        if self.discretize_actions:
            if type(act) == np.ndarray and len(act.shape) > 1:
                return np.array([find_closest(a) for a in act])
            return find_closest(act)


    def get_action(self, action, mean, variance):
        if self.discrete_actions:
            return mean[torch.arange(mean.size(0)), action.squeeze().long()], torch.log(mean[torch.arange(mean.size(0)), action.squeeze().long()])
        idx = action
        dist = torch.distributions.normal.Normal # TODO: hardcoded action distribution as diagonal gaussian
        log_probs = dist(mean, variance).log_probs(action)
        return torch.exp(log_probs), log_probs

    def get_critic(self, state, action, mean):
        return self.policy.compute_Q(state, action)

class ForwardModelCounterfactualOption(ModelCounterfactualOption):
    ''' Uses the forward model to choose the action '''
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, relative_param=0, 
        discretize_acts=False, device=-1, param_first=False, no_input=False):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions = relative_actions, relative_state=relative_state, 
            relative_param=relative_param, discretize_acts=discretize_acts, device=device, param_first=param_first, no_input=no_input)
        self.sample_per = 5
        self.max_propagate = 3
        self.epsilon_reward = .1
        self.time_range = list(range(0, 1)) # timesteps to sample for interaction/parameter (TODO: negative not supported)
        self.uniform = True
        self.stepsize = 2
        self.use_true_model = False
        if self.use_true_model:
            self.dummy_env_model = copy.deepcopy(self.environment_model)
        if not self.discrete_actions: # sample all discrete actions if action space is discrete
            self.var = 6 # samples uniformly in a range around the target position, altering the values of mask

    def single_step_search(self, center, mask):
        # weights for possible deviations uniformly around the center, and center (zero)
        if self.uniform:
            num = (self.var * 2) // self.stepsize + 1
            vals = np.stack([np.linspace(-self.var, self.var, num ) for i in range(self.next_option.mask.shape[0])], axis=1)
            samples = center + vals * mask
        else:
            vals = (np.random.rand(*[self.sample_per] + list(self.next_option.mask.shape)) - .5) * 2
            samples = center + vals * self.var * mask
        # samples are around the center with max dev self.var but only changing masked values
        
        # print(vals * self.var, mask)
        # print(center, samples)
        return samples

    def predict_state(self, factored_state, raw_action):
        if self.use_true_model:
            expand = False
            if type(factored_state) == Batch and len(factored_state.shape) > 1:
                expand = True
                factored_state = factored_state[0] # TODO: only supports expanded by 2
            self.dummy_env_model.set_from_factored_state(factored_state)
            with suppress_stdout_stderr():
                full_state, rew, done, info = self.dummy_env_model.step(raw_action)
            if expand:
                return None, Batch([full_state['factored_state']])
            return None, Batch(full_state['factored_state'])
        else:
            return super().predict_state(factored_state, raw_action)

    def collect(self, full_state):
        # search by calling single step search time range number of times
        all_samples = list()
        all_orig = list()
        factored_state = full_state['factored_state']
        for i in self.time_range:
            # gather samples around the given state reached
            # print(factored_state)
            inters, next_factored_state = self.predict_state(factored_state, 0) # TODO: raw action hacked to no-op for now
            next_factored_state = self.np_state(next_factored_state)
            full_state = {"factored_state": next_factored_state, "PARTIAL": 1}
            center = self.next_option.get_state(full_state, setting=self.next_option.output_setting, factored=True)
            obj_samples = self.single_step_search(center, self.next_option.mask)
            # for each of the samples, broadcast the object state
            broadcast_obj_state = np.stack([next_factored_state[self.object_name].copy() for i in range(len(obj_samples))], axis=0)
            # print(next_factored_state[self.object_name].copy(), broadcast_obj_state, obj_samples)
            # factored state with only the pair of objects needed, because we ONLY forward predict the next factored state of the current object
            all_samples.append({self.object_name: broadcast_obj_state, self.next_option.object_name: obj_samples})
            all_orig.append({self.object_name: next_factored_state[self.object_name].copy(), self.next_option.object_name: center})
            factored_state = Batch(next_factored_state)

        # returns the factored states, the first is giving back the original state, but propagated for the time range, the second is the factored state for the samples
        return ( Batch({self.object_name: np.stack([all_orig[i][self.object_name] for i in range(len(all_orig))], axis=0), 
        self.next_option.object_name: np.stack([all_orig[i][self.next_option.object_name] for i in range(len(all_orig))], axis=0)}), 
        Batch({self.object_name: np.concatenate([all_samples[i][self.object_name] for i in range(len(all_samples))], axis=0), 
        self.next_option.object_name: np.concatenate([all_samples[i][self.next_option.object_name] for i in range(len(all_samples))], axis=0)}))
    
    def enumerate_rewards(self, factored_state):
        # outputs the rewards, and the action state (state that can be converted to an action by the CURRENT option)
        inters, preds = self.dataset_model.predict_next_state(factored_state)
        state = {"factored_state": factored_state, "PARTIAL": 1} # hopefully limited factored state is sufficient
        input_state = self.get_state(state, setting=self.inter_setting, factored=True)
        action_state = self.next_option.get_state(state, setting=self.next_option.output_setting, factored=True)
        # print("enum", preds, state, action_state)
        object_state = pytorch_model.unwrap(preds)
        # get the first action
        # print(input_state, object_state, self.mask, self.param)
        rewards = self.reward.get_reward(input_state, object_state, self.param, self.mask, 0)
        return action_state, rewards

    def propagate_state(self, batch, state_chain, mapped_act):
        '''
        roll forward until time limit or we hit the end of temporal extension, then start guessing future states
        '''
        state = copy.deepcopy(batch['full_state'])
        input_state = self.next_option.get_state(state, setting=self.next_option.inter_setting, factored=True)
        object_state = self.next_option.get_state(state, setting=self.next_option.output_setting, factored=True)
        # get the first action
        act, chain, policy_batch, pol_state, resampled = self.sample_action_chain(batch, state_chain, preserve=True)
        # print(mapped_act, chain[-1])
        # if self.mask is None:
        #     self.mask = self.dataset_model.get_active_mask() # doesn't use the sampler's mask
        # while we haven't reached the target location
        timer = 0
        term = False
        factored_state = state['factored_state']
        while (timer < self.max_propagate and not term):
            # get the next state
            # print("curr fac", Batch(factored_state))
            inters, factored_state = self.predict_state(factored_state, chain[0]) # TODO: add optional replacement with predictions from environment model
            factored_state = self.np_state(factored_state)
            state = {"factored_state": factored_state, "PARTIAL": 1} # hopefully limited factored state is sufficient
            input_state = self.next_option.get_state(state, setting=self.next_option.inter_setting, factored=True)
            object_state = self.next_option.get_state(state, setting=self.next_option.output_setting, factored=True)
            # get the first action
            batch = copy.deepcopy(batch) # TODO: is copy inefficient?
            batch.update(full_state = [state], obs = self.get_state(state, setting=self.input_setting, factored=True, param=self.param))
            # print(batch)
            act, chain, policy_batch, pol_state, resampled = self.sample_action_chain(batch, state_chain, preserve=True, use_model=False)
            # print("input_state", self.object_name, input_state)
            term = self.next_option.termination.check(input_state, object_state, self.next_option.convert_param(chain[-1]), self.next_option.mask, 0)
            factored_state = Batch([factored_state])
            timer += 1
            # print(timer, self.max_propagate, term, factored_state, chain)
        state = {"factored_state": factored_state, "PARTIAL": 1}
        return state

    def search(self, batch, state_chain, act, mapped_act):
        # print("starting", act, mapped_act)
        full_state = self.propagate_state(batch, state_chain, mapped_act)
        base_factored_states, sample_factored_states = self.collect(full_state)
        # print(base_factored_states, sample_factored_states)
        sample_action_state, sample_rewards = self.enumerate_rewards(sample_factored_states)
        base_action_state, base_rewards = self.enumerate_rewards(base_factored_states)
        def convert_state_to_action(obj_state):
            # TODO: assumes that obj_state is in the order of mask, which is NOT a given
            act = list()
            for cfs in self.next_option.dataset_model.cfselectors:
                # print(cfs.feature_selector(obj_state))
                act.append(cfs.feature_selector(obj_state)[0])
            return np.array(act)

        best_given_reward = np.max(base_rewards)
        best_sampled_reward = np.max(sample_rewards)
        # print(best_given_reward, best_sampled_reward)
        if best_given_reward + self.epsilon_reward > best_sampled_reward: # if no places have reward, return the action given
            # print("sending back given", act, mapped_act)
            return act, mapped_act
        else:
            best_sampled_at = np.argmax(sample_rewards)
            new_act_dict = {self.next_option.object_name: sample_action_state[best_sampled_at]}
            new_mapped_act = convert_state_to_action(new_act_dict)
            new_act = self.reverse_map_action(new_mapped_act, batch)
            # print("new_selection", new_act_dict, new_act[0], new_mapped_act, act, mapped_act)
            return new_act[0], mapped_act

    def sample_action_chain(self, batch, state_chain, random=False, force=False, use_model = False, preserve=False):
        return super().sample_action_chain(batch, state_chain, random=random, force=force, use_model=True, preserve=preserve)




option_forms = {'model': ModelCounterfactualOption, "raw": RawOption, 'forward': ForwardModelCounterfactualOption}