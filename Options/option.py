import numpy as np
import os, cv2, time
import torch
import gym
from Networks.network import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from EnvironmentModels.environment_model import FeatureSelector, discretize_actions
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


INPUT_STATE = 0
OUTPUT_STATE = 1
PARAM_STATE = 2

FULL = 0
FEATURIZED = 1
DIFF = 2


class Option():
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, discretize_acts=False, device=-1):
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
        self.range_limiter = relative_actions
        self.discrete = False
        self.iscuda = False
        self.device = device
        self.use_mask = True
        self.param_process = None # must be assigned externally
        self.last_factor = self.get_state(form=1, inp=1)
        self.last_act = None
        self.policy_batch = None # the policy batch for previous

        self.discretize_actions = discretize_acts # convert a continuous state into a discrete one
        # print(self.get_state(form=0))
        self.inter_shape = self.get_state(form=FEATURIZED, inp=INPUT_STATE).shape
        self.rel_shape = (0,) if not self.relative_state else self.get_relative(inp=INPUT_STATE).shape
        print(self.get_state(form=0))
        if self.sampler:
            self.param, self.mask = self.sampler.sample(self.get_state(form=0))
            self.param_shape = self.param.shape
        else:
            self.param_shape = self.inter_shape
            self.param, self.mask = np.array([1]), np.array([1])
        self.input_shape = self.get_state(form=FEATURIZED, rel=self.relative_state, inp=PARAM_STATE).shape
        # print("last_factor", self.last_factor)
        # parameters for temporal extension TODO: move this to a single function
        self.temp_ext = temp_ext 
        self.last_action = None
        self.terminated = True
        self.time_cutoff = -1
        self.timer = 0
        self.resample_timer = 0 # tracking when the last resample was
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


    def get_relative(self, flat = None, full_state=None, inp=0):
        if flat is None:
            if full_state is None:
                full_state = self.environment_model.get_state()
            factored_state = full_state['factored_state']
            flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
            base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
            n_unsqueeze = len(base_shape) - 1
            for _ in range(n_unsqueeze):
                flat = np.expand_dims(flat, 0)
        featurize = self.gamma_featurizer if (inp == 0 or inp == 2) else self.delta_featurizer
        rel = featurize.get_relative(flat) # there might be some difficulty managing concatenate for relative state
        return rel
    # def get_flattened_input_state(self, factored_state):
    #     return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)
    def get_state(self, full_state=None, form=1, inp=0, rel=0, param=None): # param is expected 
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # param can either be None (add a dummy param), a list of the same length as full_state, or a param dimensional numpy array
        if full_state is None:
            full_state = self.environment_model.get_state()
        if type(full_state) is list or type(full_state) is np.ndarray:
            # if type(param) is list or type(param) is np.ndarray:
            #     return np.array([self.get_single_state(f, form=form, inp=inp, param=p) for f,p in zip(full_state, param)])
            if param is not None:
                return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param=param.copy()) for f in full_state])
            return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param=param) for f in full_state])
        else: # assume it is a dict
            return self.get_single_state(full_state, form=form, inp=inp, rel=rel, param=param)

    def get_single_state(self, full_state, form=1, inp=0, rel= 0, param=None):
        factored_state = full_state['factored_state']
        featurize = self.gamma_featurizer if (inp == 0 or inp == 2) else self.delta_featurizer
        comb_param = lambda x: self.add_param(x, param) if (inp == 2) else x

        if form == 0:
            return self.environment_model.flatten_factored_state(factored_state, instanced=True)
        elif form == 1:
            '''
            concatenation occurs in the order: relative, gamma, param
            '''
            # if inp == 2:
                # print(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True)), param)
                # print("getting combined", comb_param(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True))))
            base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
            n_unsqueeze = len(base_shape) - 1
            flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
            # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
            for _ in range(n_unsqueeze):
                flat = np.expand_dims(flat, 0)
            state = comb_param(featurize(flat))
            # print(state.shape)
            # print("cat param", state.shape, rel)
            if rel == 1:
                rel = self.get_relative(flat=flat, inp=inp) # there might be some difficulty managing concatenate for relative state
                if len(state.shape) == 1:
                    state = np.concatenate((rel, state), axis=0)
                else:
                    state = np.concatenate((rel, state), axis=1)
                # print("cat rel", rel.shape, state.shape)
            return state
        else:
            return self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state, instanced=True)) - self.last_factor

    def strip_param(self, combined):
        '''
        TODO: only handles stripping concatenated state with one dimension
        TODO: name is slightly confusing, strips BOTH param and relative state
        '''
        if self.param_process is None:
            if len(combined.shape) > 1:
                return combined[:,  self.rel_shape[0]:self.inter_shape[0] + self.rel_shape[0]]
            return combined[self.rel_shape[0]:self.inter_shape[0] + self.rel_shape[0]]
        return combined

    def assign_param(self, state, param):
        '''
        similar to add_param, but for a state which is already added
        TODO: assumes that param_process is an inplace operation
        '''
        if len(param.shape) != len(state.shape): # assume that state is batch and param is single
            param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
        if self.param_process is None:
            # print(state.shape, param.shape, self.inter_shape, self.rel_shape, self.relative_state)
            if len(state.shape) == 1:
                state[self.inter_shape[0] + self.rel_shape[0]:] = param
            else:
                state[:, self.inter_shape[0] + self.rel_shape[0]:] = param # default to was concatenated
        else:
            state = self.param_process(state, param)
        return state

    def add_param(self, state, param):
        '''
        only handles single states and batches
        '''
        if param is None: # insert a dummy param
            param = np.zeros(self.param_shape)
        if len(param.shape) != len(state.shape): # assume that state is batch and param is single
            param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
        if self.param_process is None:
            if len(state.shape) == 1:
                param_process = lambda x,y: np.concatenate((x,y), axis=0)
            else:
                param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
        else:
            param_process = self.param_process
        return param_process(state, param)

    def get_param(self, full_state, last_done, force=False):
        # print(self.timer, last_done)
        new_param = False
        if last_done or self.timer == 0 or force:
            # print("resample")
            if self.object_name == 'Raw': # raw param handling is questionable at best
                self.param, self.mask = np.array([1]), np.array([1])
            else: 
                if last_done or self.timer == 0 or force: # force forces it to resample
                    self.param, self.mask = self.sampler.sample(self.get_state(full_state, form=0))
                    self.timer = 0
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
            mask = self.dataset_model.get_active_mask()
            new_param = (mask.copy())
            param = param.squeeze()
            new_param[new_param == 1] = param
            param = new_param
        return param

    def convert_relative_action(self, state, act):
        # print("act", act)
        if self.relative_actions:
            new_act = list()
            for a, cfs in zip(act, self.next_option.dataset_model.cfselectors):
                new_act.append(min(cfs.feature_range[1], max(cfs.feature_range[0], (cfs.feature_selector(state) + a).squeeze()))) # add the action, then constrain to the range
                # print("new_act", new_act, cfs.feature_selector(state), a, cfs.feature_range)
            return np.array(new_act)
        return new_act

    def map_action(self, act, resampled, batch):
        if self.discretize_actions: # if actions are discretized, then converts discrete action to continuous
            act = self.get_cont(act)
        act = self.policy.map_action(act) # usually converts from policy space to environment space (even for options)
        if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # converts relative actions maintaining value
            if resampled:
                act = self.convert_relative_action(self.next_option.get_state(batch["full_state"], form=0, inp=1), act)
                self.last_mapped_act = act
            else:
                act = self.last_mapped_act # otherwise we get a moving target problem
        return act

    def sample_action_chain(self, batch, state_chain, random=False, force=False): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
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
        if (self.temp_ext > 0 and # using temporal extension
            (not (self.resample_timer == self.temp_ext)) and # temporal extension timer ellapsed
            (self.next_option is not None and not self.next_option.terminated) # waiting for next option termination 
            and not force): # forces a new action
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
            policy_batch = None
            state = None
            mapped_act = self.map_action(act, resampled, batch)
        else:
            # batch['obs'] = self.get_state(batch['full_state'], form=FEATURIZED, inp=INPUT_STATE)
            # batch['next_obs'] = self.get_state(batch['next_full_state'], form=FEATURIZED, inp=INPUT_STATE) if 'next_full_state' in batch else None
            # print(self.object_name, self.iscuda)
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
            mapped_act = self.map_action(act, resampled, batch)
        if resampled: self.resample_timer = 0
        # print(self.iscuda, param, baction, action)
        # print(act, mapped_act)
        chain = [mapped_act]
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            param = batch['param']
            obs = batch['obs']
            param_act = self.next_option.convert_param(mapped_act)
            batch['param'] = [param_act]
            if self.next_option.object_name != "Action": batch['obs'] = self.next_option.get_state(batch["full_state"], form=1, inp=2, rel = 1 if self.next_option.relative_state else 0, param=param_act) # will always concatenate param
            # print("params", self.next_option.object_name, batch.param, param_act, param, batch.obs)
            if state_chain is None: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, None) # , random=random # TODO: only sample top level randomly
            else: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, state_chain[:-1], force=resampled) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain = rem_chain + chain
            state = rem_state_chain + [state]
            batch['param'] = param
            batch['obs'] = obs
        self.policy_batch = policy_batch
        self.last_act = act
        return act, chain, policy_batch, state, resampled

    def step(self, last_state, chain):
        # This can only be called once per time step because the state diffs are managed here
        if self.next_option is not None:
            self.next_option.step(last_state, chain[:len(chain)-1])
        self.last_action = chain[-1]
        self.last_factor = self.get_state(last_state, form=FEATURIZED, inp=OUTPUT_STATE)

    def step_timer(self, done): # TODO: does this need to handle done chains?
        # return true if the timer ellapsed
        self.timer += 1
        self.resample_timer += 1
        # print(done, self.timer)
        if done or (self.timer == self.time_cutoff and self.time_cutoff > 0):
            self.timer = 0
            return self.timer == self.time_cutoff and self.time_cutoff > 0
        return False

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=True):
        # recursively get all of the dones and rewards
        dones, rewards = list(), list()
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            last_dones, last_rewards = self.next_option.terminate_reward(state, next_state, self.next_option.convert_param(chain[-1]), chain[:len(chain)-1], needs_reward=False)
        # get the state to be entered into the termination check
        input_state = self.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
        object_state = self.get_state(next_state, form = FEATURIZED, inp=OUTPUT_STATE)
        # object_state = self.get_state(state, form = DIFF if self.data.use_diff else FEATURIZED, inp=OUTPUT_STATE) # don't predict diff state
        if mask is None:
            mask = self.dataset_model.get_active_mask()

        # assign terminated, done and reward ( might need to unwrap)
        # print("first", object_state, mask, param, input_state)
        op_done = self.termination.check(input_state, object_state * mask, param * mask, self.environment_model.get_done(next_state))
        if needs_reward:
            reward = self.reward.get_reward(input_state, object_state * mask, param * mask, self.environment_model.get_reward(next_state))
        else:
            reward = 0
        self.terminated = op_done
        done = op_done
        # environment termination overrides
        if self.environment_model.get_done(next_state):
            # print("true_termination")
            done = True
        # print(self.environment_model.get_done(state), done)
        # manage a maximum time duration to run an option, NOT used, quietly switches option
        if self.time_cutoff > 0:
            if self.timer == self.time_cutoff - 1:
                # print("timer termination")
                done = True
        dones = last_dones + [done]
        rewards = last_rewards + [reward]
        # print(self.object_name, done, op_done, reward, object_state, param)
        return dones, rewards

    def tensor_state(self, factored_state): #TODO: this doesn't really belong here
        for k in factored_state.keys():
            factored_state[k] = pytorch_model.wrap(factored_state[k], cuda = self.iscuda)

    def predict_state(self, factored_state, raw_action):
        # predict the next factored state given the action chain
        # This is different only at the primitive-option level, where the state of the Action is used from the environment model
        factored_state = self.tensor_state(factored_state)
        inters, new_factored_state = self.next_option.predict_state(factored_state, raw_action)
        if self.next_option.object_name == "Action": # special handling of actions, which are evaluated IN BETWEEN states
            factored_state["Action"] = new_factored_state["Action"] 
        inter, next_state = self.dataset_model.predict_next_state(factored_state) # uses the mean, no variance
        return inters + [inter], {**new_factored_state, **{self.object_name: next_state}}


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
        policy = self.policy
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
            if clear:
                self.policy = None# removes the policy and rollouts for saving
            return policy
        return None, None

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.object_name +"_policy.pt"))
            print(self.policy)



class PrimitiveOption(Option): # primative discrete actions
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, discretize_acts=False, device=-1, action_featurizer=None):
        self.num_params = models[1].environment.num_actions
        self.object_name = "Action"
        self.action_featurizer = action_featurizer
        environment = models[1].environment
        self.action_space = models[1].environment.action_space
        self.action_shape = environment.action_space.shape or (1,)
        self.action_prob_shape = environment.action_space.shape or (1,)
        self.output_prob_shape = environment.action_space.shape or environment.action_space.n# (models[1].environment.num_actions, )
        self.discrete = environment.action_space.shape is not None
        self.discrete_actions = models[1].environment.discrete_actions
        self.control_max = environment.action_space.n if self.discrete_actions else environment.action_space.high[0]
        self.control_min = None if self.discrete_actions else environment.action_space.low[0]
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

    def sample_action_chain(self, batch, state, random=False, force=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        if random:
            chain = [self.action_space.sample()]
        else:
            param = batch['param']
            chain = [int(param.squeeze())]
        return int(param.squeeze()), chain, None, list(), True # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return [1], [0]

    def predict_state(self, factored_state, raw_action):
        new_action = copy.deepcopy(factored_state["Action"])
        new_action = self.action_featurizer.assign_feature({"Action": new_action}, )
        return [1], new_action


class RawOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, discretize_acts=False, device = -1):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions=relative_actions, relative_state=relative_state, discretize_acts=discretize_acts)
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

    def assign_param(self, state, param):
        return self.param_process(state, param)

    def get_param(self, full_state, done, force=False):
        if done or force:
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


    def sample_action_chain(self, batch, state_chain, random=False, force=False):
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
        return state["factored_state"]["Done"], state["factored_state"]["Reward"]
        # return [int(self.environment_model.environment.done or self.timer == (self.time_cutoff - 1))], [self.environment_model.environment.reward]#, torch.tensor([self.environment_model.environment.reward]), None, 1

    # The definition of this function has changed
    def get_action(self, action, mean, variance):
        idx = action
        return mean[torch.arange(mean.size(0)), idx.squeeze().long()], None#torch.log(mean[torch.arange(mean.size(0)), idx.squeeze().long()])

class ModelCounterfactualOption(Option):
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, discretize_acts=False, device=-1):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions=relative_actions, relative_state=relative_state, discretize_acts=discretize_acts)
        self.action_prob_shape = self.next_option.output_prob_shape
        if self.discrete_actions:
            self.action_shape = (1,)
        else:
            self.action_shape = self.next_option.output_prob_shape
        self.action_max = np.array(self.next_option.control_max)
        self.action_min = np.array(self.next_option.control_min)
        self.control_max = np.array(self.dataset_model.control_max)
        self.control_min = np.array(self.dataset_model.control_min)
        self.policy_min = -1 * np.ones(self.action_min.shape) # policy space is always the same
        self.policy_max = 1 * np.ones(self.action_min.shape)
        self.last_action = np.zeros(self.action_shape)
        self.last_act = np.zeros(self.action_shape)

        # if we are converting the space to be 
        if discretize_acts:
            self.discrete_dict = discretize_actions(self.action_min.shape)

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
        
        
        self.output_prob_shape = (self.dataset_model.delta.output_size(), ) # continuous, so the size will match
        # TODO: fix this so that the output size is equal to the nonzero elements of the self.dataset_model.selection_binary() at each level

        # force all previous options to have zero 
        self.next_option.set_behavior_epsilon(0)

    def set_behavior_epsilon(self, epsilon):
        if self.policy is not None:
            self.policy.set_eps(epsilon)
        if self.next_option is not None and self.next_option.policy is not None:
            self.policy.set_behavior_epsilon(epsilon)

    def get_cont(self, act):
        if self.discretize_actions:
            if type(act) == np.ndarray:
                return np.array([self.discrete_dict[a].copy() for a in act])
            return self.discrete_dict[act].copy()


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
    def __init__(self, policy_reward, models, object_name, temp_ext=False, relative_actions = -1, relative_state=False, discretize_acts=False, device=-1):
        super().__init__(policy_reward, models, object_name, temp_ext=temp_ext, relative_actions = relative_actions, relative_state=relative_state, discretize_acts=discrete_actions, device=device)
        self.sample_per = 5
        self.max_propagate = 4
        self.time_range = [0, 3] # timesteps to sample for interaction/parameter (TODO: negative not supported)
        if not self.discrete_actions: # sample all discrete actions if action space is discrete
            self.var = 10 * self.mask # samples uniformly in a range around the target position, altering the values of mask

    def single_step_search(self, center):
        # weights for possible deviations uniformly around the center, and center (zero)
        vals = np.concatenate(((np.random.rand([self.sample_per] + list(self.next_option.mask.shape)) - .5) * 2, np.zeros(1, *self.next_option.mask.shape)))
        # samples are around the center with max dev self.var but only changing masked values
        samples = center + vals * self.var * self.mask
        return samples 

    def collect(self, full_state):
        # search by calling single step search time range number of times
        all_samples = list()
        factored_state = full_state['factored_state']
        for i in range(self.time_range):
            next_factored_state = self.predict_state(factored_state, 0) # TODO: raw action hacked to no-op for now
            full_state = {"full_state": {"factored_state": next_factored_state}}
            obj_samples = self.single_step_search(self.next_option.get_state(full_state, inp=OUTPUT_STATE))
            broadcast_obj_state = torch.stack([next_factored_state[self.object_name].clone() for i in range(len(obj_samples))], dim=1)
            all_samples.append({self.object_name: broadcast_obj_state, self.next_option.object_name: obj_samples})
            factored_state = next_factored_state
        return all_samples
        # {self.object_name: torch.cat([all_samples[i][self.object_name] for i in range(len(all_samples))] dim=1), 
        # self.next_option.object_name: torch.cat([all_samples[i][self.next_option.object_name] for i in range(len(all_samples))] dim=1)}

    def search(self, batch, state_chain, target, mapped_act):
        full_state = self.propagate_state(batch, state_chain, target, mapped_act)
        factored_states = self.collect(full_state)
        for factored_state in factored_states:
            inters, preds = self.dataset_model.predict_next_state(factored_state)
            state = {"full_state": {"factored_state": factored_state}} # hopefully limited factored state is sufficient
            input_state = self.next_option.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
            object_state = preds
            # get the first action
            rewards = self.reward.check(input_state, object_state * mask, self.param * mask, 0)



    def propagate_state(self, batch, state_chain, target, mapped_act):
        '''
        roll forward until time limit or we hit the end of temporal extension, then start guessing future states
        '''
        state, next_state = batch['full_state'], batch['next_full_state'] # these should be the same at the point when this is called
        input_state = self.next_option.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
        object_state = self.next_option.get_state(next_state, form = FEATURIZED, inp=OUTPUT_STATE)
        # get the first action
        act, chain, policy_batch, state, resampled = self.sample_action_chain(batch, state_chain, preserve=True)
        
        if mask is None:
            mask = self.dataset_model.get_active_mask()
        # while we haven't reached the target location
        timer = 0
        while (timer < self.max_propagate and 
            not self.next_option.termination.check(input_state, object_state * mask, self.next_option.convert_param(chain[-1]) * mask, 0)):
            # get the next state
            factored_state = self.predict_state(factored_state, chain[0])
            input_state = self.next_option.get_state(state, form=FEATURIZED, inp=INPUT_STATE)
            object_state = self.next_option.get_state(next_state, form = FEATURIZED, inp=OUTPUT_STATE)
            # get the first action
            state = {"full_state": {"factored_state": factored_state}} # hopefully limited factored state is sufficient
            batch = copy.deepcopy(batch) # TODO: is copy inefficient?
            batch.update(full_state = state, obs = self.get_state(state, form=FEATURIZED, inp=2 if self.use_param else 0, rel=1 if self.use_rel else 0, param=self.param))
            act, chain, policy_batch, state, resampled = self.sample_action_chain(batch, state_chain, preserve=True, use_model=False)
        return state

    def sample_action_chain(self, batch, state_chain, random=False, force=False, preserve=False, use_model=True): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
        random takes a random action
        force forces the policy to decide the action
        preserve forces the policy to take the same last action
        use_model uses the model when selecting a nonrandom state
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
        if preserve or ((self.temp_ext > 0 and # using temporal extension
            (not (self.resample_timer == self.temp_ext)) and # temporal extension timer ellapsed
            (self.next_option is not None and not self.next_option.terminated) # waiting for next option termination 
            and not force)): # forces a new action
            act = self.last_act # the baction is the discrete index of the action, where the action is the parameterized form that is a parameter
            mapped_act = self.last_action
            if state_chain is None: state = None
            else: state = state_chain[-1] 
            policy_batch = self.policy_batch
            resampled = False
        elif random:
            act = self.policy_action_space.sample()
            policy_batch = None
            state = None
            mapped_act = self.map_action(act, resampled, batch)
        else:
            if state_chain is None: policy_batch = self.policy.forward(batch, None)
            else: policy_batch = self.policy.forward(batch, state_chain[-1]) # uncomment this
            state = policy_batch['state']
            act = policy_batch.act
            act = to_numpy(act)
            act = self.policy.exploration_noise(act, batch)
            act = act[0]
            # print(self.object_name, batch.obs, batch.obs_next, batch.param, act, mapped_act, self.convert_relative_action(self.next_option.get_state(batch["full_state"], form=0, inp=1), act) if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action" else 0)
            # print(self.object_name, batch.obs, act)
            # print("relative", act)
            mapped_act = self.map_action(act, resampled, batch)
        if resampled: self.resample_timer = 0
        # print(self.iscuda, param, baction, action)
        # print(act, mapped_act)
        chain = [mapped_act]
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            param = batch['param']
            obs = batch['obs']
            param_act = self.next_option.convert_param(mapped_act)
            batch['param'] = [param_act]
            if self.next_option.object_name != "Action": batch['obs'] = self.next_option.get_state(batch["full_state"], form=1, inp=2, rel = 1 if self.next_option.relative_state else 0, param=param_act) # will always concatenate param
            # print("params", self.next_option.object_name, batch.param, param_act, param, batch.obs)
            if state_chain is None: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, None) # , random=random # TODO: only sample top level randomly
            else: next_policy_act, rem_chain, result, rem_state_chain, last_resampled = self.next_option.sample_action_chain(batch, state_chain[:-1], force=resampled) # , random=random # TODO: only sample top level randomly, might need force=resampled
            chain = rem_chain + chain
            state = rem_state_chain + [state]
            batch['param'] = param
            batch['obs'] = obs
        self.policy_batch = policy_batch
        self.last_act = act
        return act, chain, policy_batch, state, resampled



option_forms = {'model': ModelCounterfactualOption, "raw": RawOption}