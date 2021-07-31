import numpy as np
import os, cv2, time, copy
import torch
import gym
from Networks.network import pytorch_model
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from EnvironmentModels.environment_model import FeatureSelector
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from file_management import suppress_stdout_stderr

class Option():
    def __init__(self, args, models, policy, next_option):
        '''
        policy_reward is a PolicyReward object, which contains the necessary components to run a policy
        models is a tuple of dataset model and environment model
        featurizers is Featurizers object, which contains the object name, the gamma features, the delta features, and a vector representing which features are contingent
        '''
        # parameters for saving
        self.name = args.object

        # primary models
        self.sampler = models.sampler # samples params
        self.policy = policy # policy to run during opion
        self.next_option = next_option # the option which controls the actions
        self.assign_models(models)

        # cuda handling
        self.iscuda = False
        self.device = args.gpu

    def assign_models(self, models):
        self.state_extractor = models.state_extractor # extracts the desired state
        self.terminate_reward = models.terminate_reward # handles termination, reward and temporal extension termination
        self.action_map = models.action_map # object dict with action spaces
        self.temporal_extension_manager = models.temporal_extension_manager # manages when to temporally extend
        self.done_model = models.done_model # decides whether to apply end of episode
        self.dataset_model = models.dataset_model # this should only be called for forward prediction
        self.initiation_set = None # TODO: handle initiation states

    def set_device(self, device_no):
        device = 'cpu' if not self.iscuda else 'cuda:' + str(device_no)
        if self.policy is not None:
            self.policy.to(device)
        # if self.sampler is not None: # TODO: I think I will need this eventually
        #     self.sampler.to(device)
        if self.next_option is not None:
            self.next_option.set_device(device_no)

    def cuda(self):
        self.iscuda = True
        if self.dataset_model is not None:
            self.dataset_model.cuda()
        if self.policy is not None:
            self.policy.cuda()
        if self.sampler is not None:
            self.sampler.cuda()
        if self.next_option is not None:
            self.next_option.cuda()

    def cpu(self):
        self.iscuda = False
        if self.dataset_model is not None:
            self.dataset_model.cpu()
        if self.policy is not None:
            self.policy.cpu()
        if self.sampler is not None:
            self.sampler.cpu()
        if self.next_option is not None:
            self.next_option.cpu()

    def print_epsilons(self):
        print("epsilons", self.terminate_reward.epsilon_close, self.terminate_reward.interaction_probability) #self.action_map.epsilon_policy)

    def _set_next_option(self, batch, mapped_act):
        param = batch['param']
        obs = batch['obs']
        mask = batch['mask']
        next_mask = self.next_option.sampler.get_mask(param) if self.next_option.sampler is not None else self.next_option.mask
        if type(self.next_option) != PrimitiveOption: param_act = self.next_option.sampler.convert_param(mapped_act) # self.next_option.state_extractor.convert_param(mapped_act) if self.next_option.state_extractor is not None else mapped_act
        else: param_act = [mapped_act]
        batch['param'] = param_act
        batch['mask'] = [next_mask]
        batch['obs'] = self.next_option.state_extractor.get_obs(batch["full_state"], param_act, next_mask) if self.next_option.state_extractor is not None else batch['obs']
        return param, obs, mask

    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False):
        '''
        get a new action (resample) or not based on the result of TEM.check. If we don't, check downstream options
        batch must contain full_state and termination_chain
        '''
        needs_sample, act, chain, policy_batch, state, masks = self.temporal_extension_manager.check(term_chain[-1], ext_terms[-1])
        if needs_sample: 
            result_tuple = self.sample_action_chain(batch, state_chain, random=random, use_model=use_model)
            if not random: self.temporal_extension_manager.update_policy(result_tuple[2], result_tuple[3][-1] if result_tuple[3] is not None else None) # result tuple  2 and 3 are policy_batch and state-chain respectively
            resampled = True
        else:
            # if we don't need a new sample
            param, obs, mask = self._set_next_option(batch, chain[-1])
            _, rem_chain, _, rem_state, rem_masks, last_resmp = self.next_option.extended_action_sample(batch, state_chain, term_chain[:-1], ext_terms[:-1], random=False, use_model=use_model)
            batch['param'], batch['obs'], batch['mask'] = param, obs, mask
            result_tuple = (act, rem_chain + [chain[-1]], policy_batch, rem_state + [state[-1]] if state is not None else None, rem_masks + [mask[0]])
            resampled = False
        return (*result_tuple, resampled)

    def sample_action_chain(self, batch, state_chain, random=False, use_model=False): # TODO: change this to match the TS parameter format, in particular, make sure that forward returns the desired components in RLOutput
        '''
        takes in a tianshou.data.Batch object and param, and runs the policy on it
        the batch object can only contain a single full state (computes one state at a time), because of handling issues
        use_model is only for model based search
        if the batch object contains a partial flag (key with PARTIAL=1), then treat the state as a partial
        '''
        if random:
            state = None
            act = self.action_map.sample_policy_space()
            act, mapped_act = self.action_map.map_action(act, batch)
            policy_batch = Batch()
        else:
            policy_batch = self.policy.forward(batch, state_chain[-1] if state_chain is not None else None) # uncomment this
            state = policy_batch.state
            act, mapped_act = self.action_map.map_action(policy_batch.act, batch)
            if use_model: 
                act, mapped_act = self.search(batch, state_chain, act, mapped_act) # only this line differs from the main
        chain = [mapped_act]
        
        # recursively propagate action up the chain
        if self.next_option is not None:
            param, obs, mask = self._set_next_option(batch, mapped_act)
            next_policy_act, rem_chain, result, rem_state_chain, last_masks = self.next_option.sample_action_chain(batch, state_chain[-1] if state_chain is not None else None) # , random=random # TODO: only sample top level randomly, if we resampled make sure not to temporally extend the next layer
            chain, state, masks = rem_chain + chain, rem_state_chain + [state], last_masks + [mask[0]] # TODO: mask is one dimension too large because batch has multiple environments
            batch['param'], batch['obs'], batch['mask'] = param, obs, mask
        return act, chain, policy_batch, state, masks

    def reset(self, full_state):
        # reset the timers for temporal extension, termination
        # does NOT reset the environment
        init_terms = self.next_option.reset(full_state)
        init_term = self.terminate_reward.reset()
        self.temporal_extension_manager.reset()
        return init_terms + [init_term]

    def update(self, buffer, done, last_state, act, chain, term_chain, param, masks, update_policy=True):
        # updates internal states of the option
        if self.next_option is not None:
            self.next_option.update(buffer, done, last_state, chain[:len(chain)-1], chain[:len(chain)-1], term_chain[:len(term_chain)-1], chain[-1], masks[:len(masks)-1], update_policy = False)
        self.temporal_extension_manager.update(act, chain, term_chain[-2], masks)
        self.state_extractor.update(last_state)
        self.terminate_reward.update(term_chain[-1])
        self.done_model.update(done)
        self.sampler.update(param, masks[-1]) # TODO: sampler also handles its own param, mask
        if update_policy:
            self.policy.update_norm(buffer)
            self.policy.update_la()

    def terminate_reward_chain(self, full_state, next_full_state, param, chain, mask, mask_chain):
        # recursively get all of the dones and rewards
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            if type(self.next_option) != PrimitiveOption:
                next_param = self.next_option.sampler.convert_param(chain[-1]) # mapped actions need to be expanded to fit param dimensions
                next_mask = mask_chain[-2]
            else:
                next_param = chain[-1]
                next_mask = mask_chain[-1]
            last_done, last_rewards, last_termination, last_ext_term, _, _ = self.next_option.terminate_reward_chain(full_state, next_full_state, next_param, chain[:len(chain)-1], next_mask, mask_chain[:len(mask_chain)-1])
        termination, reward, inter, time_cutoff = self.terminate_reward.check(full_state, next_full_state, param, mask)
        ext_term = self.temporal_extension_manager.get_extension(termination, last_termination[-1])
        done = self.done_model.check(termination, self.state_extractor.get_true_done(next_full_state))

        rewards, terminations, ext_term = last_rewards + [reward], last_termination + [termination], last_ext_term + [ext_term]
        return done, rewards, terminations, ext_term, inter, time_cutoff

    def predict_state(self, factored_state, raw_action):
        # predict the next factored state given the action chain
        # This is different only at the primitive-option level, where the state of the Action is used from the environment model
        factored_state = self.tensor_state(factored_state)
        inters, new_factored_state = self.next_option.predict_state(factored_state, raw_action)
        if self.next_option.name == "Action": # special handling of actions, which are evaluated IN BETWEEN states
            factored_state["Action"] = new_factored_state["Action"] 
        inter, next_state = self.dataset_model.predict_next_state(factored_state) # uses the mean, no variance
        return inters + [inter], {**new_factored_state, **{self.name: next_state}}

    def save(self, save_dir, clear=False):
        # checks and prepares for saving option as a pickle
        policy = self.policy
        if len(save_dir) > 0:
            try:
                os.makedirs(save_dir)
            except OSError:
                pass
            self.policy.cpu() 
            self.policy.save(save_dir, self.name +"_policy")
            if self.iscuda:
                self.policy.cuda()
            if clear:
                self.policy = None# removes the policy and rollouts for saving
            return self
        return None

    def load_policy(self, load_dir):
        if len(load_dir) > 0:
            self.policy = torch.load(os.path.join(load_dir, self.name +"_policy.pt"))

class PrimitiveOption(Option): # primitive discrete actions
    def __init__(self, args, policy):

        # parameters for saving
        self.name = "Action"

        # primary models
        self.sampler = None
        self.state_extractor = None
        self.policy = None
        self.terminate_reward = None # handles termination, reward and temporal extension termination
        self.next_option = None # the option which controls the actions
        self.action_map = args.primitive_action_map
        self.temporal_extension_manager = None # manages when to temporally extend
        self.dataset_model = None
        self.initiation_set = None # TODO: handle initiation states
        self.mask = np.ones((1,))

        # cuda handling
        self.iscuda = False


    def reset(self, full_state):
        return [True]

    def save(self, save_dir, clear=False):
        return self

    def load_policy(self, load_dir):
        pass

    def update(self, buffer, done, last_state, act, chain, term_chain, param, masks, update_policy=True):
        pass

    def cpu(self):
        self.iscuda = False

    def cuda(self):
        self.iscuda = True
    
    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False):
        return (*self.sample_action_chain(batch, state_chain, random, use_model), True)

    def sample_action_chain(self, batch, state, random=False, use_model=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        param = batch['param']
        if self.action_map.discrete_actions:
            sq_param = int(param.squeeze())
        else:
            sq_param = param.squeeze()
        if random:
            sq_param = self.action_map.sample()
        chain = [sq_param]
        return sq_param, chain, None, list(), list() # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_reward_chain(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return 1, [0], [1], list(), True, True

    def predict_state(self, factored_state, raw_action):
        new_action = copy.deepcopy(factored_state["Action"])
        new_action = self.action_featurizer.assign_feature({"Action": new_action}, raw_action, factored=True)
        return [1], new_action


class RawOption(Option):
    def __init__(self, args, models, policy, next_option):
        super().__init__(args, models, policy, next_option)

        # parameters for saving
        self.name = "Raw"

        # primary models
        self.sampler = args.environment.sampler
        self.state_extractor = models.state_extractor # state extractor for action
        self.policy = args.policy
        self.terminate_reward = None # handles termination, reward and temporal extension termination
        self.next_option = None # raw options do not have this
        self.action_spaces = args.env_action_space
        self.temporal_extension_manager = None # raw options do not have this
        self.dataset_model = None # raw options do not have this
        self.initiation_set = None # raw options do not have this

        # cuda handling
        self.iscuda = False
        self.device = device

    def cuda(self):
        super().cuda()

    def sample_action_chain(self, batch, state_chain, random=False, force=False, preserve=False, use_model=False):
        '''
        Takes an action in the state, only accepts single states. Since the initiation condition extends to all states, this is always callable
        also returns whether the current state is a termination condition. The option will still return an action in a termination state
        The temporal extension of options is exploited using temp_ext, which first checks if a previous option is still running, and if so returns the same action as before
        '''
        if random:
            act = self.action_space.sample()
            policy_batch = None
            state = None
        else:
            batch['obs'] = self.state_extractor.get_state(batch['full_state'])
            policy_batch = self.policy.forward(batch, state_chain[-1] if state_chain is not None else None)
            act, mapped_act = self.action_map.map(policy_batch.act)
        chain = [act]
        return act, chain, policy_batch, state, True, list() # resampled is always true since there is no temporal extension

    def terminate_reward(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return state["factored_state"]["Done"], state["factored_state"]["Reward"], state["factored_state"]["Done"], True, False # even if cut off with time, don't return
        # return [int(self.environment_model.environment.done or self.timer == (self.time_cutoff - 1))], [self.environment_model.environment.reward]#, torch.tensor([self.environment_model.environment.reward]), None, 1


class ModelCounterfactualOption(Option):
    def __init__(self, args, models, policy, next_option):
        super().__init__(args, models, policy, next_option)

    def get_critic(self, state, action, mean):
        return self.policy.compute_Q(state, action)

class ForwardModelCounterfactualOption(ModelCounterfactualOption):
    ''' Uses the forward model to choose the action '''
    def __init__(self, args, models, policy, next_option):
        super().__init__(args, models, policy, next_option)
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
        # collect only collects the states of the tool and target objects
        all_samples = list()
        all_orig = list()
        factored_state = full_state['factored_state']
        for i in self.time_range:
            # gather samples around the given state reached
            inters, next_factored_state = self.predict_state(factored_state, 0) # TODO: raw action hacked to no-op for now
            next_factored_state = self.np_state(next_factored_state)
            full_state = {"factored_state": next_factored_state, "PARTIAL": 1}
            center = self.next_option.get_state(full_state, setting=self.next_option.output_setting, factored=True)
            obj_samples = self.single_step_search(center, self.next_option.mask)
            # for each of the samples, broadcast the object state
            broadcast_obj_state = np.stack([next_factored_state[self.name].copy() for i in range(len(obj_samples))], axis=0)
            # factored state with only the pair of objects needed, because we ONLY forward predict the next factored state of the current object
            all_samples.append({self.name: broadcast_obj_state, self.next_option.name: obj_samples})
            all_orig.append({self.name: next_factored_state[self.name].copy(), self.next_option.name: center})
            factored_state = Batch(next_factored_state)

        # returns the factored states, the first is giving back the original state, but propagated for the time range, the second is the factored state for the samples
        return ( Batch({self.name: np.stack([all_orig[i][self.name] for i in range(len(all_orig))], axis=0), 
        self.next_option.name: np.stack([all_orig[i][self.next_option.name] for i in range(len(all_orig))], axis=0)}), 
        Batch({self.name: np.concatenate([all_samples[i][self.name] for i in range(len(all_samples))], axis=0), 
        self.next_option.name: np.concatenate([all_samples[i][self.next_option.name] for i in range(len(all_samples))], axis=0)}))
    
    def enumerate_rewards(self, factored_state):
        # outputs the rewards, and the action state (state that can be converted to an action by the CURRENT option)
        inters, preds = self.dataset_model.predict_next_state(factored_state)
        state = {"factored_state": factored_state, "PARTIAL": 1} # hopefully limited factored state is sufficient
        input_state = self.get_state(state, setting=self.inter_setting, factored=True)
        action_state = self.next_option.get_state(state, setting=self.next_option.output_setting, factored=True)
        object_state = pytorch_model.unwrap(preds)
        # get the first action
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
        # if self.mask is None:
        #     self.mask = self.dataset_model.get_active_mask() # doesn't use the sampler's mask
        # while we haven't reached the target location
        timer = 0
        term = False
        factored_state = state['factored_state']
        while (timer < self.max_propagate and not term):
            # get the next state
            inters, factored_state = self.predict_state(factored_state, chain[0]) # TODO: add optional replacement with predictions from environment model
            factored_state = self.np_state(factored_state)
            state = {"factored_state": factored_state, "PARTIAL": 1} # hopefully limited factored state is sufficient
            input_state = self.next_option.get_state(state, setting=self.next_option.inter_setting, factored=True)
            object_state = self.next_option.get_state(state, setting=self.next_option.output_setting, factored=True)
            # get the first action
            batch = copy.deepcopy(batch) # TODO: is copy inefficient?
            batch.update(full_state = [state], obs = self.get_state(state, setting=self.input_setting, factored=True, param=self.param))
            act, chain, policy_batch, pol_state, resampled = self.sample_action_chain(batch, state_chain, preserve=True, use_model=False)
            term = self.next_option.termination.check(input_state, object_state, self.next_option.convert_param(chain[-1]), self.next_option.mask, 0)
            factored_state = Batch([factored_state])
            timer += 1
        state = {"factored_state": factored_state, "PARTIAL": 1}
        return state

    def search(self, batch, state_chain, act, mapped_act):
        full_state = self.propagate_state(batch, state_chain, mapped_act)
        base_factored_states, sample_factored_states = self.collect(full_state)
        sample_action_state, sample_rewards = self.enumerate_rewards(sample_factored_states)
        base_action_state, base_rewards = self.enumerate_rewards(base_factored_states)
        def convert_state_to_action(obj_state):
            # TODO: assumes that obj_state is in the order of mask, which is NOT a given
            act = list()
            for cfs in self.next_option.dataset_model.cfselectors:
                act.append(cfs.feature_selector(obj_state)[0])
            return np.array(act)

        best_given_reward = np.max(base_rewards)
        best_sampled_reward = np.max(sample_rewards)
        if best_given_reward + self.epsilon_reward > best_sampled_reward: # if no places have reward, return the action given
            return act, mapped_act
        else:
            best_sampled_at = np.argmax(sample_rewards)
            new_act_dict = {self.next_option.name: sample_action_state[best_sampled_at]}
            new_mapped_act = convert_state_to_action(new_act_dict)
            new_act = self.reverse_map_action(new_mapped_act, batch)
            return new_act[0], mapped_act

    def sample_action_chain(self, batch, state_chain, random=False, force=False, use_model = False, preserve=False):
        return super().sample_action_chain(batch, state_chain, random=random, force=force, use_model=True, preserve=preserve)




option_forms = {'model': ModelCounterfactualOption, "raw": RawOption, 'forward': ForwardModelCounterfactualOption}