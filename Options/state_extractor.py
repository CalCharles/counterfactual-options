# state extractor
class StateExtractor():
    def __init__(self, args, full_state):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''
        self._gamma_featurizer = args.dataset_model.gamma
        self._delta_featurizer = args.dataset_model.delta
        self._flatten_factored_state = args.environment_model.flatten_factored_state
        self._action_feature_selector = args.action_feature_selector # option.next_option.dataset_model.feature_selector

        self.obs_setting = args.obs_setting

        # get the amount each component contributes to the input observation
        inter, target, flat, use_param, param_relative, relative, action = self.obs_setting
        self.obs_inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0)).shape
        self.obs_target_shape = self.get_state(full_state, (0,target,0,0,0,0,0)).shape
        self.obs_flat_shape = self.get_state(full_state, (0,0,flat,0,0,0,0)).shape
        self.pre_param = self.obs_inter_shape[0] + self.obs_target_shape[0] + self.obs_flat_shape[0]
        self.obs_param_shape = self.get_state(full_state, (0,0,0,use_param,0,0,0)).shape
        self.obs_param_rel_shape = self.get_state(full_state, (0,0,0,0,param_relative,0,0)).shape
        self.obs_relative_shape = self.get_state(full_state, (0,0,0,0,0,relative,0)).shape
        self.obs_action_shape = self.get_state(full_state, (0,0,0,0,0,0,action)).shape



    def get_obs(self, full_state, param):
        return self.get_state(full_state, self.obs_setting, param)

    def get_target(self, full_state):
        return self.get_state(full_state, (0,1,0,0,0,0,0))

    def get_inter(self, full_state):
        return self.get_state(full_state, (1,0,0,0,0,0,0))

    def get_flat(self, full_state):
        return self.get_state(full_state, (0,0,1,0,0,0,0))

    def get_action(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,1))

    def get_state(self, full_state, setting, param=None): # param is expected 
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # full_state is a batch or dict containing raw_state, factored_state
        # raw state may be a batch: [env_num, batch_num, raw state shape]
        # factored_state may also be a batch/dict: {name: [env_num, batch_num, factored_shape]}
        inter, target, flat, use_param, param_relative, relative, action = setting
        factored_state = full_state["factored_state"]
        return self._combine_state(full_state, inter, target, flat, use_param, param_relative, relative, action, param=param):

    def _combine_state(self, full_state, inter, target, flat, use_param, param_relative, relative, action, param=None):
        # combines the possible components of state:
        # param relative
        # print(full_state)
        factored_state = full_state['factored_state']
        featurize = self.gamma_featurizer if (inp == 0 or inp == 2) else self.delta_featurizer
        comb_param = lambda x: self.add_param(x, param) if (inp == 2 or inp == 3) else x
        shape = factored_state["Action"].shape[:-1]

        state_comb = list()
        if inter:
            state_comb.append(self._gamma_featurizer(factored_state))
        if target:
            state_comb.append(self._delta_featurizer(factored_state))
        if flat: 
            state_comb.append(self._flatten_factored_state(factored_state, instanced=True))
        if use_param:
            state_comb.append(self._broadcast_param(shape, param)) # only handles param as a concatenation
        if param_relative:
            state_comb.append(self._relative_param(shape, factored_state, param))
        if relative:
            state_comb.append(self._get_relative(factored_state))
        if action:
            state_comb.append(self._action_feature_selector(factored_state))

        if len(state_comb) == 0:
            return np.zeros((0,))
        if len(state_comb) == 1:
            return np.concatenate(state_comb, axis=len(shape))
        return state_comb[0]

    def _broadcast_param(self, shape, param):
        if len(shape) == 0:
            return param
        if len(shape) == 1:
            return np.stack([param.copy() for i in range(shape[0])], axis=0)
        if len(shape) == 2:
            return np.stack([np.stack([param.copy() for i in range(shape[1])], axis=0) for j in range(shape[0])], axis=0)

    def _relative_param(self, shape, factored_state, param):
        target = self._delta_featurizer(factored_state)
        param = self._broadcast_param(shape, param)
        return param - target

    def _get_relative(self, factored_state):
        return self._gamma_featurizer.get_relative(factored_state)

    def assign_param(self, obs, param):
        '''
        obs may be 1, 2 or 3 dimensional vector. param should be 1d vector
        '''
        if len(obs.shape) == 1:
            obs[self.pre_param:self.param_shape[0]] = param
        elif len(obs.shape) == 2:
            obs[:, self.pre_param:self.param_shape[0]] = param
        elif len(obs.shape) == 3:
            obs[:,:,self.pre_param:self.param_shape[0]] = param

        if self.param_rel_shape[0] > 0:
            target = self.get_target(obs)
            shape = obs.shape[:-1]
            if len(obs.shape) == 1:
                obs[self.pre_param:self.param_shape[0]] = self._broadcast_param(shape, param) - target
            elif len(obs.shape) == 2:
                obs[:, self.pre_param:self.param_shape[0]] = self._broadcast_param(shape, param) - target
            elif len(obs.shape) == 3:
                obs[:,:,self.pre_param:self.param_shape[0]] = self._broadcast_param(shape, param) - target

        return obs


#         if form == 0:
#             return self.environment_model.flatten_factored_state(factored_state, instanced=True)
#         elif form == 1:
#             '''
#             concatenation occurs in the order: relative, param, gamma, param
#             '''
#             # if inp == 2:
#                 # print(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True)), param)
#                 # print("getting combined", comb_param(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True))))
#             base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
#             n_unsqueeze = len(base_shape) - 1
#             if not factored:
#                 flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
#                 # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
#                 for _ in range(n_unsqueeze):
#                     flat = np.expand_dims(flat, 0)
#             if inp == 3 or inp == 5:
#                 if len(base_shape) == 1:
#                     state = comb_param(np.zeros((0,)))
#                 else:
#                     state = comb_param(np.zeros((base_shape[0],0,)))
#             else:
#                 if factored:
#                     state = comb_param(featurize(factored_state))
#                 else:
#                     # flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
#                     # # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
#                     # for _ in range(n_unsqueeze):
#                     #     flat = np.expand_dims(flat, 0)
#                     state = comb_param(featurize(flat))
#             # print(state.shape)
#             # print("cat param", state.shape, rel)
#             if rel == 1:
#                 if factored:
#                     rel = self.get_relative(factored=factored_state, inp=inp)
#                 else:
#                     rel = self.get_relative(flat=flat, inp=inp) # there might be some difficulty managing concatenate for relative state
#                 if len(state.shape) == 1:
#                     state = np.concatenate((rel, state), axis=0)
#                 else:
#                     state = np.concatenate((rel, state), axis=1)
#             if param_rel > 0: # TODO: add mask to relative param
#                 # print(factored_state)
#                 if factored:
#                     os = self.delta_featurizer(factored_state)
#                 else:
#                     os = self.delta_featurizer(flat)
#                 # print(os, param)
#                 param = self.handle_param(os, param)
#                 param_rel =  os - param
#                 if len(state.shape) == 1:
#                     # print(state.shape, param_rel.shape)
#                     state = np.concatenate((param_rel, state), axis=0)
#                 else:
#                     state = np.concatenate((param_rel, state), axis=1)
#                 # print("cat rel", rel.shape, state.shape)
#             return state
#         else:
#             return self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state, instanced=True)) - self.last_factor

# # from raw option
#     def assign_param(self, state, param):
#         return self.param_process(state, param)

#     def get_param(self, full_state, terminate, force=False):
#         if terminate or force:
#             self.param = self.environment_model.get_param(full_state)
#             return self.param, [1], True
#         return self.param, [1], False
#     def get_input_state(self):
#         # stack = stack.roll(-1,0)
#         # stack[-1] = pytorch_model.wrap(self.environment_model.environment.frame, cuda=self.iscuda)
#         # input_state = stack.clone().detach()

#         input_state = self.get_state(self.environment_model.get_state())
#         return input_state


#     def get_state(self, full_state = None, form=1, inp=0, rel=0, param=None):
#         if not full_state: return self.environment_model.get_state()['raw_state']
#         if type(full_state) is list or type(full_state) is np.ndarray: 
#             if inp == 1:
#                 return np.array([self.environment_model.get_object(f) for f in full_state])
#             return np.array([f['raw_state'] for f in full_state])
#         else:
#             if inp == 1:
#                 return self.environment_model.get_object(full_state)
#             return full_state['raw_state']


#     def update(self, last_state):
#         # updates internal states
#         self.last_factor = self.get_state(last_state, setting=self.output_setting)
#         self.resample_timer += 1

#     def tensor_state(self, factored_state): #TODO: this doesn't really belong here
#         # might need to copy factored state
#         for k in factored_state.keys():
#             factored_state[k] = pytorch_model.wrap(factored_state[k], cuda = self.iscuda)
#             if len(factored_state[k].shape) > 1: # flattens only up to one extra dimension
#                 factored_state[k] = factored_state[k][0]
#         return factored_state

#     def np_state(self, factored_state): #TODO: this doesn't really belong here
#         for k in factored_state.keys():
#             factored_state[k] = pytorch_model.unwrap(factored_state[k])
#             if len(factored_state[k].shape) > 1: # flattens only up to one extra dimension
#                 factored_state[k] = factored_state[k][0]
#         return factored_state


#     def get_relative(self, flat = None, factored=None, full_state=None, inp=0):
#         state = flat
#         if flat is None:
#             if factored is None:
#                 if full_state is None:
#                     full_state = self.environment_model.get_state()
#                 factored_state = full_state['factored_state']
#                 flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
#                 base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
#                 n_unsqueeze = len(base_shape) - 1
#                 for _ in range(n_unsqueeze):
#                     flat = np.expand_dims(flat, 0)
#                 state = flat
#             else:
#                 state = factored
#         featurize = self.gamma_featurizer if (inp == 0 or inp == 2 or inp == 3) else self.delta_featurizer
#         rel = featurize.get_relative(state) # there might be some difficulty managing concatenate for relative state
#         return rel
    

#     def get_action: gets the action space of the state
#     # def get_flattened_input_state(self, factored_state):
#     #     return pytorch_model.wrap(self.environment_model.get_flattened_state(names=self.names), cuda=self.iscuda)
#     def get_state(self, full_state=None, setting = (1,0,0,0), param=None, factored=False): # param is expected 
#         # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
#         # inp indicates if gamma or delta or gamma+param (if param is not None)
#         # param can either be None (add a dummy param), a list of the same length as full_state, or a param dimensional numpy array
#         # factored indicates if the state should NOT be flattened because only the factored components are given
#         form, inp, rel, param_rel = setting
#         if full_state is None:
#             full_state = self.environment_model.get_state()
#         if type(full_state) is list or type(full_state) is np.ndarray:
#             # if type(param) is list or type(param) is np.ndarray:
#             #     return np.array([self.get_single_state(f, form=form, inp=inp, param=p) for f,p in zip(full_state, param)])
#             if param is not None:
#                 return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param.copy(), factored=factored) for f in full_state])
#             return np.array([self.get_single_state(f, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param, factored=factored) for f in full_state])
#         else: # assume it is a dict
#             return self.get_single_state(full_state, form=form, inp=inp, rel=rel, param_rel=param_rel, param=param, factored=factored)

#     def get_single_state(self, full_state, form=1, inp=0, rel= 0, param_rel=0, param=None, factored=False):
#         # print(full_state)
#         factored_state = full_state['factored_state']
#         featurize = self.gamma_featurizer if (inp == 0 or inp == 2) else self.delta_featurizer
#         comb_param = lambda x: self.add_param(x, param) if (inp == 2 or inp == 3) else x

#         if form == 0:
#             return self.environment_model.flatten_factored_state(factored_state, instanced=True)
#         elif form == 1:
#             '''
#             concatenation occurs in the order: relative, param, gamma, param
#             '''
#             # if inp == 2:
#                 # print(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True)), param)
#                 # print("getting combined", comb_param(featurize(self.environment_model.flatten_factored_state(factored_state, instanced=True))))
#             base_shape = np.array(factored_state[list(factored_state.keys())[0]]).shape
#             n_unsqueeze = len(base_shape) - 1
#             if not factored:
#                 flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
#                 # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
#                 for _ in range(n_unsqueeze):
#                     flat = np.expand_dims(flat, 0)
#             if inp == 3 or inp == 5:
#                 if len(base_shape) == 1:
#                     state = comb_param(np.zeros((0,)))
#                 else:
#                     state = comb_param(np.zeros((base_shape[0],0,)))
#             else:
#                 if factored:
#                     state = comb_param(featurize(factored_state))
#                 else:
#                     # flat = self.environment_model.flatten_factored_state(factored_state, instanced=True)
#                     # # print("flat", flat.shape, n_unsqueeze, base_shape, factored_state["Ball"])
#                     # for _ in range(n_unsqueeze):
#                     #     flat = np.expand_dims(flat, 0)
#                     state = comb_param(featurize(flat))
#             # print(state.shape)
#             # print("cat param", state.shape, rel)
#             if rel == 1:
#                 if factored:
#                     rel = self.get_relative(factored=factored_state, inp=inp)
#                 else:
#                     rel = self.get_relative(flat=flat, inp=inp) # there might be some difficulty managing concatenate for relative state
#                 if len(state.shape) == 1:
#                     state = np.concatenate((rel, state), axis=0)
#                 else:
#                     state = np.concatenate((rel, state), axis=1)
#             if param_rel > 0: # TODO: add mask to relative param
#                 # print(factored_state)
#                 if factored:
#                     os = self.delta_featurizer(factored_state)
#                 else:
#                     os = self.delta_featurizer(flat)
#                 # print(os, param)
#                 param = self.handle_param(os, param)
#                 param_rel =  os - param
#                 if len(state.shape) == 1:
#                     # print(state.shape, param_rel.shape)
#                     state = np.concatenate((param_rel, state), axis=0)
#                 else:
#                     state = np.concatenate((param_rel, state), axis=1)
#                 # print("cat rel", rel.shape, state.shape)
#             return state
#         else:
#             return self.delta_featurizer(self.environment_model.flatten_factored_state(factored_state, instanced=True)) - self.last_factor

#     def strip_param(self, combined):
#         '''
#         TODO: only handles stripping concatenated state with one dimension
#         TODO: name is slightly confusing, strips BOTH param and relative states
#         '''
#         if self.param_process is None: # param process is not none would mean that the environment handles things like this
#             if len(combined.shape) > 1:
#                 if self.param_first:
#                     return combined[:, self.param_shape[0]:]
#                 return combined[:,  self.rel_shape[0] + self.param_rel_shape[0]:self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]]
#             if self.param_first:
#                 return combined[self.param_shape[0]:]
#             return combined[self.rel_shape[0] + self.param_rel_shape[0]:self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]]
#         return combined

#     def assign_param(self, state, param, obj_state=None):
#         '''
#         similar to add_param, but for a state which is already added
#         TODO: assumes that param_process is an inplace operation
#         '''
#         if len(param.shape) != len(state.shape): # assume that state is batch and param is single
#             param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
#         if self.param_process is None:
#             # print(state.shape, param.shape, self.inter_shape, self.rel_shape, self.relative_state)
#             inter_rel = self.inter_shape[0] + self.rel_shape[0]
#             prel = self.param_rel_shape[0]
#             if len(state.shape) == 1:
#                 if self.param_first: # param is at the beginning
#                     state[:self.param_shape[0]] = param
#                     inter_rel += self.param_shape[0]
#                 else: # param is at the end
#                     state[inter_rel + prel:] = param
#                 if self.param_rel_shape[0] > 0 and obj_state is not None:
#                     state[inter_rel:inter_rel + prel] = obj_state - param 
#             else:
#                 if self.param_first:
#                     state[:, :self.param_shape[0]] = param
#                     inter_rel += self.param_shape[0]
#                 else:
#                     state[:, self.inter_shape[0] + self.rel_shape[0] + self.param_rel_shape[0]:] = param # default to was concatenated
#                 if self.param_rel_shape[0] > 0 and obj_state is not None:
#                     state[:, inter_rel:inter_rel + prel] = obj_state - param 
#         else:
#             state = self.param_process(state, param)
#         return state

#     def handle_param(self, state, param):
#         if param is None: # insert a dummy param
#             param = np.zeros(self.param_shape)
#         if len(param.shape) != len(state.shape): # assume that state is batch and param is single
#             param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
#         return param        

#     def add_param(self, state, param):
#         '''
#         only handles single states and batches
#         '''
#         param = self.handle_param(state, param)
#         # if param is None: # insert a dummy param
#         #     param = np.zeros(self.param_shape)
#         # if len(param.shape) != len(state.shape): # assume that state is batch and param is single
#         #     param = np.stack([param.copy() for i in range(state.shape[0])], axis=0)
#         if self.param_process is None:
#             if len(state.shape) == 1:
#                 param_process = lambda x,y: np.concatenate((x,y), axis=0)
#             else:
#                 param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
#         else:
#             param_process = self.param_process
#         if self.param_first:
#             return param_process(param, state)
#         return param_process(state, param)


#     def convert_param(self, param): # TODO: only handles single params at a time
#         if self.discrete:
#             if type(param) == np.ndarray: param = int(param.squeeze())
#             if type(param) == torch.tensor: param = param.squeeze().long()
#             return self.get_possible_parameters()[param][0]
#         else:
#             if self.object_name == "Action":
#                 mask = self.action_mask
#             else:
#                 mask = self.dataset_model.get_active_mask()
#             new_param = (mask.copy())
#             param = param.squeeze()
#             # print(mask, new_param, param)
#             new_param[new_param == 1] = param
#             param = new_param
#         return param

#     def convert_relative_action(self, state, act):
#         # print("act", act)
#         if self.relative_actions:
#             new_act = list()
#             for a, cfs in zip(act, self.next_option.dataset_model.cfselectors):
#                 cfs.feature_selector(state) + a
#                 new_act.append(min(cfs.feature_range[1], max(cfs.feature_range[0], (cfs.feature_selector(state) + a).squeeze()))) # add the action, then constrain to the range
#                 # print("new_act", new_act, cfs.feature_selector(state), a, cfs.feature_range)
#             return np.array(new_act)
#         return new_act

#     def reverse_relative_action(self, state, act):
#         # print("act", act)
#         new_act = list()
#         for a, cfs in zip(act, self.next_option.dataset_model.cfselectors):
#             new_act.append(cfs.feature_selector(state) - a)
#             # new_act.append(min(cfs.feature_range[1], max(cfs.feature_range[0], (cfs.feature_selector(state) + a).squeeze()))) # add the action, then constrain to the range
#             # print("new_act", new_act, cfs.feature_selector(state), a, cfs.feature_range)
#         return np.array(new_act)


#     def map_action(self, act, resampled, batch):
#         if self.discretize_actions: # if actions are discretized, then converts discrete action to continuous
#             act = self.get_cont(act)
#         act = self.policy.map_action(act) # usually converts from policy space to environment space (even for options)
#         if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # converts relative actions maintaining value
#             if resampled:
#                 act = self.convert_relative_action(self.next_option.get_state(batch["full_state"], setting=self.full_flat_setting), act)
#                 self.last_mapped_act = act
#             else:
#                 act = self.last_mapped_act # otherwise we get a moving target problem
#         return act

#     def reverse_map_action(self, mapped_act, batch):
#         if self.relative_actions and self.next_option is not None and self.next_option.object_name != "Action": # converts relative actions maintaining value
#             mapped_act = self.reverse_relative_action(self.next_option.get_state(batch["full_state"], setting=self.full_flat_setting), mapped_act)
#         act = self.policy.reverse_map_action(mapped_act) # usually converts from policy space to environment space (even for options)
#         if self.discretize_actions: # if actions are discretized, then converts discrete action to continuous
#             act = self.get_discrete(act)
#         return act


#     def get_input_state(self, state = None): # gets the state used for the forward model/policy
#         if state is not None:
#             input_state = self.gamma_featurizer(self.pytorch_model.wrap(environment_model.get_flattened_state(), cuda=args.cuda))
#         else:
#             input_state = self.gamma_featurizer(self.pytorch_model.wrap(environment_model.flatten_factored_state(state)))
#         return input_state
