import numpy as np

def array_state(factored_state):
    # converts a factored state into numpy arrays from lists (not tensors)
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state


# state extractor
class StateExtractor():
    def __init__(self, args, option_selector, full_state, param, mask):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''
        self._gamma_featurizer = args.dataset_model.gamma
        self._option_featurizer = option_selector
        self._delta_featurizer = args.dataset_model.delta
        self._flatten_factored_state = args.environment_model.flatten_factored_state
        self._action_feature_selector = args.action_feature_selector # option.next_option.dataset_model.feature_selector
        self.combine_param_mask = not args.no_combine_param_mask

        self.obs_setting = args.observation_setting
        self.update(full_state)

        # get the amount each component contributes to the input observation
        inter, target, flat, use_param, param_relative, relative, action, option, diff = self.obs_setting
        self.inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0,0,0)).shape
        self.target_shape = self.get_state(full_state, (0,target,0,0,0,0,0,0,0)).shape
        self.flat_shape = self.get_state(full_state, (0,0,flat,0,0,0,0,0,0)).shape
        self.pre_param = self.inter_shape[0] + self.target_shape[0] + self.flat_shape[0]
        self.param_shape = self.get_state(full_state, (0,0,0,use_param,0,0,0,0,0), param=param).shape
        self.param_rel_shape = self.get_state(full_state, (0,0,0,0,param_relative,0,0,0,0), param=param, mask=mask).shape
        self.relative_shape = self.get_state(full_state, (0,0,0,0,0,relative,0,0,0)).shape
        self.action_shape = self.get_state(full_state, (0,0,0,0,0,0,action,0,0)).shape
        self.option_shape = self.get_state(full_state, (0,0,0,0,0,0,0,option,0)).shape
        self.diff_shape = self.get_state(full_state, (0,0,0,0,0,0,0,0,diff)).shape
        self.obs_shape = self.get_obs(full_state, param=param, mask=mask).shape

        self.first_obj_shape = self.option_shape
        self.object_shape = self.target_shape



    def get_obs(self, full_state, param, mask=None):
        return self.get_state(full_state, self.obs_setting, param, mask)

    def get_target(self, full_state):
        return self.get_state(full_state, (0,1,0,0,0,0,0,0,0))

    def get_inter(self, full_state):
        return self.get_state(full_state, (1,0,0,0,0,0,0,0,0))

    def get_flat(self, full_state):
        return self.get_state(full_state, (0,0,1,0,0,0,0,0,0))

    def get_action(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,1,0,0))

    def get_first(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,0,1,0))

    def get_diff(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,0,0,1))

    def get_true_done(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state["Done"][-1]



    def get_state(self, full_state, setting, param=None, mask=None): # param is expected 
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # full_state is a batch or dict containing raw_state, factored_state
        # raw state may be a batch: [env_num, batch_num, raw state shape]
        # factored_state may also be a batch/dict: {name: [env_num, batch_num, factored_shape]}
        inter, target, flat, use_param, param_relative, relative, action, option, diff = setting
        factored_state = full_state["factored_state"]
        return self._combine_state(full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, param=param, mask=mask)

    def _combine_state(self, full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, param=None, mask=None):
        # combines the possible components of state:
        # param relative
        factored_state = array_state(full_state['factored_state'])
        shape = factored_state["Action"].shape[:-1]

        state_comb = list()
        if option: state_comb.append(self._option_featurizer(factored_state))
        if inter: state_comb.append(self._gamma_featurizer(factored_state))
        if target: state_comb.append(self._delta_featurizer(factored_state))
        if flat: state_comb.append(self._flatten_factored_state(factored_state, instanced=True))
        if use_param: state_comb.append(self._broadcast_param(shape, param)) # only handles param as a concatenation
        if param_relative: state_comb.append(self._relative_param(shape, factored_state, param, mask))
        if relative: state_comb.append(self._get_relative(factored_state))
        if action: state_comb.append(self._action_feature_selector(factored_state))
        if diff: state_comb.append(self._get_diff(factored_state))

        if len(state_comb) == 0:
            return np.zeros((0,))
        else:
            return np.concatenate(state_comb, axis=len(shape))
        return state_comb[0]

    def _get_diff(self, factored_state):
        return self._delta_featurizer(factored_state) - self.last_state

    def _broadcast_param(self, shape, param):
        if len(shape) == 0:
            return param
        if len(shape) == 1:
            return np.stack([param.copy() for i in range(shape[0])], axis=0)
        if len(shape) == 2:
            return np.stack([np.stack([param.copy() for i in range(shape[1])], axis=0) for j in range(shape[0])], axis=0)

    def _relative_param(self, shape, factored_state, param, mask):
        target = self._delta_featurizer(factored_state)
        param = self._broadcast_param(shape, param)
        return param - self.get_mask_param(target, mask)

    def _get_relative(self, factored_state):
        return self._gamma_featurizer.get_relative(factored_state)

    def update(self, state):
        factored_state = array_state(state['factored_state'])
        self.last_state = self._delta_featurizer(factored_state)

    def get_mask_param(self, param, mask):
        if self.combine_param_mask:
            return param * mask
        return param

    def assign_param(self, full_state, obs, param, mask):
        '''
        obs may be 1, 2 or 3 dimensional vector. param should be 1d vector
        full state is needed for relative param, as is mask
        '''
        if len(obs.shape) == 1:
            obs[self.pre_param:self.pre_param + self.param_shape[0]] = param
        elif len(obs.shape) == 2:
            obs[:, self.pre_param:self.pre_param + self.param_shape[0]] = param
        elif len(obs.shape) == 3:
            obs[:,:,self.pre_param:self.pre_param + self.param_shape[0]] = param

        if self.param_rel_shape[0] > 0:
            target = self.get_mask_param(self.get_target(full_state), mask)
            shape = obs.shape[:-1]
            if len(obs.shape) == 1:
                obs[self.pre_param:self.pre_param + self.param_shape[0]] = self._broadcast_param(shape, param) - target
            elif len(obs.shape) == 2:
                obs[:, self.pre_param:self.pre_param + self.param_shape[0]] = self._broadcast_param(shape, param) - target
            elif len(obs.shape) == 3:
                obs[:,:,self.pre_param:self.pre_param + self.param_shape[0]] = self._broadcast_param(shape, param) - target

        return obs
