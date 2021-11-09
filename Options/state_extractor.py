import numpy as np

def array_state(factored_state):
    # converts a factored state into numpy arrays from lists (not tensors)
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state

breakout_action_norm = (np.array([0,0,0,0,1.5]), np.array([1,1,1,1,1.5]))
breakout_paddle_norm = (np.array([72, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_state_norm = (np.array([84 // 2, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_block_norm = (np.array([32, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_relative_norm = (np.array([0,0,0,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_paddle_ball_norm = (np.array([20,0,1.5,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_ball_block_norm = (np.array([20,0,-1.5,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))

# .10, -.31
# .21, -.31
# .915, .83

# -.105, .2
# -.05, .26
# .8725, .0425
robopush_action_norm = (np.array([0,0,0]), np.array([1,1,1]))
robopush_gripper_norm = (np.array([-.105,-.05,.8725]), np.array([.2,.26,.0425]))
robopush_state_norm = (np.array([-.105,-.05,.824]), np.array([.2,.26,.001]))
robopush_relative_norm = (np.array([0,0,0]), np.array([.2,.26,.0425]))
robopush_gripper_block_norm = (np.array([0,0,0.03]), np.array([.2,.26,.0425]))

def hardcode_norm_inter(anorm, v1norm, v2norm, hardcoded_normalization):
    if hardcoded_normalization[1] == '1':
        mean = np.concatenate([anorm[0], v1norm[0]])
        var = np.concatenate([anorm[1], v1norm[1]])
    elif hardcoded_normalization[1] == '2':
        mean = np.concatenate([anorm[0], v1norm[0], v2norm[0]])
        var = np.concatenate([anorm[1], v1norm[1], v2norm[1]])
    elif hardcoded_normalization[1] == '3':
        mean = np.concatenate([v1norm[0], v2norm[0]])
        var = np.concatenate([v1norm[1], v2norm[1]])
    else:
        mean = np.concatenate([v2norm[0], v2norm[0]])
        var = np.concatenate([v2norm[1], v2norm[1]])
    return mean, var

def hardcode_norm_option(hardcoded_normalization, anorm, v1norm, v2norm):
    if hardcoded_normalization[1] == '1':
        mean = anorm[0]
        var = anorm[1]
    else:
        mean = v1norm[0]
        var = v1norm[1]
    return mean, var

def hardcode_norm_target(hardcoded_normalization, anorm, v1norm, v2norm):
    if hardcoded_normalization[1] == '1':
        mean = v1norm[0]
        var = v1norm[1]
    else:
        mean = v2norm[0]
        var = v2norm[1]
    return mean, var

def hardcode_norm_param(get_mask_param, hardcoded_normalization, mask, v1norm, v2norm):
    if hardcoded_normalization[1] == '1':
        mean = get_mask_param(v1norm[0], mask)
        var = v1norm[1]
    else:
        mean = get_mask_param(v2norm[0], mask)
        var = v2norm[1]
    return mean, var

def hardcode_norm_relative(hardcoded_normalization, bnorm, v1norm, v2norm):
    if hardcoded_normalization[1] == '1':
        mean = bnorm[0]
        var = bnorm[1]
    if hardcoded_normalization[1] == '2' or hardcoded_normalization[1] == '3':
        mean = v1norm[0]
        var = v1norm[1]
    else:
        mean = v2norm[0]
        var = v2norm[1]
    return mean, var


# state extractor
class StateExtractor():
    def __init__(self, args, option_selector, full_state, param, mask):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''
        self.use_parametrized = not args.true_environment
        self.use_pair_gamma = args.use_pair_gamma
        if not args.true_environment:
            self._pair_featurizer = args.environment_model.create_entity_selector(args.name_pair)
        self._gamma_featurizer = args.dataset_model.gamma
        self._option_featurizer = option_selector # selects the TAIL
        self._delta_featurizer = args.dataset_model.delta
        self._flatten_factored_state = args.environment_model.flatten_factored_state
        self._action_feature_selector = args.action_feature_selector # option.next_option.dataset_model.feature_selector
        self.combine_param_mask = not args.no_combine_param_mask
        self.hardcoded_normalization = args.hardcode_norm
        self.scale = float(self.hardcoded_normalization[2]) if len(self.hardcoded_normalization) > 0 else 1
        self.use_breakout_pair_obs = args.use_breakout_pair_observations

        self.obs_setting = args.observation_setting
        self.update(full_state)

        # get the amount each component contributes to the input observation
        inter, target, flat, use_param, param_relative, relative, action, option, diff, raw = self.obs_setting
        self.inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0,0,0,0)).shape
        self.target_shape = self.get_state(full_state, (0,target,0,0,0,0,0,0,0,0)).shape
        self.flat_shape = self.get_state(full_state, (0,0,flat,0,0,0,0,0,0,0)).shape
        obs_inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0,0,0,0), use_pair=self.use_pair_gamma).shape
        self.pre_param = obs_inter_shape[0] + self.target_shape[0] + self.flat_shape[0]
        self.param_shape = self.get_state(full_state, (0,0,0,use_param,0,0,0,0,0,0), param=param, mask=mask).shape
        self.param_rel_shape = self.get_state(full_state, (0,0,0,0,param_relative,0,0,0,0,0), param=param, mask=mask).shape
        self.relative_shape = self.get_state(full_state, (0,0,0,0,0,relative,0,0,0,0)).shape
        self.action_shape = self.get_state(full_state, (0,0,0,0,0,0,action,0,0,0)).shape
        self.option_shape = self.get_state(full_state, (0,0,0,0,0,0,0,option,0,0)).shape
        self.diff_shape = self.get_state(full_state, (0,0,0,0,0,0,0,0,diff,0)).shape
        self.obs_shape = self.get_obs(full_state, param=param, mask=mask).shape

        self.first_obj_shape = self.option_shape
        self.object_shape = self.target_shape


    def get_raw(self, full_state):
        return full_state["raw_state"]

    def get_obs(self, full_state, param, mask=None):
        return self.get_state(full_state, self.obs_setting, param, mask, normalize=True, use_pair=self.use_pair_gamma)

    def get_target(self, full_state):
        return self.get_state(full_state, (0,1,0,0,0,0,0,0,0,0))

    def get_inter(self, full_state):
        return self.get_state(full_state, (1,0,0,0,0,0,0,0,0,0))

    def get_flat(self, full_state):
        return self.get_state(full_state, (0,0,1,0,0,0,0,0,0,0))

    def get_action(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,1,0,0,0))

    def get_first(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,0,1,0,0))

    def get_diff(self, full_state):
        return self.get_state(full_state, (0,0,0,0,0,0,0,0,1,0))

    def get_true_done(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state["Done"][-1]

    def get_true_reward(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state["Reward"][-1]

    def get_state(self, full_state, setting, param=None, mask=None, normalize = False, use_pair=False): # param is expected
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # full_state is a batch or dict containing raw_state, factored_state
        # raw state may be a batch: [env_num, batch_num, raw state shape]
        # factored_state may also be a batch/dict: {name: [env_num, batch_num, factored_shape]}
        inter, target, flat, use_param, param_relative, relative, action, option, diff, raw = setting
        return self._combine_state(full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, raw, param=param, mask=mask, normalize=normalize, use_pair=use_pair)

    def _combine_state(self, full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, raw, param=None, mask=None, normalize=False, use_pair=False):
        # combines the possible components of state:
        # param relative
        factored_state = array_state(full_state['factored_state'])
        if self.use_breakout_pair_obs:
            factored_state['Delta'] = factored_state['Ball'] - factored_state['Paddle']

        k = list(factored_state.keys())[0]
        shape = factored_state[k].shape[:-1]

        state_comb = list()

        if option: state_comb.append(self._option_featurizer(factored_state))
        if inter: state_comb.append(self._get_inter(factored_state, normalize=normalize, use_pair = use_pair))
        if target: state_comb.append(self._delta_featurizer(factored_state))
        if flat: state_comb.append(self._flatten_factored_state(factored_state, instanced=True))
        if use_param and self.use_parametrized: state_comb.append(self._broadcast_param(shape, param, mask, normalize=normalize)) # only handles param as a concatenation
        if param_relative: state_comb.append(self._relative_param(shape, factored_state, param, mask, normalize=normalize))
        if relative: state_comb.append(self._get_relative(factored_state, normalize=normalize, use_pair=use_pair))
        if action: state_comb.append(self._select_action_feature(factored_state))
        if diff: state_comb.append(self._get_diff(factored_state))
        if raw: state_comb.append(self.get_raw(full_state))

        if len(state_comb) == 0:
            return np.zeros((0,))
        else:
            return np.concatenate(state_comb, axis=len(shape))
        return state_comb[0]

    def _select_action_feature(self, factored_state):
        if type(self._action_feature_selector) == list:
            return np.concatenate([a(factored_state) for a in self._action_feature_selector], axis=-1)
        else:
            return self._action_feature_selector(factored_state)

    def _get_target(self, factored_state, normalize=False):
        unnorm_target = self._delta_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: 
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_target(breakout_action_norm, breakout_paddle_norm, breakout_state_norm, self.hardcoded_normalization)
                return (unnorm_target - mean) / var * self.scale
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_target(robopush_action_norm, robopush_gripper_norm, robopush_state_norm, self.hardcoded_normalization)
            return (unnorm_target - mean) / var * self.scale
        return unnorm_target

    def _get_option(self, factored_state, normalize=False):
        unnorm_target = self._option_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: 
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_option(breakout_action_norm, breakout_paddle_norm, breakout_state_norm, self.hardcoded_normalization)
                return (unnorm_target - mean) / var * self.scale
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_option(robopush_action_norm, robopush_gripper_norm, robopush_state_norm, self.hardcoded_normalization)
            return (unnorm_target - mean) / var * self.scale
        return unnorm_target


    def _get_inter(self, factored_state, normalize=False, use_pair=False):
        if use_pair:
            inter_state = self._pair_featurizer(factored_state)
        else:
            inter_state = self._gamma_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0:
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_inter(breakout_action_norm, breakout_paddle_norm, breakout_state_norm, self.hardcoded_normalization)
                return (inter_state - mean) / var * self.scale
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_inter(robopush_action_norm, robopush_gripper_norm, robopush_state_norm, self.hardcoded_normalization)
            return (inter_state - mean) / var * self.scale
        return inter_state


    def _get_diff(self, factored_state):
        return self._delta_featurizer(factored_state) - self.last_state

    def _broadcast_param(self, shape, param, mask, normalize=False):
        if normalize and len(self.hardcoded_normalization) > 0:
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, breakout_paddle_norm, breakout_state_norm)
                param = (param - mean) / var * self.scale
            if self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robopush_gripper_norm, robopush_state_norm)
                param = (param - mean) / var * self.scale

        if len(shape) == 0:
            return param
        if len(shape) == 1:
            return np.stack([param.copy() for i in range(shape[0])], axis=0)
        if len(shape) == 2:
            return np.stack([np.stack([param.copy() for i in range(shape[1])], axis=0) for j in range(shape[0])], axis=0)

    def _relative_param(self, shape, factored_state, param, mask, normalize=False):
        target = self._delta_featurizer(factored_state)
        param = self._broadcast_param(shape, param, mask, normalize=normalize)
        if normalize and len(self.hardcoded_normalization) > 0:
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, breakout_relative_norm, breakout_relative_norm)
                target = (target - mean) / var * self.scale
            if self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robopush_relative_norm, robopush_relative_norm)
                target = (target - mean) / var * self.scale
        return param - self.get_mask_param(target, mask)

    def _get_relative(self, factored_state, normalize = False, use_pair=False):
        if use_pair:
            rel_state = self._pair_featurizer.get_relative(factored_state)
        else:
            rel_state = self._gamma_featurizer.get_relative(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0:
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_relative(self.hardcoded_normalization, breakout_relative_norm, breakout_paddle_ball_norm, breakout_ball_block_norm)
                rel_state = (rel_state - mean) / var
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_relative(self.hardcoded_normalization, robopush_relative_norm, robopush_gripper_block_norm, robopush_relative_norm)
                rel_state = (rel_state - mean) / var
        return rel_state

    def update(self, state):
        if self.use_parametrized:
            factored_state = array_state(state['factored_state'])
            self.last_state = self._delta_featurizer(factored_state)
        else:
            self.last_state = state

    def get_mask_param(self, param, mask):
        if self.combine_param_mask:
            return param * mask
        return param

    def assign_param(self, full_state, obs, param, mask):
        '''
        obs may be 1, 2 or 3 dimensional vector. param should be 1d vector
        full state is needed for relative param, as is mask
        since we are assigning to obs, make sure this is normalized
        '''
        shape = obs.shape[:-1]
        param_norm = self._broadcast_param(shape, param, mask, normalize=True)
        if len(obs.shape) == 1:
            obs[self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        elif len(obs.shape) == 2:
            obs[:, self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        elif len(obs.shape) == 3:
            obs[:,:,self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        if self.param_rel_shape[0] > 0:
            target = self.get_mask_param(self.get_target(full_state), mask)
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, breakout_paddle_norm, breakout_state_norm)
                target = (target - mean) / var * self.scale
            if self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robopush_gripper_norm, robopush_state_norm)
                target = (target - mean) / var * self.scale
            diff = param_norm - target
            pre_rel = self.pre_param + self.param_shape[0]
            if len(obs.shape) == 1:
                obs[pre_rel:pre_rel + self.param_rel_shape[0]] = diff
            elif len(obs.shape) == 2:
                obs[:, pre_rel:pre_rel + self.param_rel_shape[0]] = diff
            elif len(obs.shape) == 3:
                obs[:,:,pre_rel:pre_rel + self.param_rel_shape[0]] = diff

        return obs
