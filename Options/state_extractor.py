import numpy as np

def array_state(factored_state):
    # converts a factored state into numpy arrays from lists (not tensors)
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state

breakout_action_norm = (np.array([0,0,0,0,1.5]), np.array([1,1,1,1,1.5]))
breakout_paddle_norm = (np.array([72, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_state_norm = (np.array([84 // 2, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_block_norm = (np.array([32, 84 // 2, 0,0,0]), np.array([10, 84 // 2, 2,1,1]))
breakout_relative_norm = (np.array([0,0,0,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_paddle_ball_norm = (np.array([0,0,0,0,0]), np.array([84 // 2, 84 // 2,4,2,1]))
breakout_ball_block_norm = (np.array([20,0,-1.5,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))

# .10, -.31
# .21, -.31
# .915, .83

# -.105, .2
# -.05, .26
# .8725, .0425
robopush_action_norm = (np.array([0,0,0]), np.array([1,1,1]))
robopush_gripper_norm = (np.array([-0.1,-.05,.8725]), np.array([.2,.26,.0425]))
robopush_state_norm = (np.array([-0.1,-.00,.824]), np.array([.2,.26,.1]))
robopush_target_norm = (np.array([-0.1,0.0,.802]), np.array([.2,.26,.1]))
robopush_obstacle_norm = (np.array([-0.1,0.0,.801]), np.array([.2,.26,.1]))
robopush_relative_norm = (np.array([0,0,0]), np.array([.2,.26,.1]))
robopush_gripper_block_norm = (np.array([0,0,0.03]), np.array([.2,.26,.0425]))
robopush_block_target_relative_norm = (np.array([0,0,.023]), np.array([.2,.26,.1]))


robostick_action_norm = (np.array([0,0,0,0]), np.array([1,1,1,1]))
robostick_gripper_norm = (np.array([-0.1, 0,.9,0]), np.array([.1,.15,.1,1]))
robostick_stick_norm = (np.array([-0.1,0.0,.824]), np.array([.2,.15,.1]))
robostick_block_norm = (np.array([-0.1,0.0,.802]), np.array([.1,.15,.1]))
robostick_target_norm = (np.array([0.0,0.0,.802]), np.array([.07,.15,.1]))
robostick_gripper_relative_norm = (np.array([0,0,0,0]), np.array([.2,.3,.2,1]))
robostick_relative_norm = (np.array([0,0,0]), np.array([.2,.15,.1]))
robostick_gripper_stick_norm = (np.array([0,0,0.00]), np.array([.2,.15,.07]))
robostick_stick_block_norm = (np.array([0,0,0.03]), np.array([.2,.15,.07]))
robostick_block_target_relative_norm = (np.array([0,0,.023]), np.array([.2,.15,.1]))



def hardcode_norm_inter(anorm, v1norm, v2norm, v3norm, hardcoded_normalization, num_instance=0):
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
        mean = np.concatenate([v2norm[0]] + [ v3norm[0].copy() for _ in range(num_instance)] )
        var = np.concatenate([v2norm[1]] + [ v3norm[1].copy() for _ in range(num_instance)])
    # print("inter", hardcoded_normalization, mean, var)
    return mean, var

def hardcode_norm_option(hardcoded_normalization, anorm, v1norm, v2norm):
    if hardcoded_normalization[1] == '1':
        mean = anorm[0]
        var = anorm[1]
    elif hardcoded_normalization[1] == '2' or hardcoded_normalization[1] == '3':
        mean = v1norm[0]
        var = v1norm[1]
    else:
        mean = v2norm[0]
        var = v2norm[1]
    return mean, var

def hardcode_norm_target(hardcoded_normalization, v1norm, v2norm, v3norm):
    if hardcoded_normalization[1] == '1':
        mean = v1norm[0]
        var = v1norm[1]
    elif hardcoded_normalization[1] == '2' or hardcoded_normalization[1] == '3':
        mean = v2norm[0]
        var = v2norm[1]
    else:
        mean = v3norm[0]
        var = v3norm[1]
    return mean, var

def hardcode_norm_param(get_mask_param, hardcoded_normalization, mask, v1norm, v2norm, v3norm):
    if hardcoded_normalization[1] == '1':
        mean = get_mask_param(v1norm[0], mask)
        var = v1norm[1]
    elif hardcoded_normalization[1] == '2' or hardcoded_normalization[1] == '3':
        mean = get_mask_param(v2norm[0], mask)
        var = v2norm[1]
    else:
        mean = get_mask_param(v3norm[0], mask)
        var = v3norm[1]
    # print("param", hardcoded_normalization, mean, var)
    return mean, var

def hardcode_norm_relative(hardcoded_normalization, bnorm, v1norm, v2norm, num_instance=0):
    if hardcoded_normalization[1] == '1':
        mean = bnorm[0]
        var = bnorm[1]
    if hardcoded_normalization[1] == '2' or hardcoded_normalization[1] == '3':
        mean = v1norm[0]
        var = v1norm[1]
    else:
        mean = np.concatenate([v2norm[0].copy() for _ in range(num_instance)])
        var = np.concatenate([v2norm[1].copy() for _ in range(num_instance)])
    return mean, var


# state extractor
class StateExtractor():
    def __init__(self, args, option_selector, full_state, param, mask):
        '''
        hyperparameters for deciding the getter functions actual getting process
        '''
        self.use_pair_gamma = args.use_pair_gamma
        self._pair_featurizer = args.environment_model.create_entity_selector(args.name_pair)
        self._gamma_featurizer = args.dataset_model.gamma
        self._option_featurizer = option_selector # selects the TAIL
        self._delta_featurizer = args.dataset_model.delta
        self._flatten_factored_state = args.environment_model.flatten_factored_state
        self._action_feature_selector = args.action_feature_selector # option.next_option.dataset_model.feature_selector
        self.combine_param_mask = not args.no_combine_param_mask
        self.hardcoded_normalization = args.hardcode_norm
        self.scale = float(self.hardcoded_normalization[2]) if len(self.hardcoded_normalization) > 0 else 1
        self.num_instance = args.num_instance
        self.keep_target = args.keep_target # keeps the target state in relative_param

        self.obs_setting = args.observation_setting
        self.update(full_state)

        # get the amount each component contributes to the input observation
        inter, target, flat, use_param, param_relative, relative, action, option, diff = self.obs_setting
        self.inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0,0,0)).shape
        self.relative_shape = self.get_state(full_state, (0,0,0,0,0,relative,0,0,0), use_pair = self.use_pair_gamma).shape
        self.target_shape = self.get_state(full_state, (0,target,0,0,0,0,0,0,0)).shape
        self.flat_shape = self.get_state(full_state, (0,0,flat,0,0,0,0,0,0)).shape
        obs_inter_shape = self.get_state(full_state, (inter,0,0,0,0,0,0,0,0), use_pair=self.use_pair_gamma).shape
        self.option_shape = self.get_state(full_state, (0,0,0,0,0,0,0,option,0)).shape
        self.pre_param = self.option_shape[0] + obs_inter_shape[0] + self.target_shape[0] + self.flat_shape[0] + self.relative_shape[0]
        print(self.option_shape[0], obs_inter_shape[0], self.target_shape[0], self.flat_shape[0], self.relative_shape[0])
        self.param_shape = self.get_state(full_state, (0,0,0,use_param,0,0,0,0,0), param=param, mask=mask).shape
        self.param_rel_shape = self.get_state(full_state, (0,0,0,0,param_relative,0,0,0,0), param=param, mask=mask).shape
        self.action_shape = self.get_state(full_state, (0,0,0,0,0,0,action,0,0)).shape
        self.diff_shape = self.get_state(full_state, (0,0,0,0,0,0,0,0,diff)).shape
        self.lengths = [ self.option_shape, self.inter_shape, self.relative_shape, self.target_shape, self.flat_shape, self.param_shape, self.param_rel_shape, self.action_shape, self.diff_shape]
        self.component_names = ["option", "inter", "rel", "target", "flat", "param", "param_rel", "action", "diff"]
        self.obs_shape = self.get_obs(full_state, param=param, mask=mask).shape

        print(self.target_shape, self.flat_shape, self.param_shape, self.param_rel_shape, self.relative_shape, self.action_shape, self.option_shape)
        self.first_obj_shape = self.get_state(full_state, (0,0,0,0,0,0,0,1,0)).shape
        self.param_contained = args.param_contained
        if self.param_contained:
            # contains the parameter and first object in the first_obj_shape, and moves param relative information into the object shape
            self.pre_param = 0
            self.post_dim = self.target_shape[0] + self.flat_shape[0] + self.action_shape[0] + self.diff_shape[0]
            self.full_object_shape = self.get_state(full_state, (0,1,0,0,0,0,0,0,0)).shape if self.inter_shape[0] > 0 else self.param_rel_shape
            self.object_shape = ((self.get_state(full_state, (0,1,0,0,0,0,0,0,0)).shape if self.inter_shape[0] > 0 else self.param_rel_shape)[0] // self.num_instance, )
            self.first_obj_shape = (self.get_state(full_state, (0,0,0,0,0,0,0,1,0)).shape[0] + self.param_shape[0], )
            print(self.object_shape, self.first_obj_shape)
        else:
            self.post_dim = self.target_shape[0] + self.flat_shape[0] + self.param_shape[0] + self.param_rel_shape[0] + self.action_shape[0] + self.diff_shape[0]
            self.full_object_shape = self.get_state(full_state, (0,1,0,0,0,0,0,0,0)).shape
            self.object_shape = ((self.get_state(full_state, (0,1,0,0,0,0,0,0,0)).shape)[0] // self.num_instance, )
        

    def split_obs(self, state):
        self.lengths = [ self.option_shape, self.inter_shape, self.relative_shape, self.target_shape, self.flat_shape, self.param_shape, self.param_rel_shape, self.action_shape, self.diff_shape]
        self.component_names = ["option", "inter", "rel", "target", "flat", "param", "param_rel", "action", "diff"]
        at = 0
        components = list()
        for l in lengths:
            if l[0] > 0:
                components.append(state[...,at:at+l[0]])
                at = at + l[0]
        return components

    def get_raw(self, full_state):
        return full_state["raw_state"]

    def get_obs(self, full_state, param, mask=None):
        return self.get_state(full_state, self.obs_setting, param, mask, normalize=True, use_pair=self.use_pair_gamma)

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

    def get_true_reward(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state["Reward"][-1]

    def get_state(self, full_state, setting, param=None, mask=None, normalize = False, use_pair=False): # param is expected 
        # form is an enumerator, 0 is flattened state, 1 is gamma/delta state, 2 is diff using last_factor
        # inp indicates if gamma or delta or gamma+param (if param is not None)
        # full_state is a batch or dict containing raw_state, factored_state
        # raw state may be a batch: [env_num, batch_num, raw state shape]
        # factored_state may also be a batch/dict: {name: [env_num, batch_num, factored_shape]}
        inter, target, flat, use_param, param_relative, relative, action, option, diff = setting
        factored_state = full_state["factored_state"]
        return self._combine_state(full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, param=param, mask=mask, normalize=normalize, use_pair=use_pair)

    def _combine_state(self, full_state, inter, target, flat, use_param, param_relative, relative, action, option, diff, param=None, mask=None, normalize=False, use_pair=False):
        # combines the possible components of state:
        # param relative
        factored_state = array_state(full_state['factored_state'])
        k = list(factored_state.keys())[0]
        shape = factored_state[k].shape[:-1]

        state_comb = list()
        state_comb_dict = dict()
        if option: state_comb_dict['option'] = self._get_option(factored_state, normalize=normalize)
        if inter: state_comb_dict['inter'] = self._get_inter(factored_state, normalize=normalize, use_pair = use_pair)
        if relative: state_comb_dict['relative'] = self._get_relative(factored_state, normalize=normalize, use_pair=use_pair)
        if target: state_comb_dict['target'] = self._get_target(factored_state, normalize=normalize)
        if flat: state_comb_dict['flat'] = self._flatten_factored_state(factored_state, instanced=True)
        if use_param: state_comb_dict['use_param'] = self._broadcast_param(shape, param, mask, normalize=normalize) # only handles param as a concatenatio
        if param_relative: state_comb_dict['param_relative'] = self._relative_param(shape, factored_state, param, mask, normalize=normalize)
        if action: state_comb_dict['action'] = self._select_action_feature(factored_state)
        if diff: state_comb_dict['diff'] = self._get_diff(factored_state)

        # if option: state_comb.append(self._get_option(factored_state, normalize=normalize))
        # if inter: state_comb.append(self._get_inter(factored_state, normalize=normalize, use_pair = use_pair))
        # if relative: state_comb.append(self._get_relative(factored_state, normalize=normalize, use_pair=use_pair))
        # if target: state_comb.append(self._get_target(factored_state, normalize=normalize))
        # if flat: state_comb.append(self._flatten_factored_state(factored_state, instanced=True))
        # if use_param: state_comb.append(self._broadcast_param(shape, param, mask, normalize=normalize)) # only handles param as a concatenation
        # if param_relative: state_comb.append(self._relative_param(shape, factored_state, param, mask, normalize=normalize))
        # if action: state_comb.append(self._select_action_feature(factored_state))
        # if diff: state_comb.append(self._get_diff(factored_state))


        if hasattr(self, 'param_contained') and self.param_contained:
            name_order = ['option', 'use_param', 'inter', 'relative', 'target', 'flat', 'param_relative', 'action', 'diff']
        else:
            name_order = ['option', 'inter', 'relative', 'target', 'flat', 'use_param', 'param_relative', 'action', 'diff']
        for name in name_order:
            if name in state_comb_dict:
                state_comb.append(state_comb_dict[name])
        if len(state_comb) == 0:
            return np.zeros((0,))
        else:
            return np.concatenate(state_comb, axis=len(shape))
        return state_comb[0]

    def reverse_obs_norm(self, states):
        components = self.split_obs(states)
        denorm = list()
        for name, l in zip(self.component_names, self.lengths):
            if l[0] > 0:
                denorm.append(self.apply_normalization(states, name, reverse=True))
        return np.stack(denorm)


    def _select_action_feature(self, factored_state):
        if type(self._action_feature_selector) == list:
            return np.concatenate([a(factored_state) for a in self._action_feature_selector], axis=-1)
        else:
            return self._action_feature_selector(factored_state)

    def apply_normalization(self, state, setting, mask=None, target=None, reverse = False):
        if setting == "target":
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_target(self.hardcoded_normalization, breakout_paddle_norm, breakout_state_norm, breakout_block_norm)
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_target(self.hardcoded_normalization, robopush_gripper_norm, robopush_state_norm, robopush_target_norm)  
            elif self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_target(self.hardcoded_normalization, robostick_gripper_norm, robostick_stick_norm, robostick_block_norm) # TODO:add target norm 
            if reverse:
                return state * var / self.scale + mean
            return (state - mean) / var * self.scale
        if setting == "option":
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_option(self.hardcoded_normalization, breakout_action_norm, breakout_paddle_norm, breakout_state_norm)
                return (state - mean) / var * self.scale
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_option(self.hardcoded_normalization, robopush_action_norm, robopush_gripper_norm, robopush_state_norm)  
            elif self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_option(self.hardcoded_normalization, robostick_action_norm, robostick_gripper_norm, robostick_stick_norm)  
            if reverse:
                return state * var / self.scale + mean
            return (state - mean) / var * self.scale
        if setting == "inter":
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_inter(breakout_action_norm, breakout_paddle_norm, breakout_state_norm, breakout_block_norm, self.hardcoded_normalization, num_instance=self.num_instance)
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_inter(robopush_action_norm, robopush_gripper_norm, robopush_state_norm, robopush_obstacle_norm, self.hardcoded_normalization, num_instance=self.num_instance)  
            elif self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_inter(robostick_action_norm, robostick_gripper_norm, robostick_stick_norm, robostick_block_norm, self.hardcoded_normalization, num_instance=self.num_instance)  
            # print("post_norm", (inter_state - mean) / var * self.scale)
            if reverse:
                return state * var / self.scale + mean
            return (state - mean) / var * self.scale
        if setting == "param":
            # print('param', self.combine_param_mask)
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, breakout_paddle_norm, breakout_state_norm, breakout_block_norm)
            if self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robopush_gripper_norm, robopush_state_norm, robopush_target_norm)
            if self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robostick_gripper_norm, robostick_stick_norm, robostick_block_norm)
            if reverse:
                return state * var / self.scale + mean
            return (state - mean) / var * self.scale
        if setting == "param_rel":
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, breakout_relative_norm, breakout_relative_norm, breakout_relative_norm)
            if self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robopush_relative_norm, robopush_relative_norm, robopush_block_target_relative_norm)
            if self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_param(self.get_mask_param, self.hardcoded_normalization, mask, robostick_gripper_relative_norm, robostick_relative_norm, robostick_block_target_relative_norm)
            if reverse:
                return state * var / self.scale + mean
            return (self.get_mask_param(target, mask) - state - mean) / var * self.scale
        if setting == "rel":
            if self.hardcoded_normalization[0] == 'breakout':
                mean, var = hardcode_norm_relative(self.hardcoded_normalization, breakout_relative_norm, breakout_paddle_ball_norm, breakout_ball_block_norm, num_instance=self.num_instance)
            elif self.hardcoded_normalization[0] == 'robopush':
                mean, var = hardcode_norm_relative(self.hardcoded_normalization, robopush_relative_norm, robopush_gripper_block_norm, robopush_block_target_relative_norm, num_instance=self.num_instance)
            elif self.hardcoded_normalization[0] == 'robostick':
                mean, var = hardcode_norm_relative(self.hardcoded_normalization, robostick_relative_norm, robostick_gripper_stick_norm, robostick_stick_block_norm, num_instance=self.num_instance)
            if reverse:
                return state * var / self.scale + mean
            return (state - mean) / var * self.scale


    def _get_target(self, factored_state, normalize=False):
        target = self._delta_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: target = apply_normalization(state, "target")
        return target

    def _get_option(self, factored_state, normalize=False):
        option = self._option_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: option = self.apply_normalization(option, "option")
        return option

    def _get_inter(self, factored_state, normalize=False, use_pair=False):
        if use_pair:
            inter_state = self._pair_featurizer(factored_state)
        else:
            inter_state = self._gamma_featurizer(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: inter_state = self.apply_normalization(inter_state, 'inter')
        # print("pre_norm", inter_state)
        return inter_state

    def _get_diff(self, factored_state):
        return self._delta_featurizer(factored_state) - self.last_state

    def _broadcast_param(self, shape, param, mask, normalize=False):
        if normalize and len(self.hardcoded_normalization) > 0: param = self.apply_normalization(param, "param", mask)
        if len(shape) == 0:
            return param
        if len(shape) == 1:
            return np.stack([param.copy() for i in range(shape[0])], axis=0)
        if len(shape) == 2:
            return np.stack([np.stack([param.copy() for i in range(shape[1])], axis=0) for j in range(shape[0])], axis=0)

    def _relative_param(self, shape, factored_state, param, mask, normalize=False):
        target = self._delta_featurizer(factored_state)
        param = self._broadcast_param(shape, param, mask, normalize=False)
        if self.num_instance > 1:
            if len(target.shape) == 2:
                target = target.reshape(-1, self.num_instance, mask.shape[0])
            else:
                target = target.reshape(self.num_instance, mask.shape[0])
        if normalize and len(self.hardcoded_normalization) > 0: 
            rel_state = self.apply_normalization( param,"param_rel", mask, target)
        else:
            rel_state = self.get_mask_param(target, mask) - param
        if self.num_instance > 1:
            if self.keep_target:
                rel_state = np.concatenate((target, rel_state), axis = -1)
            if len(rel_state.shape) == 3:
                rel_state = rel_state.reshape(-1, self.num_instance * rel_state.shape[-1])
            else:
                rel_state = rel_state.reshape(self.num_instance * rel_state.shape[-1])
        return rel_state

    def _get_relative(self, factored_state, normalize = False, use_pair=False, add_tail=False):
        if use_pair:
            rel_state = self._pair_featurizer.get_relative(factored_state)
        else:
            rel_state = self._gamma_featurizer.get_relative(factored_state)
        if normalize and len(self.hardcoded_normalization) > 0: rel_state = self.apply_normalization(rel_state, "rel")
        return rel_state

    def update(self, state):
        factored_state = array_state(state['factored_state'])
        self.last_state = self._delta_featurizer(factored_state)

    def get_mask_param(self, param, mask):
        if self.combine_param_mask:
            param = param * mask
        return param

    def assign_param(self, full_state, obs, param, mask):
        '''
        obs may be 1, 2 or 3 dimensional vector. param should be 1d vector
        full state is needed for relative param, as is mask
        since we are assigning to obs, make sure this is normalized
        '''
        if self.param_shape[0] == 0:
            return obs
        shape = obs.shape[:-1]
        param_norm = self._broadcast_param(shape, param, mask, normalize=True)
        if len(obs.shape) == 1:
            obs[self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        elif len(obs.shape) == 2:
            obs[:, self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        elif len(obs.shape) == 3:
            obs[:,:,self.pre_param:self.pre_param + self.param_shape[0]] = param_norm
        if self.param_rel_shape[0] > 0:
            target = self.get_target(full_state)
            if self.num_instance > 1:
                if len(target.shape) == 2:
                    target = target.reshape(-1, self.num_instance, mask.shape[0])
                else:
                    target = target.reshape(self.num_instance, mask.shape[0])
            target = self.get_mask_param(target, mask)
            target = self.apply_normalization(target, 'param', mask)
            diff = param_norm - target
            if self.num_instance > 1:
                if self.keep_target:
                    diff = np.concatenate((target, diff), axis = -1)
                if len(diff.shape) == 3:
                    diff = diff.reshape(-1, self.num_instance * diff.shape[-1])
                else:
                    diff = diff.reshape(self.num_instance * diff.shape[-1])

            pre_rel = self.pre_param + self.param_shape[0]
            # if len(obs.shape) == 1:
            obs[...,pre_rel:pre_rel + self.param_rel_shape[0]] = diff
            # elif len(obs.shape) == 2:
            #     obs[:, pre_rel:pre_rel + self.param_rel_shape[0]] = diff
            # elif len(obs.shape) == 3:
            #     obs[:,:,pre_rel:pre_rel + self.param_rel_shape[0]] = diff

        return obs
