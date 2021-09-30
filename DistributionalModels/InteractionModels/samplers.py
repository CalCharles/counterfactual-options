import numpy as np
import torch
import copy
import collections
import torch.nn as nn
import torch.optim as optim
from Networks.network import pytorch_model
from Networks.DistributionalNetworks.forward_network import forward_nets
from Options.state_extractor import array_state
from tianshou.data import Batch
from Rollouts.param_buffer import SamplerBuffer

def check_dict(states):
    return type(states) == dict or type(states) == Batch or type(states) == collections.OrderedDict

class Sampler():
    def __init__(self, **kwargs):
        self.dataset_model = kwargs["dataset_model"]
        self.delta = self.dataset_model.delta
        self.sample_continuous = True # kwargs["sample_continuous"] # hardcoded for now
        self.combine_param_mask = not kwargs["no_combine_param_mask"] # whether to multiply the returned param with the mask
        self.rate = 0
        self.current_distance = 0

        # specifit to CFselectors
        self.cfselectors = self.dataset_model.cfselectors
        self.lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
        self.upper_cfs = np.array([i for i in [cfs.feature_range[1] for cfs in self.cfselectors]])
        self.len_cfs = np.array([j-i for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])

        self.param, self.mask = self.sample(kwargs["init_state"])
        self.iscuda = False

    def cuda(self, device=None):
        self.iscuda= True 

    def cpu(self, device=None):
        self.iscuda= False 

    def get_targets(self, states):
        # takes in states of size num_sample x state_size, and return samples (num_sample x state_size)
        return

    def sample_subset(self, selection_binary):
        return selection_binary

    def get_mask(self, param):
        return self.mask

    def update(self, param, mask, data=None):
        self.param, self.mask = param, mask

    def update_rates(self, masks, results):
        # for prioritizing different masks
        pass

    def get_binary(self, states):
        selection_binary = self.sample_subset(self.dataset_model.selection_binary)
        if len(self.delta(states).shape) > 1: # if a stack, duplicate mask for all
            return pytorch_model.unwrap(torch.stack([selection_binary.clone() for _ in range(new_states.size(0))], dim=0))
        return pytorch_model.unwrap(selection_binary.clone())

    def weighted_samples(self, states, weights, centered=False, edited_features=None):
        # gives back the samples based on normalized weights
        if edited_features is None:
            edited_features = self.lower_cfs + self.len_cfs * weights
        new_states = states.copy()
        if centered:
            for f, w, cfs in zip(edited_features, self.len_cfs * weights, self.cfselectors):
                cfs.assign_feature(new_states, w, factored=check_dict(states), edit=True, clipped=True) # TODO: factored might be possible to be batch
        else:
            for f, cfs in zip(edited_features, self.cfselectors):
                cfs.assign_feature(new_states, f, factored=check_dict(states))
        return self.delta(new_states)

    def sample(self, states):
        '''
        expects states to be a full_state ( a tianshou.batch or dict with factored_state, raw_state inside )
        factored state may have a single value or multiple
        '''
        states = states['factored_state']
        states = array_state(states)
        mask = self.get_binary(states)
        return self.get_mask_param(self.get_targets(states), mask), mask # TODO: not masking out the target not always precise
        # return self.get_targets(states), mask 

    def get_param(self, full_state, terminate):
        # samples new param and mask if terminate. If there are more reasons to change param, that logic can be added
        if terminate:
            self.param, self.mask = self.sample(full_state)
            self.param, self.mask = self.param.squeeze(), self.mask.squeeze() # this could be a problem with 1 dim params and masks
        return self.param, self.mask, terminate

    def get_mask_param(self, param, mask):
        if self.combine_param_mask:
            return param * mask
        return param

    def convert_param(self, param): # TODO: only handles single params at a time
        new_param = self.mask.copy().squeeze()
        param = param.squeeze()
        new_param[new_param == 1] = param
        param = new_param
        return param

    def reset(self, init_state):
        # resets internal state if needed
        pass

class TargetSampler(Sampler):
    def __init__(self, **kwargs):
        # a sampler that just takes the "Target" object as the "sample"
        print(kwargs["target_object"])
        self.delta = kwargs["environment_model"].create_entity_selector([kwargs["target_object"]])
        self.combine_param_mask = True
        self.current_distance = 0
        self.param, self.mask = self.sample(kwargs["init_state"])
        self.iscuda = False

    def sample(self, states):
        states = states["factored_state"]
        target = self.delta(states)
        return target, np.zeros(target.shape)

class RawSampler(Sampler):
    # never actually samples
    def get_targets(self, states):
        return self.dataset_model.sample(states)

class LinearUniformSampling(Sampler):
    def get_targets(self, states):
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            weights = np.random.random((len(cfselectors,))) # random weight vector
            return self.weighted_samples(states, weights)
        else: # sample discrete with weights
            num_sample = 1
            if len(self.delta(states).shape) > 1:
                num_sample = states.shape[0]
            #     masks = [pytorch_model.unwrap(self.dataset_model.selection_binary.clone()) for i in range(num_sample)]
            # else:
            #     masks = pytorch_model.unwrap(self.dataset_model.selection_binary.clone())
            if num_sample > 1:
                value = np.array([self.dataset_model.sample_able.vals[np.random.randint(len(self.dataset_model.sample_able.vals))].copy() for i in range(num_sample)])
            else:
                value = self.dataset_model.sample_able.vals[np.random.randint(len(self.dataset_model.sample_able.vals))].copy()
            return copy.deepcopy(pytorch_model.unwrap(value))

def find_inst_feature(state_values, sample_able, nosample, sample_exposed, environment_model): # state values has shape [num_instances, state size]
    found = False
    sample_able_inst = list()
    sample_able_all = list()
    for idx, s in enumerate(state_values):
        for h in sample_able:
            if np.linalg.norm(s * m - h) > nosample:
                if sample_exposed:
                    for exp in environment_model.environment.exposed_blocks.values():
                        if np.linalg.norm(exp.getMidpoint() - s[:2]) < nosample:
                            sample_able_inst.append((s, h))
                            break
                else:
                    sample_able_inst.append((s, h))
                sample_able_all.append((s,h))
    # s, h = sample_able_inst[np.random.randint(len(sample_able_inst))]
    if len(sample_able_inst) == 0:
        return sample_able_all[np.random.randint(len(sample_able_all))]
    return sample_able_inst[np.random.randint(len(sample_able_inst))]

class InstancePredictiveSampler(Sampler):
    def __init__(self, **kwargs):
        self.sampler_network = PairNetwork(**kwargs)

class InstanceSampling(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nosample_epsilon = .0001
        self.environment_model = kwargs["environment_model"]
        self.sample_exposed = True

    def get_targets(self, states):
        dstates = self.dataset_model.delta(states)
        values = self.dataset_model.split_instances(dstates)
        m = pytorch_model.unwrap(self.dataset_model.selection_binary) # mask is hardcoded from selection binary, not sampled
        inv_m = m.copy()
        inv_m[m == 0] = -1
        inv_m[m == 1] = 0
        inv_m *= -1
        if len(values.shape) == 3:
            params = list()
            # idxes = list()
            for v in values:
                s, val = find_inst_feature(v, self.dataset_model.sample_able.vals, self.nosample_epsilon, self.sample_exposed, self.environment_model)
                mval = val * m + inv_m * s
                param.append(mval)
                # idxes.append(idx)
            return np.stack(param, axis=0)
        elif len(values.shape) == 2:
            s, val = find_inst_feature(values, self.dataset_model.sample_able.vals, self.nosample_epsilon, self.sample_exposed, self.environment_model)
            mval = val * m + inv_m * s
            return mval

    def sample(self, states):
        states = states['factored_state']
        states = array_state(states)
        mask = self.get_binary(states)
        val = self.get_targets(states)
        return val, mask # does not mask out target, justified by behavior

class HistoryInstanceSampling(Sampler):
    def sample(self, states):
        states = states["factored_state"]
        for k in states.keys():
            states[k] = np.array(states[k])
        mask = self.get_binary(states)
        inv_m = mask.copy()
        inv_m[mask == 0] = -1
        inv_m[mask == 1] = 0
        inv_m *= -1
        val_idx = np.random.randint(len(self.dataset_model.sample_able.vals))
        m_value = self.dataset_model.sample_able.vals[val_idx]
        val = self.dataset_model.delta(states) * inv_m + m_value * mask# TODO: does not handle multiple states
        return val, mask

class RandomSubsetSampling(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampler = samplers[kwargs['sampler-type'][4:]](**kwargs)
        self.rate = kwargs['rate']
        self.min_size = kwargs["min_mask_size"]
        if self.min_size > 0: # needs to sample from masks
            self.masks = binary_string(self.dataset_model.selection_binary, self.min_size)
            if type(self.rate) == float or len(self.rate) != len(self.masks):
                self.rate = np.ones(len(self.masks)) / float(len(self.masks))

    def sample_subset(self, selection_binary):
        '''
        Samples only one subset of the selection binary, completely random
        '''
        if self.min_size > 0:
            new_mask = np.random.choice(self.masks, p =self.rate)
        elif type(self.rate) == float:
            new_mask = torch.tensor([np.random.binomial(1, p=self.rate) if i else 0 for i in selection_binary])
        else:
            new_mask = torch.tensor([np.random.binomial(1, p=r) if i else 0 for r, i in zip(self.rate, selection_binary)])
        return new_mask

def binary_string(mask, min_size=0): # constructs all binary strings for a selection_mask
    # takes action +- 1, 0 at each dimension, for every combination
    # creates combinatorially many combinations of this
    # action space is assumed to be the tuple shape of the space
    # TODO: assume action space of the form (n,)
    mask_subs = list()
    def append_mask(i, bs):
        if i == len(mask): mask_subs.append(bs)
        elif mask[i] == 0:
            append_str(i+1, bs + [0])
        else:
            bs0 = copy.copy(bs)
            bs0.append(0)
            append_str(i+1,bs0)
            bs.append(1)
            append_str(i+1, bs)
    append_mask(0, list())
    sized_subs = list()
    for m in mask_subs:
        if np.sum(m) >= min_size:
            sized_subs.append(m)
    return {i: sized_subs[i] for i in range(len(sized_subs))} # gives the ordering arrived at by 0, 1 ordering interleaved

class PrioritizedSubsetSampling(RandomSubsetSampling):
    def update_rates(self, masks, results):
        # for prioritizing different masks
        pass


class HistorySampling(Sampler):
    def get_targets(self, states):
        # if len(targets.shape) > 1: 
        #     value = np.random.randint(len(self.dataset_model.sample_able.vals), size=states.shape[0])
        #     value = np.array(self.dataset_model.sample_able.vals)[value]
        # else:
        value = np.random.choice(self.dataset_model.sample_able.vals)
        return self.weighted_samples(states, None, edited_features=value) # value.clone(), self.dataset_model.selection_binary.clone()

class GaussianCenteredSampling(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance = .03 # normalized
        self.schedule = kwargs["sample_schedule"]
        self.schedule_counter = 0

    def get_targets(self, states):
        distance = .4
        if self.schedule > 0: # the distance changes, otherwise it is a set constant .15 of the maximum TODO: add hyperparam
            distance = self.distance + (.4 - self.distance) * np.exp(-self.schedule/(self.schedule_counter + 1))
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            weights = np.random.normal(loc=0, scale=self.distance, size=(len(cfselectors,))) # random weight vector
            return self.weighted_samples(states, weights, centered=True)
        else: # sample discrete with weights
            return

class LinearUniformCenteredSampling(Sampler):
    def __init__(self, **kwargs):
        self.distance = kwargs["sample_distance"] # normalized
        self.schedule_counter = 0
        self.schedule = kwargs["sample_schedule"]
        self.current_distance = .1 if self.schedule > 0 else self.distance
        super().__init__(**kwargs)

    def update(self, param, mask, data=None):
        super().update(param, mask, data=None)
        self.schedule_counter += 1

    def get_targets(self, states):
        distance = self.current_distance
        if self.schedule > 0: # the distance changes, otherwise it is a set constant .15 of the maximum TODO: add hyperparam
            self.current_distance = self.distance - (self.distance - distance) * np.exp(-(self.schedule_counter + 1)/self.schedule)
        cfselectors = self.dataset_model.cfselectors

        weights = (np.random.random((len(cfselectors,))) - .5) * 2 * self.current_distance # random weight vector bounded between -distance, distance
        return self.weighted_samples(states, weights, centered=True)

class LinearUniformCenteredUnclipSampling(Sampler):
    def __init__(self, **kwargs):
        self.distance = kwargs["sample_distance"] # normalized
        self.schedule_counter = 0
        self.schedule = kwargs["sample_schedule"]
        super().__init__(**kwargs)
        self.current_distance = .1 if self.schedule > 0 else kwargs["sample_distance"]

    def update(self, param, mask, data=None):
        super().update(param, mask, data=None)
        self.schedule_counter += 1

    def get_targets(self, states):
        distance = self.current_distance
        if self.schedule > 0: # the distance changes, otherwise it is a set constant .15 of the maximum TODO: add hyperparam
            self.current_distance = self.distance - (self.distance - distance) * np.exp(-(self.schedule_counter + 1)/self.schedule)
        cfselectors = self.dataset_model.cfselectors

        weights = [] 
        for cfslen, cfs in zip(self.len_cfs, cfselectors):
            lower, upper = cfs.relative_range(states, factored=check_dict(states))
            lw, uw = lower / cfslen, upper / cfslen
            weights.append(-lw + np.random.random() * (uw+lw))
        weights = np.array(weights) * self.current_distance
        return self.weighted_samples(states, weights, centered=True)

class GaussianOffCenteredSampling(Sampler):
    def __init__(self, **kwargs):
        # samples to the sides, which does favor the edges of the map
        super().__init__(**kwargs)
        self.distance = .05 # normalized
        self.variance = .1
        self.schedule = kwargs["sample_schedule"]
        self.schedule_counter = 0

    def get_targets(self, states):
        if self.sctargets > 0 and self.schedule_counter % self.schedule == 0:
            self.distance = min(self.distance * 2, .4)
        self.schedule_counter += 1
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            posweights = np.random.normal(loc=self.distance, scale=self.variance, size=(len(cfselectors,))) # random weight vector
            negweights = np.random.normal(loc=-self.distance, scale=self.variance, size=(len(cfselectors,))) # random weight vector
            weights = posweights if np.random.random() > .5 else negweights
            return self.weighted_samples(states, weights, centered=True)
        else: # sample discrete with weights
            return

class PathSampler(Sampler):
    # a sampler that returns the path to the goal as samples
    def __init__(self, **kwargs):
        self.path_fn = kwargs["path_fn"]
        self.table_offset = kwargs["environment_model"].environment.env.table_offset
        self.spawn_size = kwargs["environment_model"].environment.env.SPAWN_AREA_SIZE * 2
        self.obstacle_half_sidelength = kwargs["environment_model"].environment.env.OBSTACLE_HALF_SIDELENGTH
        self.grid_resolution = kwargs["environment_model"].environment.env.OBSTACLE_GRID_RESOLUTION
        self.current_samples = list()
        self.reset(kwargs["init_state"])
        self.repeat_param = self.current_samples[-1]
        super().__init__(**kwargs)

    def _get_obstacles(self, factored_state):
        obs = list()
        for k in factored_state.keys():
            if k.find("Obstacle") != -1:
                obs.append(factored_state[k])
        return np.stack(obs, axis=0).squeeze()

    def reset(self, init_state):
        obstacles = init_state["factored_state"]
        block = init_state["factored_state"]["Block"]
        target = init_state["factored_state"]["Target"]
        self.current_samples = self.path_fn(self._get_obstacles(obstacles), self.table_offset, self.spawn_size, self.obstacle_half_sidelength, self.grid_resolution, block, target)[1]
        print(self.current_samples)
        
    def sample(self, states):
        param = self.current_samples.pop(0) if len(self.current_samples) > 0 else self.repeat_param
        if len(self.current_samples) == 0:
            self.repeat_param = param
        # print(states, param)
        if type(states) == dict:
            state_len = len(states["factored_state"]["Action"].shape)
        else:
            state_len = len(states.shape)
        # pad_vals = [(0, 0)] * len(states.shape) - 1
        # pad_vals += [(0,1)]
        # param = np.pad(param, pad_width=pad_vals, mode='constant', constant_values=0)
        mask = np.array([1.,1.,0.])
        for i in range(state_len - 1):
            mask = np.expand_dims(mask, 0)
            param = np.expand_dims(param, 0)
        return param * mask, mask

PREDICT_BINARY = 0
PREDICT_VALUE = 1
class PredictiveSampling(Sampler):
    def __init__(self, **kwargs):
        self.action_prediction = kwargs["action_prediction"] if "action_prediction" in kwargs else 0
        self.value = 10
        kwargs["value"] = 10
        self.noise_actions = kwargs["noise_actions"] if "noise_actions" in kwargs else 0

        self.selector = kwargs["entity_selector"]
        self.target_selector = kwargs["target_selector"]
        self.num_actions = kwargs["num_actions"]
        self.object_dim = kwargs["object_dim"]
        self.iscuda=False
        self.input_size = self.get_input(kwargs["init_state"], 0).shape[0]        
        kwargs["num_inputs"] = self.get_input(kwargs["init_state"], 0).shape[0]
        kwargs["num_outputs"] = self.get_output(kwargs["init_state"], 0).shape[0]
        kwargs["post_dim"] = 0
        kwargs["obs"] = self.get_input(kwargs["init_state"], 0)
        print(kwargs["num_inputs"], kwargs["num_outputs"])
        self.network = forward_nets[kwargs["sampler_type"]](**kwargs)
        self.train_rate = kwargs["sampler_train_rate"]
        self.update_counter = 0
        self.buffer = SamplerBuffer(kwargs["buffer_len"], stack_num=1)
        self.test_buffer = SamplerBuffer(kwargs["buffer_len"], stack_num=1)
        self.grad_steps = kwargs["sampler_grad_epoch"]
        if self.action_prediction == PREDICT_BINARY:
            self.inter_loss = nn.BCELoss()
        elif self.action_prediction == PREDICT_VALUE:
            self.inter_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), 1e-4, eps=1e-5, betas=(.9,.999), weight_decay=0)
        self.batch_size = 128
        self.last_done = True
        self.added_since_last = False
        self.mask = kwargs["mask"]
        self.param, self.mask = self.sample(kwargs["init_state"])
        self.train_test = .9

    def cuda(self):
        self.network.cuda()
        self.iscuda=True

    def cpu(self):
        self.network.cpu()
        self.iscuda=False

    def hot_action(self, act):
        v = np.zeros(self.num_actions)
        v[int(act)] = 1
        return v

    def get_binary(self, states):
        selection_binary = self.sample_subset(self.mask)
        if len(self.target_selector(states).shape) > 1: # if a stack, duplicate mask for all
            return pytorch_model.unwrap(torch.stack([selection_binary.clone() for _ in range(new_states.size(0))], dim=0))
        return pytorch_model.unwrap(selection_binary.clone())

    def run_optimizer(self, optimizer, model, loss):
        optimizer.zero_grad()
        (loss.mean()).backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        optimizer.step()

    def get_input(self, full_state, act=0):
        act = self.hot_action(act)
        obs = self.selector(full_state["factored_state"])
        if self.action_prediction == PREDICT_BINARY:
            obs = np.concatenate([act, obs])
        return obs

    def get_output(self, full_state, act=0):
        if self.action_prediction == PREDICT_BINARY: 
            target = self.target_selector(full_state["factored_state"])
            final_instances = self.split_instances(target)
            out_val = self.split_binary(final_instances)
        elif self.action_prediction == PREDICT_VALUE:
            out_val = self.hot_action(act) * self.value
        return out_val

    def aggregate(self, data):
        # given a data point, add
        if self.last_done:
            self.data = copy.deepcopy(data)
            self.last_done = False
        if data.act != -1 and not self.added_since_last:
            act = self.hot_action(data.act)
            if self.action_prediction == PREDICT_BINARY:
                obs = np.concatenate([act, self.data.obs])
            elif self.action_prediction == PREDICT_VALUE:
                obs = self.data.obs
            act = data.act
            if self.noise_actions > 0:
                act = np.random.randint(self.num_actions) if np.random.rand() < self.noise_actions else data.act
            self.data.update(obs=obs, act=act, true_act=data.act)
            self.added_since_last = True
        if data.done:
            final_instances = self.split_instances(data.target)
            instance_binary = self.split_binary(final_instances)
            self.data.update(done=data.done, target=data.target, instance_binary=instance_binary)
            print(self.data)
            if len(self.data.obs) == self.input_size: 
                if np.random.rand() < self.train_test: self.buffer.add(self.data)
                else: 
                    self.data.act = self.data.true_act
                    self.test_buffer.add(self.data)
            self.last_done = True
            self.added_since_last = False

    def split_binary(self, instances):
        return 1-instances[...,-1]

    def split_instances(self, state, obj_dim=-1):
        # split up a state or batch of states into instances
        if obj_dim < 0:
            obj_dim = self.object_dim
        nobj = state.shape[-1] // obj_dim
        if len(state.shape) == 1:
            state = state.reshape(nobj, obj_dim)
        elif len(state.shape) == 2:
            state = state.reshape(-1, nobj, obj_dim)
        return state

    def assign_data(self, state, act):
        factored_state = state['factored_state']
        new_data = Batch()
        new_data.update(obs=self.selector(factored_state), target=self.target_selector(factored_state), act = act, done=factored_state["Done"], rew=0,)
        return new_data

    def combine_act(self, obs, act):
        act = np.stack([self.hot_action(a) for a in act], axis=0)
        np.concatenate([act, obs])

    def predict(self, batch):
        obs = pytorch_model.wrap(batch.obs, cuda=self.iscuda)
        return self.network(obs) # outputs [batch_size, num_instances]

    def compute_loss(self, batch):
        obs = pytorch_model.wrap(batch.obs, cuda=self.iscuda)
        predicted = self.network(obs) # outputs [batch_size, num_instances]
        if self.action_prediction == PREDICT_BINARY:
            target = pytorch_model.wrap(batch.instance_binary, cuda=self.iscuda)
            target.needs_grad = False
            loss = self.inter_loss(predicted, target) # TODO: use target if predicting distancenot instance binary
        elif self.action_prediction == PREDICT_VALUE:
            instance_binary =batch.instance_binary
            target = pytorch_model.wrap(instance_binary * (self.value + 1) - 1, cuda=self.iscuda)
            # print(target, predicted, predicted[np.stack([np.arange(len(predicted)), batch.act])])
            # print(target.shape, predicted[np.stack([np.arange(len(predicted)), batch.act])].shape)
            loss = self.inter_loss(target, predicted[np.stack([np.arange(len(predicted)), batch.act])].unsqueeze(1))
        return loss, predicted, target, pytorch_model.wrap(batch.act, cuda=self.iscuda).unsqueeze(-1), pytorch_model.wrap(batch.true_act, cuda=self.iscuda).unsqueeze(-1)

    def assess_test(self):
        batch, indice = self.test_buffer.sample(0)
        return self.compute_loss(batch)


    def update(self, param, mask, data=None):
        # updates param and mask accordingly
        super().update(param, mask, data=data)
        # adds in the data if necessary
        if data is not None: self.aggregate(data)
        # updates the sampling model if necessary
        total_loss = 0 
        if self.update_counter % self.train_rate == 0:
            for i in range(self.grad_steps):
                batch, indice = self.buffer.sample(self.batch_size)
                # batch.obs = self.combine_act(batch.obs, batch.act)
                loss, predicted, target, act, true_act = self.compute_loss(batch)
                self.run_optimizer(self.optimizer, self.network, loss)
                total_loss += loss
            total_loss = total_loss.mean()
        self.update_counter += 1
        return total_loss, predicted, target, act, true_act

    def sample(self, states):
        obs = pytorch_model.wrap(self.get_input(states), cuda=self.iscuda)
        print(obs)
        predicted_binaries = pytorch_model.unwrap(self.network(obs))
        return np.array(1), np.array(1)

# class ReachedSampling

mask_samplers = {"rans": RandomSubsetSampling, "pris": PrioritizedSubsetSampling} # must be 4 characters
samplers = {"uni": LinearUniformSampling, "cuni": LinearUniformCenteredSampling, 'cuuni': LinearUniformCenteredUnclipSampling,
            "gau": GaussianCenteredSampling, "hst": HistorySampling, 'inst': InstanceSampling,
            "hstinst": HistoryInstanceSampling, 'tar': TargetSampler, 'path': PathSampler}