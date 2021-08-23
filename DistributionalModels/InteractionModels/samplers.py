import numpy as np
import torch
import copy
from Networks.network import pytorch_model
from Options.state_extractor import array_state

class Sampler():
    def __init__(self, **kwargs):
        self.dataset_model = kwargs["dataset_model"]
        self.delta = self.dataset_model.delta
        self.sample_continuous = True # kwargs["sample_continuous"] # hardcoded for now
        self.combine_param_mask = not kwargs["no_combine_param_mask"] # whether to multiply the returned param with the mask
        self.rate = 0

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

    def update(self, param, mask, buffer=None):
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
                cfs.assign_feature(new_states, w, factored=type(new_states) == dict, edit=True, clipped=True) # TODO: factored might be possible to be batch
        else:
            for f, cfs in zip(edited_features, self.cfselectors):
                cfs.assign_feature(new_states, f, factored=type(new_states) == dict)
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
        print(sample_able_all, sample_able_inst, [exp.getMidpoint() for exp in environment_model.environment.exposed_blocks.values()])
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
        # print(self.dataset_model.sample_able.vals)
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
            distance = self.distance + (.4 - self.distance) * np.exp(-self.schedule/self.schedule_counter)
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            weights = np.random.normal(loc=0, scale=self.distance, size=(len(cfselectors,))) # random weight vector
            return self.weighted_samples(states, weights, centered=True)
        else: # sample discrete with weights
            return

class LinearUniformCenteredSampling(Sampler):
    def __init__(self, **kwargs):
        self.distance = .03 # normalized
        self.schedule_counter = 0
        self.schedule = kwargs["sample_schedule"]
        super().__init__(**kwargs)

    def update(self, param, mask, buffer=None):
        super().update(param, mask, buffer=None)
        self.schedule_counter += 1

    def get_targets(self, states):
        distance = .2
        if self.schedule > 0: # the distance changes, otherwise it is a set constant .15 of the maximum TODO: add hyperparam
            distance = self.distance + (distance - self.distance) * np.exp(-self.schedule/self.schedule_counter)
        cfselectors = self.dataset_model.cfselectors
        weights = (np.random.random((len(cfselectors,))) - .5) * 2 * distance # random weight vector bounded between -distance, distance
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

# class ReachedSampling

mask_samplers = {"rans": RandomSubsetSampling, "pris": PrioritizedSubsetSampling} # must be 4 characters
samplers = {"uni": LinearUniformSampling, "cuni": LinearUniformCenteredSampling, "gau": GaussianCenteredSampling, "hst": HistorySampling, 'inst': InstanceSampling}