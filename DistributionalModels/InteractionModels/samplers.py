import numpy as np
import torch
import copy
from Networks.network import pytorch_model

class Sampler():
    def __init__(self, **kwargs):
        self.dataset_model = kwargs["dataset_model"]
        self.delta = self.dataset_model.delta
        self.sample_continuous = True # kwargs["sample_continuous"] # hardcoded for now
        self.rate = 0
        
        # specifit to CFselectors
        self.cfselectors = self.dataset_model.cfselectors
        self.lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
        self.upper_cfs = np.array([i for i in [cfs.feature_range[1] for cfs in self.cfselectors]])
        self.len_cfs = np.array([j-i for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])

    def get_targets(self, states):
        # takes in states of size num_sample x state_size, and return samples (num_sample x state_size)
        return

    def sample_subset(self, selection_binary):
        return selection_binary

    def update_rates(self, masks, results):
        # for prioritizing different masks
        pass

    def get_binary(self, states):
        selection_binary = self.sample_subset(self.dataset_model.selection_binary)
        if len(states.shape) > 1: # if a stack, duplicate mask for all
            return pytorch_model.unwrap(torch.stack([selection_binary.clone() for _ in range(new_states.size(0))], dim=0))
        return pytorch_model.unwrap(selection_binary.clone())

    def weighted_samples(self, states, weights, centered=False, edited_features=None):
        # gives back the samples based on normalized weights
        if edited_features is None:
            edited_features = self.lower_cfs + self.len_cfs * weights
        new_states = states.copy()
        if centered:
            for f, w, cfs in zip(edited_features, self.len_cfs * weights, self.cfselectors):
                cfs.assign_feature(new_states, w, edit=True, clipped=True)
        else:
            for f, cfs in zip(edited_features, self.cfselectors):
                cfs.assign_feature(new_states, f)
        return self.delta(new_states)

    def sample(self, states):
        mask = self.get_binary(states)
        return self.get_targets(states) * mask, mask 

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
            if len(states.shape) > 1:
                num_sample = states.shape[0]
            #     masks = [pytorch_model.unwrap(self.dataset_model.selection_binary.clone()) for i in range(num_sample)]
            # else:
            #     masks = pytorch_model.unwrap(self.dataset_model.selection_binary.clone())
            # print(self.dataset_model.sample_able.vals)
            if num_sample > 1:
                value = np.array([self.dataset_model.sample_able.vals[np.random.randint(len(self.dataset_model.sample_able.vals))].copy() for i in range(num_sample)])
            else:
                value = self.dataset_model.sample_able.vals[np.random.randint(len(self.dataset_model.sample_able.vals))].copy()
            # print(copy.deepcopy(pytorch_model.unwrap(value)))
            return copy.deepcopy(pytorch_model.unwrap(value))

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
        self.distance = .05 # normalized
        self.schedule = kwargs["sample_schedule"]
        self.schedule_counter = 0

    def get_targets(self, states):
        if self.sctargets > 0 and self.schedule_counter % self.schedule == 0:
            self.distance = min(self.distance * 2, .4)
        self.schedule_counter += 1
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            weights = np.random.normal(loc=0, scale=self.distance, size=(len(cfselectors,))) # random weight vector
            print(weights)
            return self.weighted_samples(states, weights, centered=True)
        else: # sample discrete with weights
            return

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
samplers = {"uni": LinearUniformSampling, "gau": GaussianCenteredSampling, "hst": HistorySampling}