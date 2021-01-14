import numpy as np
import torch

class Sampler():
    def __init__(self, **kwargs):
        self.dataset_model = kwargs["dataset_model"]
        self.delta = self.dataset_model.delta
        self.sample_continuous = True # kwargs["sample_continuous"] # hardcoded for now
        
        # specifit to CFselectors
        self.cfselectors = self.dataset_model.cfselectors
        self.lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
        self.upper_cfs = np.array([i for i in [cfs.feature_range[1] for cfs in self.cfselectors]])
        self.len_cfs = np.array([j-i for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])

    def sample(self, states):
        # takes in states of size num_sample x state_size, and return samples 
        return

    def weighted_samples(self, states, weights, centered=False):
        # gives back the samples based on normalized weights
        edited_features = self.lower_cfs + self.len_cfs * weights
        new_states = states.clone()
        for f, w, cfs in zip(edited_features, self.len_cfs * weights, self.cfselectors):
            if centered:
                cfs.assign_feature(new_states, w, edit=True, clipped=True)
            else:
                cfs.assign_feature(new_states, f)
        selection_binary = self.dataset_model.selection_binary
        if len(new_states.shape) > 1: # if a stack, duplicate mask for all
            return self.delta(new_states), pytorch_model.wrap(torch.stack([selection_binary.clone() for _ in range(new_states.size(0))], dim=0), cuda=self.iscuda)
        return self.delta(new_states), selection_binary.clone()


class LinearUniformSampling(Sampler):
    def sample(self, states):
        if self.dataset_model.sample_continuous:
            cfselectors = self.dataset_model.cfselectors
            weights = np.random.random((len(cfselectors,))) # random weight vector
            return self.weighted_samples(states, weights)
        else: # sample discrete with weights
            return

class GaussianCenteredSampling(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distance = .05 # normalized
        self.schedule = kwargs["sample_schedule"]
        self.schedule_counter = 0

    def sample(self, states):
        if self.schedule > 0 and self.schedule_counter % self.schedule == 0:
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

    def sample(self, states):
        if self.schedule > 0 and self.schedule_counter % self.schedule == 0:
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

class ReachedSampling


samplers = {"uni": LinearUniformSampling, "gau": GaussianCenteredSampling}