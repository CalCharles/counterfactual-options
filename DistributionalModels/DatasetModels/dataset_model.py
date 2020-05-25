import numpy as np
import os, cv2, time
from Counterfactual.counterfactual_dataset import counterfactual_mask

class DatasetModel(DistributionalModel):
    def __init__(self, **kwargs):
        '''
        conditionally models the distribution of some output variable from some input variable. 
        @attr has_relative indicates if the model also models the input from some output
        @attr outcome is the outcome distribution 
        '''
        super().__init__(self, **kwargs)
        # TODO: does not record which option produced which outcome
        self.observed_outcomes = [] # keeps a set of all the observed outcomes as a list, which is inefficient TODO: a better distribution method
        self.outcome_counts = [] # a running counter of how many times an outcome was seen
        self.observed_differences = [] # keeps a set of all observed state differences as a list
        self.difference_counts = []
        self.total_count = 0
        self.EPSILON = 1e-5

    def check_observed(self, outcome, observed):
        for i, o in enumerate(observed):
            if (o - outcome).norm() < self.EPSILON:
                return i
        return -1

    def add_observed(self, outcome, observed, counter):
        i = self.check_observed(outcome, observed)
        if i < 0:
            observed.append(outcome)
            counter.append(1)
        else:
            counter[i] += 1

    def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
        '''
        Records the different counterfactual outputs. 
        '''
        states = outcome_rollouts.get_values("state")
        state_len = state.size(1)
        state_diffs = outcome_rollouts.get_values("state_diff")
        fullstate_masks, outcome_probs = counterfactual_mask(outcome_rollouts)
        state_masks = fullstate_masks[:, :state_len]
        state_diff_masks = fullstate_masks[:, state_len:]
        for i in [k*self.num_options for k in range(outcome_rollouts.length // self.num_options)]: # TODO: assert evenly divisible
            for j in range(self.num_options):
                self.add_observed(states[i+j] * state_masks[i+j], self.observed_outcomes, self.outcome_counts)
                self.add_observed(state_diffs[i+j] * state_diff_masks[i+j], self.observed_differences, self.difference_counts)
                self.total_count += 1


    def sample(self, rollouts, diff=True):
        '''
        takes a random sample of the outcomes or diffs and then returns it. This has issues since you don't know which one you are getting,
        and the outcomes and difference have been separated, so you cannot just sample from each. 
        '''
        if diff:
            possible_indexes = list(range(len(self.difference_counts)))
            indexes = np.random.choice(possible_indexes, rollouts.length, replace=True, p=np.array(self.difference_counts) / self.total_count)
            diffs = []
            for i in indexes:
                diffs.append(self.observed_differences[i].copy())
            return torch.stack(diffs, dim=0)
        else:
            possible_indexes = list(range(len(self.outcome_counts)))
            indexes = np.random.choice(possible_indexes, rollouts.length, replace=True, p=np.array(self.outcome_counts) / self.total_count)
            outs = []
            for i in indexes:
                outs.append(self.observed_outcomes[i].copy())
            return torch.stack(outs, dim=0)

class FactoredDatasetModel(DistributionalModel):

    def __init__(self, **kwargs):
        '''
        conditionally models the distribution of some output variable from some input variable. 
        @attr has_relative indicates if the model also models the input from some output
        @attr outcome is the outcome distribution 
        '''
        super().__init__(self, **kwargs)
        # TODO: does not record which option produced which outcome
        self.names = kwargs["environment_model"].object_names
        self.unflatten = kwargs["environment_model"].unflatten_state        
        self.observed_outcomes = {n: [] for n in self.names} # keeps a set of all the observed outcomes as a dictionary of names to lists, which is inefficient TODO: a better distribution method
        self.outcome_counts = {n: [] for n in self.names} # a running counter of how many times an outcome was seen
        self.observed_differences = {n: [] for n in self.names}
        self.difference_counts = {n: [] for n in self.names}
        self.total_counts = {n: 0 for n in self.names}
        self.EPSILON = 1e-5

    def check_observed(self, outcome, observed):
        for i, o in enumerate(observed):
            if (o - outcome).norm() < self.EPSILON:
                return i
        return -1

    def add_observed(self, outcome, observed, counter):
        i = self.check_observed(outcome, observed)
        if i < 0:
            observed.append(outcome)
            counter.append(1)
        else:
            counter[i] += 1

    def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
        '''
        Records the different counterfactual outputs. 
        '''
        states = outcome_rollouts.get_values("state")
        state_len = state.size(1)
        states = self.unflatten(outcome_rollouts.get_values("state"), vec = True, typed=False)
        diffs = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
        fullstate_masks, outcome_probs = counterfactual_mask(outcome_rollouts)
        state_masks = self.unflatten(fullstate_masks[:, :state_len], vec = True, typed=False)
        state_diff_masks = self.unflatten(fullstate_masks[:, state_len:], vec = True, typed=False)
        for i in [k*self.num_options for k in range(outcome_rollouts.length // self.num_options)]: # TODO: assert evenly divisible
            for j in range(self.num_options):
                for name in self.names:
                    self.add_observed(states[name][i+j] * state_masks[name][i+j], self.observed_outcomes[name], self.outcome_counts[name])
                    self.add_observed(state_diffs[name][i+j] * state_diff_masks[name][i+j], self.observed_differences[name], self.difference_counts[name])
                    self.total_count[name] += 1


    def sample(self, rollouts, diff=True, name=""):
        '''
        takes a random sample of the outcomes or diffs from a particular name (or a random name) and then returns it. This has issues since you don't know which one you are getting,
        and the outcomes and difference have been separated, so you cannot just sample from each. 
        '''
        if len(name) == 0:
            counts = np.array([self.total_counts[n] for n in self.names])
            total = sum(counts)
            possible_indexes = list(range(len(self.total_counts)))
            name_index = np.random.choice(possible_indexes, rollouts.length, replace=True, p=counts / total)
            name = self.names[name_index]
        if diff:
            possible_indexes = list(range(len(self.difference_counts[name])))
            indexes = np.random.choice(possible_indexes, rollouts.length, replace=True, p=np.array(self.difference_counts[name]) / self.total_counts[name])
            diffs = []
            for i in indexes:
                diffs.append(self.observed_differences[name][i].copy())
            return torch.stack(diffs, dim=0)
        else:
            possible_indexes = list(range(len(self.outcome_counts[name])))
            indexes = np.random.choice(possible_indexes, rollouts.length, replace=True, p=np.array(self.outcome_counts[name]) / self.total_counts[name])
            outs = []
            for i in indexes:
                outs.append(self.observed_outcomes[name][i].copy())
            return torch.stack(outs, dim=0)
