import numpy as np
import os, cv2, time
import torch
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from file_management import save_to_pickle, load_from_pickle

# lock_mask = torch.tensor([-1,-1,-1,-1,-1]).float()
lock_mask = torch.tensor([0,0,1,1,0]).float()

class DatasetModel(DistributionalModel):
    def __init__(self, **kwargs):
        '''
        conditionally models the distribution of some output variable from some input variable. 
        @attr has_relative indicates if the model also models the input from some output
        @attr outcome is the outcome distribution 
        '''
        super().__init__(**kwargs)
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
        for i in [k*self.num_params for k in range(outcome_rollouts.length // self.num_params)]: # TODO: assert evenly divisible
            for j in range(self.num_params):
                self.add_observed(states[i+j] * state_masks[i+j], self.observed_outcomes, self.outcome_counts)
                self.add_observed(state_diffs[i+j] * state_diff_masks[i+j], self.observed_differences, self.difference_counts)
                self.total_count += 1


    def sample(self, state, length, diff=True):
        '''
        takes a random sample of the outcomes or diffs and then returns it. This has issues since you don't know which one you are getting,
        and the outcomes and difference have been separated, so you cannot just sample from each. 
        '''
        if diff:
            possible_indexes = list(range(len(self.difference_counts)))
            indexes = np.random.choice(possible_indexes, length, replace=True, p=np.array(self.difference_counts) / self.total_count)
            diffs = []
            for i in indexes:
                diffs.append(self.observed_differences[i].copy())
            return torch.stack(diffs, dim=0)
        else:
            possible_indexes = list(range(len(self.outcome_counts)))
            indexes = np.random.choice(possible_indexes, length, replace=True, p=np.array(self.outcome_counts) / self.total_count)
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
        super().__init__(**kwargs)
        # TODO: does not record which option produced which outcome
        self.names = kwargs["environment_model"].object_names
        self.unflatten = kwargs["environment_model"].unflatten_state        
        self.observed_outcomes = {n: [] for n in self.names} # keeps a set of all the observed outcomes as a dictionary of names to lists, which is inefficient TODO: a better distribution method
        self.outcome_counts = {n: [] for n in self.names} # a running counter of how many times an outcome was seen
        self.observed_differences = {n: [] for n in self.names}
        self.difference_counts = {n: [] for n in self.names}
        self.observed_both = {n: [] for n in self.names}
        self.both_counts = {n: [] for n in self.names}
        self.total_counts = {n: 0 for n in self.names}
        self.EPSILON = 1e-5
        self.sample_zero = True
        self.flat_sample = True
        self.save_path = ""

    def check_observed(self, outcome, observed):
        for i, o in enumerate(observed):
            if (o[0] - outcome).norm() < self.EPSILON:
                return i
        return -1

    def add_observed(self, outcome, mask, observed, counter):
        i = self.check_observed(outcome, observed)
        if i < 0:
            observed.append((outcome, mask))
            counter.append(1)
        else:
            counter[i] += 1

    def save(self, pth):
        def string_1dtensor(t):
            s = ""
            if len(t.squeeze().cpu().numpy().shape) > 0: 
                for v in t.squeeze().cpu().numpy():
                    s += str(v) + " "
            else:
                s += str(t.squeeze().cpu().numpy()) + " "
            return s[:len(s)-1]
        self.save_path = pth
        outcomes = open(os.path.join(pth, "outcomes.txt"), 'w')
        print(self.observed_outcomes)
        for n in self.names:
            for i, (v, m) in enumerate(self.observed_outcomes[n]):
                print(n, v, m, self.outcome_counts[n][i])
                outcomes.write(n + ":out:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
            for (v, m) in self.observed_differences[n]:
                outcomes.write(n + ":dif:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
            for (v, m) in self.observed_both[n]:
                outcomes.write(n + ":both:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
        self.observed_outcomes = {n:  [] for n in self.names} # keeps a set of all the observed outcomes as a dictionary of names to lists, which is inefficient TODO: a better distribution method
        self.observed_differences = {n: [] for n in self.names}
        self.observed_both = {n: [] for n in self.names}
        save_to_pickle(os.path.join(pth, "dataset_model.pkl"), self)

    def load(self):
        def tensor1d_string(s):
            t = []
            for v in s.split(" "):
                t += [float(v)]
            return torch.tensor(t)
        outcomes = open(os.path.join(self.save_path, "outcomes.txt"), 'r')
        for line in outcomes:
            n, tpe, vm = line.split(":")
            v,m = vm.split("\t")
            # print(n, tpe, vm)
            # print(v)
            # print(m)
            if tpe == "out":
                self.observed_outcomes[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]
            if tpe == "dif":
                self.observed_differences[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]
            if tpe == "both":
                self.observed_both[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]


    def reduce_range(self, names):
        self.names = [ n for n in names if n in self.observed_outcomes]
        self.observed_outcomes = {n:  self.observed_outcomes[n] for n in self.names} # keeps a set of all the observed outcomes as a dictionary of names to lists, which is inefficient TODO: a better distribution method
        # self.outcome_counts = {n: self.outcome_counts[n] for n in self.names} # a running counter of how many times an outcome was seen
        self.observed_differences = {n: self.observed_differences[n] for n in self.names}
        # self.difference_counts = {n: self.difference_counts[n] for n in self.names}
        self.observed_both = {n: self.observed_both[n] for n in self.names}
        # self.both_counts = {n: self.both_counts[n] for n in self.names}
        # self.total_counts = {n: self.total_counts[n] for n in self.names}

    def train(self, counterfactual_rollouts, non_counterfactual_rollouts, outcome_rollouts):
        '''
        Records the different counterfactual outputs. 
        '''
        states = outcome_rollouts.get_values("state")
        state_len = states.size(1)
        states = self.unflatten(states, vec = True, typed=False)
        diffs = self.unflatten(outcome_rollouts.get_values("state_diff"), vec = True, typed=False)
        both = {n: torch.cat((states[n], diffs[n]), dim=1) for n in self.names}
        fullstate_masks, outcome_probs = counterfactual_mask(self.names, self.num_params, outcome_rollouts)
        if lock_mask.sum() > 0: # a hack to fix the mask 
            # TODO: a hack to remove positive z, a hack to only reward once every N time steps
            state_masks = self.unflatten(fullstate_masks[:, :state_len], vec = True, typed=False)
            state_diff_masks = self.unflatten(fullstate_masks[:, state_len:], vec = True, typed=False)
            for n in self.names:
                # print(n, state_masks[n].shape, lock_mask.shape, state_masks[n][(state_masks[n].sum(dim=1) > 0).nonzero()].squeeze().shape)
                # print(n, "posvel", state_masks[n][(state_masks[n][:,2] > 0).nonzero().squeeze()])
                if state_masks[n][0].shape == lock_mask.shape:
                    state_masks[n][(states[n][:,2] > 0).nonzero().squeeze(),:] = torch.zeros(lock_mask.shape)
                    state_masks[n][(state_masks[n].sum(dim=1) > 0).nonzero().squeeze(),:] = lock_mask.clone()
                    state_diff_masks[n][(state_diff_masks[n].sum(dim=1) > 0).nonzero().squeeze(),:] = lock_mask.clone()
            both_masks = {n: torch.cat((state_masks[n], state_diff_masks[n]), dim=1) for n in self.names}
        else:
            state_masks = self.unflatten(fullstate_masks[:, :state_len], vec = True, typed=False)
            state_diff_masks = self.unflatten(fullstate_masks[:, state_len:], vec = True, typed=False)
            both_masks = {n: torch.cat((state_masks[n], state_diff_masks[n]), dim=1) for n in self.names}
        for i in [k*self.num_params for k in range(outcome_rollouts.filled // self.num_params)]: # TODO: assert evenly divisible
            for j in range(self.num_params):
                for name in self.names:
                    self.add_observed(states[name][i+j] * state_masks[name][i+j], state_masks[name][i+j], self.observed_outcomes[name], self.outcome_counts[name])
                    self.add_observed(diffs[name][i+j] * state_diff_masks[name][i+j], state_diff_masks[name][i+j], self.observed_differences[name], self.difference_counts[name])
                    self.add_observed(both[name][i+j] * both_masks[name][i+j], both_masks[name][i+j], self.observed_both[name], self.both_counts[name])
                    self.total_counts[name] += 1


    def sample(self, state, length, both=False, diff=True, name=""):
        '''
        takes a random sample of the outcomes or diffs from a particular name (or a random name) and then returns it. This has issues since you don't know which one you are getting,
        and the outcomes and difference have been separated, so you cannot just sample from each. 
        '''
        if len(name) == 0:
            counts = np.array([self.total_counts[n] for n in self.names])
            total = sum(counts)
            possible_indexes = list(range(len(self.total_counts)))
            name_index = np.random.choice(possible_indexes, length, replace=True, p=counts / total)[0]
            name = self.names[name_index]
        def get_sample(counts, observed):
            possible_indexes = list(range(len(counts[name])))
            # possible_indexes = [2 for _ in range(len(counts[name]))]
            # if self.sample_zero:

            # else:
            #     total_counts = self.total_counts[name]
            # print(counts[name])
            # print(self.total_counts[name])
            if self.flat_sample:
                p = np.ones(len(counts[name])) / float(len(counts[name]))
            else:
                p = np.array(counts[name]) / self.total_counts[name]
            indexes = np.random.choice(possible_indexes, length, replace=True, p=p)
            samples = []
            masks = []
            for i in indexes:
                # print(i, observed[name], len(counts[name]))
                samples.append(observed[name][i][0].clone())
                masks.append(observed[name][i][1].clone())
            return torch.stack(samples, dim=0), torch.stack(masks, dim=0)
        if both:
            return get_sample(self.both_counts, self.observed_both)
        elif diff:
            return get_sample(self.difference_counts, self.observed_differences)
        else:
            return get_sample(self.outcome_counts, self.observed_outcomes)

    def merge_sample(self, name, both=False, diff=True):
        '''
        merges samples based on frequency, which might improve performance
        '''
        if both:
            dataset = dataset_model.observed_both[name]
        elif diff:
            dataset = dataset_model.observed_differences[name]
        else:
            dataset = dataset_model.observed_outcomes[name]
        mask = dataset