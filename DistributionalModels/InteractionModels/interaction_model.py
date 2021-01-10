import numpy as np
import os, cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from EnvironmentModels.environment_model import get_selection_list, FeatureSelector, ControllableFeature
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from file_management import save_to_pickle, load_from_pickle
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from Networks.DistributionalNetworks.forward_network import forward_nets
from Networks.DistributionalNetworks.interaction_network import interaction_nets
from Networks.network import ConstantNorm, pytorch_model
from Rollouts.rollouts import ObjDict


class InteractionModel(DistributionalModel):

    def __init__(self, **kwargs):
        '''
        conditionally models the distribution of some output variable from some input variable. 
        @attr has_relative indicates if the model also models the input from some output
        @attr outcome is the outcome distribution 
        '''
        # super().__init__(**kwargs)
        # TODO: does not record which option produced which outcome
        self.has_relative = False
        self.outcome = None
        # self.option_name = kwargs["option_node"].name # might need this for something, if so, convert from contingent nodes

        self.names = kwargs["environment_model"].object_names
        self.unflatten = kwargs["environment_model"].unflatten_state
        self.target_name = kwargs["target_name"]
        self.contingent_nodes = kwargs["contingent_nodes"]
        self.num_params = sum([cn.num_params for cn in self.contingent_nodes])
        self.cont_names = [n.name for n in self.contingent_nodes]
        self.save_path = ""
        self.iscuda = False

    def featurize(self, states):
        # takes in the states, using self.target_name and self.contingent_nodes to get a list of features
        # features: target node state, state difference, always 1, always zero
        # features breakdown: target position, target velocity, target attribute, position difference
        unflattened = self.unflatten(states, vec=True, instanced=True)
        # print(unflattened.keys())
        # print([unflattened[self.target_name].clone().shape]
        #  + [unflattened[n].shape for n in self.cont_names]
        #  + [(unflattened[self.target_name] - unflattened[n]).shape for n in self.cont_names]
        #  + [torch.ones((unflattened[self.target_name].size(0), 1)).shape, torch.zeros((unflattened[self.target_name].size(0), 1)).shape])
        if type(unflattened[self.target_name]) is not torch.Tensor:
            for n in [self.target_name] + self.cont_names:
                unflattened[n] = torch.tensor(unflattened[n])
        featurized = torch.cat([unflattened[self.target_name].clone()]
         + [unflattened[n] for n in self.cont_names]
         + [unflattened[self.target_name] - unflattened[n] for n in self.cont_names], dim=1).float()
        if self.iscuda:
            return featurized.cuda()
        return featurized
         # + [torch.ones((unflattened[self.target_name].size(0), 1)), torch.zeros((unflattened[self.target_name].size(0), 1))])


    def save(self, pth):
        # def string_1dtensor(t):
        #     s = ""
        #     if len(t.squeeze().cpu().numpy().shape) > 0: 
        #         for v in t.squeeze().cpu().numpy():
        #             s += str(v) + " "
        #     else:
        #         s += str(t.squeeze().cpu().numpy()) + " "
        #     return s[:len(s)-1]
        # self.save_path = pth
        # outcomes = open(os.path.join(pth, "outcomes.txt"), 'w')
        # print(self.observed_outcomes)
        # for n in self.names:
        #     for i, (v, m) in enumerate(self.observed_outcomes[n]):
        #         print(n, v, m, self.outcome_counts[n][i])
        #         outcomes.write(n + ":out:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
        #     for (v, m) in self.observed_differences[n]:
        #         outcomes.write(n + ":dif:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
        #     for (v, m) in self.observed_both[n]:
        #         outcomes.write(n + ":both:" + string_1dtensor(v)+"\t" + string_1dtensor(m) + "\n")
        # self.observed_outcomes = {n:  [] for n in self.names} # keeps a set of all the observed outcomes as a dictionary of names to lists, which is inefficient TODO: a better distribution method
        # self.observed_differences = {n: [] for n in self.names}
        # self.observed_both = {n: [] for n in self.names}
        save_to_pickle(os.path.join(pth, "dataset_model.pkl"), self)

    def load(self):
        # def tensor1d_string(s):
        #     t = []
        #     for v in s.split(" "):
        #         t += [float(v)]
        #     return torch.tensor(t)
        # outcomes = open(os.path.join(self.save_path, "outcomes.txt"), 'r')
        # for line in outcomes:
        #     n, tpe, vm = line.split(":")
        #     v,m = vm.split("\t")
        #     # print(n, tpe, vm)
        #     # print(v)
        #     # print(m)
        #     if tpe == "out":
        #         self.observed_outcomes[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]
        #     if tpe == "dif":
        #         self.observed_differences[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]
        #     if tpe == "both":
        #         self.observed_both[n] += [(tensor1d_string(v[:len(v)]), tensor1d_string(m[:len(m)-1]))]
        return

    def train(self, passive_rollout, contingent_active_rollout, irrelevant_rollout):
        '''
        trains a model to differentiate the passive states, active states and irrelevant states. 
        '''
        return

    def sample(self, state, length, both=False, diff=True, name=""):
        '''
        takes in a list of states of length, and then 
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

    def forward_model(self, state):
        return

    def backward_model(self, target):
        return

class ClassificationNetwork(nn.Module):
    def __init__(self, num_inputs, featurizer):
        self.num_inputs = num_inputs
        self.mask = nn.parameter.Parameter(torch.ones(num_inputs))
        self.bias = nn.parameter.Parameter(torch.zeros(num_inputs))
        self.range = nn.parameter.Parameter(torch.zeros(num_inputs))
        self.renormalize = nn.parameter.Parameter(torch.zeros(num_inputs))
        self.featurizer = featurizer

    def forward(self, x):
        x = x / 84
        x = x - self.bias # centers
        l = x + self.range # if too low, will stay negative
        l = F.relu(l)
        u = x - self.range # if too high, will be positive
        u = F.relu(-u)
        x = l * u
        x = x / self.range # pushes to 0-1
        # x = x * self.mask
        return x.prod(), x

class SimpleInteractionModel(InteractionModel):
    def __init__(self, **kwargs):
        '''
        conditionally models the distribution of some output variable from some input variable. 
        @attr has_relative indicates if the model also models the input from some output
        @attr outcome is the outcome distribution 
        '''
        super().__init__(**kwargs)
        # TODO: does not record which option produced which outcome
        self.reset_lists()
        self.EPSILON = 1e-2
        self.classifier = None


    def check_observed(self, outcome, observed):
        for i, o in enumerate(observed):
            if (o - outcome).norm() < self.EPSILON:
                return i
        return -1

    def add_observed(self, inp, outcome, observed, outcomes, counter):
        i = self.check_observed(inp, observed)
        if i < 0:
            observed.append(inp)
            outcomes.append([outcome])
            counter.append([1])
        else:
            idx = self.check_observed(outcome, outcomes[i])
            if idx < 0:
                outcomes[i].append(outcome)
                counter[i].append(1)
            else:
                counter[i][idx] += 1

    def remove_outcome(self, outcome, outcomes, i):
        new_outcomes = list()
        j = self.check_observed(outcome, outcomes[i])
        if j > 0:
            outcomes[i].pop(j)

    def filter_outcomes(self, forwards, outcomes, counts):
        remove = list()
        for i in range(len(outcomes)):
            removej = list()
            for j in range(len(outcomes[i])):
                if counts[i][j] < 2:
                    removej = [j] + removej
            for j in removej:
                # print(outcomes[i][j])
                outcomes[i].pop(j)
                # self.remove_outcome(outcomes[i][j], outcomes, i)
                counts[i].pop(j)
            if len(outcomes[i]) == 0:
                remove = [i] + remove
        for i in remove:
            outcomes.pop(i)
            counts.pop(i)
            forwards.pop(i)

    def determine_mask(self, rollouts, ca_indexes):
        states = rollouts.get_values("state")
        outputs, vals = self.classifier(states)
        usage = 1 - vals.min(dim=0) # could use quantile to account for noise
        mask = usage > threshold
        return mask

    def reset_lists(self):
        self.forward_model = list()
        self.forward_reverse = list()
        self.state_model = list()
        self.state_reverse = list()
        self.forward_counts = list()
        self.state_counts = list()

    def assign_dummy_lists(self):
        self.observed_differences = {self.target_name: [(s, self.output_mask.clone()) for s in self.state_model]} # keeps a set of all observed state differences as a list
        self.difference_counts = {self.target_name: [sum(s) for s in self.state_counts]}
        self.option_name = self.contingent_nodes[0].name

    def get_target(self, rollouts, nxt=False, indexes=None):
        if nxt:
            states = rollouts.get_values("next_state")
        else:
            states = rollouts.get_values("state")
        diffs = rollouts.get_values("state_diff")
        if indexes is None: # might have issues with single states?
            # print(self.unflatten(states, vec=True, instanced=True), self.unflatten(diffs, vec=True, instanced=True))
            return torch.cat((self.unflatten(states, vec=True, instanced=True)[self.target_name], self.unflatten(diffs, vec=True, instanced=True)[self.target_name]), dim=1)
        else:
            return torch.cat((self.unflatten(states, vec=True, instanced=True)[self.target_name][indexes], self.unflatten(diffs, vec=True, instanced=True)[self.target_name][indexes]), dim=1)

    def add_observations_mask(self, rollouts, ca_indexes, featurized, input_mask, output_mask):
        target = self.get_target(rollouts, nxt = True, indexes =ca_indexes)
        features = featurized[ca_indexes]
        self.reset_lists()
        for f, o in zip(features, target):
            # if input_mask[4] == 1 and output_mask[-1] == 1:
            #     print(f*input_mask, o*output_mask, output_mask, o)
            self.add_observed(f*input_mask, o*output_mask,self.forward_model, self.forward_reverse, self.forward_counts)
            self.add_observed(o*output_mask, f*input_mask,self.state_model, self.state_reverse, self.state_counts)
        self.filter_outcomes(self.forward_model, self.forward_reverse, self.forward_counts)
        self.filter_outcomes(self.state_model, self.state_reverse, self.state_counts)
        self.assign_dummy_lists()

    def train(self, identifiers, passive_rollout, contingent_active_rollout, irrelevant_rollout, rollouts):
        '''
        trains a model to differentiate the passive states, active states and irrelevant states. 
        '''
        # compute some normal values, and then train a minimal mask when predicting the states in the contingent active rollouts
        # using a very small state mask distribution
        # This assumes that good state features already exist, and also that there is only one set of characteristic interactions
        # When training RL, this allows using the mask features for a shaped reward, or as inputs, or even with planning towards the state
        ca_indexes = np.where(identifiers > 0)
        classes = np.zeros(identifiers.shape)
        classes[ca_indexes] = 1.0
        featurized = self.featurize(rollouts.get_values("state"))
        num_features = featurized.size(1)
        self.classifier = ClassificationNetwork(num_features, self.featurize)
        self.class_optimizer = optim.Adam(self.classifier.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
        lossfn = nn.BCELoss()
        lmda = .1
        # train classifier
        for i in range(1000):
            idxes, batchvals = rollouts.get_batch(20)
            outputs, vals = self.classifier(featurized[idxes])
            loss = lossfn(outputs, classes[idxes]) + (1 - vals).abs().sum() / (20.0 * num_features) * lmda
            self.class_optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.args.max_grad_norm)
            self.class_optimizer.step()
        # determine mask
        self.mask = self.determine_mask(rollouts, ca_indexes)
        self.output_mask = None # TODO: finish this code
        # determine forward modeling
        self.add_observations(rollouts, ca_indexes, featurized, self.mask, output_mask)

    def sample(self, states, length):
        '''
        at each state output the target featurized state to go for
        give a function that defines the sampling factor
        '''
        return

    def forward_model(self, x):
        featurized = self.featurize(x)



class HackedInteractionModel(SimpleInteractionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = self.check_trace
        self.state_breakdown = [2,2,1]
        self.num_states = len(self.contingent_nodes) * 2 + 1 # contingent states, contingent state relative state, target state
        self.input_mask = torch.zeros(self.num_states * 5)
        self.possible_input_masks = list()
        at = 0
        for i in range(self.num_states):
            for k in self.state_breakdown:
                self.input_mask[at:at+k] = 1
                self.possible_input_masks.append(self.input_mask.clone())
                at = at + k
                self.input_mask = torch.zeros(self.num_states * 5)
        
        self.output_mask = torch.zeros(10)
        self.possible_output_masks = list()
        at = 0
        for i in range(2):
            for k in self.state_breakdown:
                self.output_mask[at:at+k] = 1
                self.possible_output_masks.append(self.output_mask.clone())
                at += k            
                self.output_mask = torch.zeros(10)
        self.iscuda = False

    def cuda(self):
        self.iscuda = True
        for i in range(len(self.possible_input_masks)):
            self.possible_input_masks[i] = self.possible_input_masks[i].cuda()
        for i in range(len(self.possible_output_masks)):
            self.possible_output_masks[i] = self.possible_output_masks[i].cuda()
        for i in range(len(self.forward_model)):
            self.forward_model[i] = self.forward_model[i].cuda()
        for i in range(len(self.state_model)):
            self.state_model[i] = self.state_model[i].cuda()
        self.input_mask = self.input_mask.cuda()
        self.output_mask = self.output_mask.cuda()


    def check_trace(self, x):
        def set_traces(self, flat_state):
            factored_state = self.unflatten(flat_state, vec=True, instanced=True)
            self.environment_model.set_interaction_traces(factored_state)
            trace = self.environment_model.get_interaction_trace(self.target_name)
            if len([name for name in trace if name in self.cont_names]) == len(trace) and len(trace) > 0:
                return 1
            return 0

        if len(x.shape) > 1:
            classes = []
            for flat_state in x:
                classes.append(set_traces(flat_state))
        else:
            classes = set_traces(x)
        return classes, None

    # def determine_mask(self, rollouts, ca_indexes):
    #     featurized = self.featurize(rollouts.get_values("state")[ca_indexes])
    #     states_shape = rollouts.get_values("state").shape
    #     featurized = self.featurize(rollouts.get_values("state")[ca_indexes])

    def train(self, identifiers, passive_rollout, contingent_active_rollout, irrelevant_rollout, rollouts):
        ca_indexes = np.where(identifiers > 0)
        featurized = self.featurize(rollouts.get_values("state"))
        # test all choices of target masks with the testing function
        # the testing function applies a mask and see how it divides the target masks
        # a score is given to a each target mask, and the minimum is taken
        # score is computed: predictability: number of outputs per input value (max(len(element of self.forward_reverse)))
        # diversity: number of different input values (len(self.forward_model))
        # combine score for predictability and diversity for both the forward model and the reverse model
        # less of both is better, minimum combined score for both
        best_target_input = None
        best_score = 1000
        for input_mask in self.possible_input_masks:
            for target_mask in self.possible_output_masks:
                self.add_observations_mask(rollouts, ca_indexes, featurized, input_mask, target_mask)
                # how well the input mask is at predicting the masked target
                input_diversity = len(self.forward_model)
                target_diversity = len(self.state_model)
                # input_predictability = max([len(fr) for fr in self.forward_reverse])
                input_predictability = sum([max(1, len(fr)) for fr in self.forward_reverse])
                # how well one-to-one the masked target is for the input
                target_predictability = sum([len(fr) for fr in self.state_reverse])
                print(input_mask, target_mask)
                # if input_mask[0] == 1 and target_mask[0] == 1:
                # if input_mask[9] == 1 and target_mask[5] == 1:
                # if input_mask[9] == 1 and target_mask[-1] == 1:
                # if input_mask[4] == 1 and target_mask[-1] == 1:
                if input_mask[2] == 1 and target_mask[5] == 1:
                    print([len(fr) for fr in self.forward_reverse], input_diversity)
                    print("forward", list(zip(self.forward_model, self.forward_reverse)))
                    print("state", list(zip(self.state_model, self.state_reverse)))
                    print("counter", self.forward_counts, self.state_counts)
                print("forward", self.forward_model[:1])
                print("state", self.state_model[:1])
                score = input_predictability / input_diversity + (target_predictability / target_diversity) + (input_diversity <= 1) * 1000 + (target_diversity <= 1) * 1000
                print(score, input_predictability, target_predictability, input_diversity,target_diversity)
                if score < best_score:
                    best_score = score
                    best_target_input = (input_mask.clone(), target_mask.clone())
        self.input_mask, self.output_mask = best_target_input
        self.add_observations_mask(rollouts, ca_indexes, featurized, self.input_mask, self.output_mask)

    def query_model(self, x, featurize, model, reverse, mask):
        if not self.iscuda:
            featurized = featurize(x).cpu()
        else:
            featurized = featurize(x).cuda()
        abbreviated_reverse = torch.stack([fr[0] for fr in reverse], dim=0)
        if len(featurized.shape) > 1:
            indexes = torch.zeros(featurized.size(0)).long()
            for i, inp in enumerate(model):
                indexes[torch.nonzero(((featurized - inp) * mask).norm() < self.EPSILON)] = i
            outputs = abbreviated_reverse[indexes]
        else:
            indexes = torch.zeros(1).long()
            for i, inp in enumerate(model):
                if ((featurized - inp) * mask).norm() < self.EPSILON:
                    indexes = i
                    break
            outputs = abbreviated_reverse[indexes]
        if self.iscuda:
            outputs = outputs.cuda()
        return indexes, outputs


    def forward_model(self, x): 
        return self.query_model(x, self.featurize, self.forward_model, self.forward_reverse, self.input_mask)

    def reverse_model(self, x):
        return self.query_model(x, lambda x: x, self.state_model, self.state_reverse, self.output_mask)

    def sample(self, state, length, both=False, diff=True, name=""):
        possible_indexes = list(range(len(self.difference_counts[self.target_name])))
        p = np.ones(len(self.difference_counts[self.target_name])) / float(len(self.difference_counts[self.target_name]))
        indexes = np.random.choice(possible_indexes, length, replace=True, p=p) 
        samples = []
        masks = []
        for i in indexes:
            # print(i, observed[name], len(counts[name]))
            samples.append(self.observed_differences[self.target_name][i][0].clone())
            masks.append(self.observed_differences[self.target_name][i][1].clone())
        return torch.stack(samples, dim=0), torch.stack(masks, dim=0)

nf1 = ConstantNorm(mean=0, variance=1, invvariance=1)
nf5 = ConstantNorm(mean=0, variance=5, invvariance=.2)
nf = ConstantNorm(mean=pytorch_model.wrap([84//2,84//2, 0, 0, 0]), variance=pytorch_model.wrap([84,84, 5, 5, 1]), invvariance=pytorch_model.wrap([1/84,1/84, 1/5, 1/5, 1]))
nfd = ConstantNorm(mean=pytorch_model.wrap([84//2,84//2, 0, 0, 0, 84//2,84//2, 0, 0, 0]), variance=pytorch_model.wrap([84,84, 5, 5, 1, 84,84, 5, 5, 1]), invvariance=pytorch_model.wrap([1/84,1/84, 1/5, 1/5, 1, 1/84,1/84, 1/5, 1/5, 1]))
def default_model_args():
    # nf = ConstantNorm(mean=0, variance=84)
    
    model_args = ObjDict({ 'model_type': 'neural',
     'dist': "Gaussian",
     'passive_class': 'base',
     "forward_class": 'base',
     'interaction_class': 'base',
     'init_form': 'xnorm',
     'acti': 'relu',
     'factor': 8,
     'num_layers': 2,
     'normalization_function': nf,
     'output_normalization_function': nf5,
     'interaction_minimum': .3,
     'interaction_binary': [],
     'active_epsilon': .5})
    return model_args

def load_hypothesis_model(pth):
    for root, dirs, files in os.walk(pth):
        for file in files:
            if file.find(".pt") != -1: # return the first pytorch file
                return torch.load(os.path.join(pth, file))

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.gamma = kwargs['gamma']
        self.delta = kwargs['delta']
        self.controllable = kwargs['controllable']
        self.environment_model = kwargs['environment_model']
        self.forward_model = forward_nets[kwargs['forward_class']](**kwargs)
        norm_fn, num_inputs = kwargs['normalization_function'], kwargs['num_inputs']
        kwargs['normalization_function'], kwargs['num_inputs'] = kwargs['output_normalization_function'], kwargs['num_outputs']
        self.passive_model = forward_nets[kwargs['passive_class']](**kwargs)
        kwargs['normalization_function'], kwargs['num_inputs'] = norm_fn, num_inputs
        self.interaction_binary = kwargs['interaction_binary']
        if len(self.interaction_binary) > 0:
            self.forward_threshold, self.passive_threshold = self.interaction_binary
        if kwargs['dist'] == "Discrete":
            self.dist = Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "Gaussian":
            self.dist = torch.distributions.normal.Normal#DiagGaussian(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "MultiBinary":
            self.dist = Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else:
            raise NotImplementedError
        kwargs["num_outputs"] = 1
        self.interaction_model = interaction_nets[kwargs['interaction_class']](**kwargs)
        self.network_args = ObjDict(kwargs)
        self.interaction_minimum = kwargs['interaction_minimum'] # minimum interaction level to use the active model
        self.interaction_prediction = kwargs['interaction_prediction']
        self.active_epsilon = kwargs['active_epsilon'] # minimum l2 deviation to use the active values
        self.iscuda = kwargs["cuda"]
        self.sample_continuous = True
        if self.iscuda:
            self.cuda()
        self.reset_parameters()
        # parameters used to determine which factors
        self.predict_dynamics = False
        self.name = ""
        self.control_feature = None
        self.cfselectors = list() # control feature selectors which the model captures the control of these selectors after training
        self.feature_selector = None
        self.selection_binary = None

    def save(self, pth):
        try:
            os.mkdir(pth)
        except OSError as e:
            pass
        torch.save(self, os.path.join(pth, self.name + "_model.pt"))

    def set_traces(self, flat_state, names, target_name):
        factored_state = self.environment_model.unflatten_state(pytorch_model.unwrap(flat_state), vec=False, instanced=False)
        self.environment_model.set_interaction_traces(factored_state)
        trace = self.environment_model.get_interaction_trace(target_name[0])
        trace = [t for it in trace for t in it]
        # print(np.sum(trace))
        if len([name for name in trace if name in names]) == len(trace) and len(trace) > 0:
            return 1
        return 0

    def generate_interaction_trace(self, rollouts, names, target_name):
        traces = []
        for state in rollouts.get_values("state"):
            traces.append(self.set_traces(state, names, target_name))
        return pytorch_model.wrap(traces, cuda=self.iscuda)

    def cuda(self):
        super().cuda()
        self.network_args.normalization_function.cuda()
        self.network_args.output_normalization_function.cuda()
        self.iscuda = True

    def reset_parameters(self):
        self.forward_model.reset_parameters()
        self.interaction_model.reset_parameters()
        self.passive_model.reset_parameters()

    def compute_interaction(self, forward_error, passive_loss):
        active_prediction = forward_error < self.forward_threshold
        not_passive = passive_loss > self.passive_threshold
        # forward threshold is used for the difference, passive threshold is used to determine that the accuracy is sufficient
        return ((forward_error - passive_loss < self.forward_threshold) * (forward_error < self.passive_threshold)).float() #(active_prediction+not_passive > 1).float()


    def train(self, rollouts, train_args, control, target_name=None):
        self.control_feature = control
        control_name = self.control_feature.object()
        self.name = control.object() + "->" + target_name
        self.optimizer = optim.Adam(self.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        self.predict_dynamics = train_args.predict_dynamics
        nf = self.network_args.output_normalization_function
        rv = self.network_args.output_normalization_function.reverse
        for i in range(train_args.pretrain_iters):
            idxes, batchvals = rollouts.get_batch(train_args.batch_size)
            # print(rollouts.iscuda, batchvals.iscuda, self.forward_model.iscuda)
            prediction_params = self.forward_model(self.gamma(batchvals.values.state))
            # print(prediction_params)
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            # print(passive_prediction_params)
            target = self.delta(self.get_targets(batchvals))
            active_loss = - self.dist(*prediction_params).log_probs(
                self.network_args.output_normalization_function(target))
            passive_loss = - self.dist(*passive_prediction_params).log_probs(
                self.network_args.output_normalization_function(target))
            loss = active_loss + passive_loss
            self.optimizer.zero_grad()
            (loss.mean()).backward()
            torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), train_args.max_grad_norm)
            self.optimizer.step()
            if i % train_args.log_interval == 0:
                print(self.environment_model.unflatten_state(batchvals.values.state)[0]["Paddle"],
                 self.environment_model.unflatten_state(batchvals.values.state)[0]["Action"],
                 self.environment_model.unflatten_state(batchvals.values.state_diff)[0]["Paddle"])
                print(i, ": tl: ", loss.mean().detach().cpu().numpy(),
                    ", pl: ", passive_loss.mean().detach().cpu().numpy(),
                    ", al: ", active_loss.mean().detach().cpu().numpy())
                print(
                    # self.network_args.normalization_function.reverse(passive_prediction_params[0][0]),
                    # self.network_args.normalization_function.reverse(passive_prediction_params[1][0]), 
                    "input", self.gamma(batchvals.values.state),
                    "\naoutput", rv(prediction_params[0]),
                    "\navariance", rv(prediction_params[1]),
                    "\npoutput", rv(passive_prediction_params[0]),
                    "\npvariance", rv(passive_prediction_params[1]),
                    # self.delta(batchvals.values.next_state[0]), 
                    # self.gamma(batchvals.values.state[0]),
                    "\ntarget: ", target,
                    "\nal: ", active_loss,
                    "\npl: ", passive_loss)
        bce_loss = nn.BCELoss()
        if train_args.interaction_iters > 0:
            trace = self.generate_interaction_trace(rollouts, [control_name], [target_name])
            weights = trace * 100 + 1
            weights = weights / weights.sum()
            for i in range(train_args.interaction_iters):
                idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=pytorch_model.unwrap(weights))
                interaction_likelihood = self.interaction_model(self.gamma(batchvals.values.state))
                target = trace[idxes].unsqueeze(1)
                trace_loss = bce_loss(interaction_likelihood, target)
                self.optimizer.zero_grad()
                (trace_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), train_args.max_grad_norm)
                self.optimizer.step()
                if i % train_args.log_interval == 0:
                    print(i, ": tl: ", trace_loss)
                    print("training: ", interaction_likelihood,
                        "\ntarget: ", target)
                    print(target, interaction_likelihood)
            self.interaction_model.needs_grad=False # no gradient will pass through the interaction model
        if train_args.epsilon_schedule == -1:
            interaction_schedule = lambda i: 1
        else:
            interaction_schedule = lambda i: np.power(0.5, (i/train_args.epsilon_schedule))
        for i in range(train_args.num_iters):
            idxes, batchvals = rollouts.get_batch(train_args.batch_size)
            prediction_params = self.forward_model(self.gamma(batchvals.values.state))
            interaction_likelihood = self.interaction_model(self.gamma(batchvals.values.state))
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            target = self.network_args.output_normalization_function(self.delta(self.get_targets(batchvals)))
            passive_loss = - self.dist(*passive_prediction_params).log_probs(target)
            forward_error = - self.dist(*prediction_params).log_probs(target)
            forward_loss = forward_error * interaction_likelihood.clone().detach() # detach so the interaction is trained only through discrimination 
            # version with interaction loss
            # interaction_loss = (1-interaction_likelihood) * train_args.weighting_lambda + interaction_likelihood * torch.exp(passive_loss) # try to make the interaction set as large as possible, but don't penalize for passive dynamics states
            
            # version which varies the input and samples the trained forward model
            # interaction_diversity_loss = interaction_likelihood * torch.sigmoid(torch.max(self.delta(batchvals.next_state) - self.forward_model(self.gamma(self.environment_model.sample_feature(batchvals.states, self.controllable))), dim = 1).norm(dim=1)) # batch size, num samples, state size
            # version which varies the input and samples the true model (all_controlled_next_state is an n x state size tensor containing the next state of the state given n settings of of the controllable feature(s))
            
            # version which uses diversity loss
            # next_state_broadcast = torch.stack([target.clone() for _ in range(batchvals.values.all_state_next.size(1))], dim=1) # manual state broadcasting
            # interaction_diversity_loss = interaction_likelihood * torch.sigmoid(torch.max(self.delta(batchvals.values.all_state_next) - next_state_broadcast, dim = 1)[0].norm(dim=1)) # batch size, num samples, state size
            # loss = (passive_loss + forward_loss + interaction_diversity_loss + interaction_loss).sum()
            # version without diversity loss or interaction loss

            # version with adversarial loss max(0, P(x|am) - P(x|pm)) - interaction_model
            # interaction_loss = bce_loss(torch.max(torch.exp(forward_error.clone().detach()) - torch.exp(passive_loss.clone().detach()), 0), interaction_likelihood)

            # version with binarized loss
            if train_args.interaction_iters <= 0:
                interaction_binaries = self.compute_interaction(forward_error.clone().detach(), passive_loss.clone().detach())
                interaction_loss = bce_loss(interaction_likelihood, interaction_binaries)

            loss = (passive_loss + forward_loss * interaction_schedule(i) + forward_error * (1-interaction_schedule(i))).sum() + interaction_loss.sum()
            self.optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), train_args.max_grad_norm)
            self.optimizer.step()
            if i % train_args.log_interval == 0:
                # print(i, ": pl: ", pytorch_model.unwrap(passive_loss.norm()), " fl: ", pytorch_model.unwrap(forward_loss.norm()), 
                #     " il: ", pytorch_model.unwrap(interaction_loss.norm()), " dl: ", pytorch_model.unwrap(interaction_diversity_loss.norm()))
                print(i, ": pl: ", pytorch_model.unwrap(passive_loss.mean()), " fl: ", pytorch_model.unwrap(forward_loss.mean()))
                print(
                    # "input", self.gamma(batchvals.values.state),
                    "\ninteraction", interaction_likelihood,
                    "\nbinaries", interaction_binaries,
                    # "\naoutput", rv(prediction_params[0]),
                    # "\navariance", rv(prediction_params[1]),
                    # "\npoutput", rv(passive_prediction_params[0]),
                    # "\npvariance", rv(passive_prediction_params[1]),
                    # self.delta(batchvals.values.next_state[0]), 
                    # self.gamma(batchvals.values.state[0]),
                    # "\ntarget: ", rv(target),
                    "\nae: ", forward_error,
                    "\nal: ", forward_loss,
                    "\npl: ", passive_loss)

    def assess_losses(self, test_rollout):
        prediction_params = self.forward_model(self.gamma(test_rollout.values.state))
        interaction_likelihood = self.interaction_model(test_rollout.values.states)
        passive_prediction_params = self.passive_model(self.delta(test_rollout.values.state))
        passive_loss = - self.dist(passive_prediction_params).log_probs(self.delta(test_rollout.values.next_state))
        forward_loss = - self.dist(prediction_params).log_probs(self.delta(test_rollout.values.next_state)) * interaction_likelihood
        interaction_loss = (1-interaction_likelihood) * train_args.interaction_lambda + interaction_likelihood * torch.exp(passive_loss) # try to make the interaction set as large as possible, but don't penalize for passive dynamics states
        return passive_loss, forward_loss, interaction_loss

    def assess_error(self, test_rollout):
        rv = self.network_args.output_normalization_function.reverse
        interaction, forward, passive = self.hypothesize(test_rollout.get_values("state"))
        targets = self.get_targets(test_rollout)
        sfe = (rv(forward) - self.delta(targets)).norm(dim=1) * interaction # per state forward error
        spe = (rv(forward) - self.delta(targets)).norm(dim=1) * interaction # per state passive error
        return sfe.abs().mean(), spe.abs().mean()

    def predict_next_state(self, state):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size
        rv = self.network_args.output_normalization_function.reverse
        inter = self.interaction_model(self.gamma(state))
        intera = inter.clone()
        intera[inter > self.interaction_minimum] = 1
        intera[inter < self.interaction_minimum] = 0
        if self.predict_dynamics:
            fpred, ppred = self.delta(state) + rv(self.forward_model(self.gamma(state))[0]), self.delta(state) + rv(self.passive_model(self.delta(state))[0])
        else:
            fpred, ppred = rv(self.forward_model(self.gamma(state))[0]), rv(self.passive_model(self.delta(state))[0])
        if len(state.shape) == 1:
            return (inter, fpred) if pytorch_model.unwrap(inter) > self.interaction_minimum else (inter, ppred)
        else:
            # inter_assign = torch.cat((torch.arange(state.shape[0]).unsqueeze(1), intera), dim=1).long()
            pred = torch.stack((fpred, ppred), dim=1)
            # print(inter_assign.shape, pred.shape)
            intera = pytorch_model.wrap(intera.squeeze().long(), cuda=self.iscuda)
            pred = pytorch_model.wrap(pred[torch.arange(pred.shape[0]).long(), intera], cuda=self.iscuda)
        return inter, pred

    def get_targets(self, rollouts):
        # the target is whether we predict the state diff or the next state
        if self.predict_dynamics:
            targets = rollouts.get_values('state_diff')
        else:
            targets = rollouts.get_values('next_state')
        return targets

    def test_forward(self, states, targets, interact=True):
        # gives back the difference between the prediction mean and the actual next state for different sampled feature values
        rv = self.network_args.output_normalization_function.reverse
        batch_pred, inters = list(), list()
        for state in states:
            # print(self.gamma(self.control_feature.sample_feature(state)))
            inter, pred_states = self.predict_next_state(self.control_feature.sample_feature(state))
            batch_pred.append(pred_states), inters.append(inter)
        batch_pred, inters = torch.stack(batch_pred, dim=0), torch.stack(inters, dim=0) # batch x samples x state, batch x samples x 1
        next_state_broadcast = pytorch_model.wrap(torch.stack([self.delta(targets).clone() for _ in range(batch_pred.size(1))], dim=1), cuda=self.iscuda)
        # compare predictions with the actual next state to make sure there are differences
        state_check = (next_state_broadcast - batch_pred).abs()
        # should be able to predict at least one of the next states accurately
        match = state_check.min(dim=1)[0]
        match_keep = match.clone()
        match_keep[match <= self.interaction_prediction] = 1
        match_keep[match > self.interaction_prediction] = 0
        if interact: # if the interaction value is less, assume there is no difference because the model is flawed
            inters[inters > self.interaction_minimum] = 1
            inters[inters <= self.interaction_minimum] = 0
            return (state_check * match_keep.unsqueeze(1)) * inters # batch size, num samples, state size
        return state_check * match_keep.unsqueeze(1)

    def determine_active_set(self, rollouts):
        states = rollouts.get_values('state')
        targets = self.get_targets(rollouts)
        sample_diffs = torch.max(self.test_forward(states, targets), dim=1)[0]
        test_diff = torch.max(sample_diffs, dim=0)[0]
        v = torch.zeros(test_diff.shape)
        v[test_diff > self.active_epsilon] = 1
        print(sample_diffs, v)
        self.selection_binary = v
        self.feature_selector = self.environment_model.get_subset(self.delta, v)
        self.cfselectors = list()
        for ff in self.feature_selector.flat_features:
            factored = self.environment_model.flat_to_factored(ff)
            single_selector = FeatureSelector([ff], {factored[0]: factored[1]})
            rng = self.determine_range(rollouts, single_selector)
            self.cfselectors.append(ControllableFeature(single_selector, rng, 1, self))
        self.selection_list = get_selection_list(self.cfselectors)
        return self.feature_selector, self.cfselectors

    def determine_range(self, rollouts, active_delta):
        if self.predict_dynamics:
            state_diffs = rollouts.get_values('state_diff')
            return active_delta(state_diffs).min(dim=0)[0], active_delta(state_diffs).max(dim=0)[0]
        else:
            states = rollouts.get_values('state')
            return active_delta(states).min(dim=0)[0], active_delta(states).max(dim=0)[0]

    def hypothesize(self, state): # at present hypothesize only looks at the means of the distributions
        rv = self.network_args.output_normalization_function.reverse
        return self.interaction_model(self.gamma(state)), rv(self.forward_model(self.gamma(state))[0]), rv(self.passive_model(self.delta(state))[0])

    def sample(self, states):
        if self.sample_continuous:
            weights = np.random.random((len(self.cfselectors,))) # random weight vector
            lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
            len_cfs = np.array([i-j for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])
            edited_features = lower_cfs + len_cfs * weights
            new_states = states.clone()
            for f, cfs in zip(edited_features, self.cfselectors):
                cfs.assign_feature(new_states, f)
            return self.delta(new_states), 
        else: # sample discrete with weights
            return


interaction_models = {'neural': NeuralInteractionForwardModel}