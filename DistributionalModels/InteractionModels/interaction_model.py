import numpy as np
import os, cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from file_management import save_to_pickle, load_from_pickle
from Networks.distributions import Bernoulli, Categorical, DiagGaussian

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

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, **kwargs):
        self.gamma = kwargs['gamma']
        self.delta = kwargs['delta']
        self.controllable = kwargs['controllable']
        self.environment_model = kwargs['environment_model']
        self.passive_model = kwargs['passive_class'](**kwargs)
        self.forward_model = kwargs['forward_class'](**kwargs)
        if kwargs['dist'] == "Discrete":
            self.forward_dist = Categorical(self.base.output_size, self.delta.num_outputs)
        elif kwargs['dist'] == "Gaussian":
            self.forward_dist = DiagGaussian(self.base.output_size, self.delta.output_size)
        elif kwargs['dist'] == "MultiBinary":
            self.forward_dist = Bernoulli(self.base.output_size, self.delta.output_size)
        else:
            raise NotImplementedError
        self.interaction_model = kwargs['interaction_class'](**kwargs)
        self.initialize_weights()

    def initialize_weights(self):
        self.forward_model.initialize_weights()
        self.interaction_model.initialize_weights()
        self.passive_model.initialize_weights()

    def test_forward(self, states, next_states):
        return self.delta(batchvals.next_state) - self.forward_model(self.gamma(self.environment_model.sample_feature(batchvals.states))) # batch size, num samples, state size

    def train(self, train_args, rollouts):
        self.optimizer = optim.Adam(self.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        for i in range(train_args.pretrain_iters):
            idxes, batchvals = rollouts.get_batch(train_args.batch_size)
            prediction_params = self.forward_model(self.gamma(batchvals.values.state))
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            loss = - self.dist(prediction_params).log_probs(self.delta(batchvals.values.next_state)) - self.dist(passive_prediction_params).log_probs(self.delta(batchvals.next_state))
            self.optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), self.args.max_grad_norm)
        for i in range(train_args.train_iters):
            idxes, batchvals = rollouts.get_batch(train_args.batch_size)
            prediction_params = self.forward_model(self.gamma(batchvals.values.state))
            interaction_likelihood = self.interaction_model(batchvals.values.states)
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            passive_loss = - self.dist(passive_prediction_params).log_probs(self.delta(batchvals.values.next_state))
            forward_loss = - self.dist(prediction_params).log_probs(self.delta(batchvals.values.next_state)) * interaction_likelihood
            interaction_loss = (1-interaction_likelihood) * train_args.interaction_lambda + interaction_likelihood * torch.exp(passive_loss) # try to make the interaction set as large as possible, but don't penalize for passive dynamics states
            # version which varies the input and samples the trained forward model
            # interaction_diversity_loss = interaction_likelihood * F.sigmoid(torch.max(self.delta(batchvals.next_state) - self.forward_model(self.gamma(self.environment_model.sample_feature(batchvals.states, self.controllable))), dim = 1).norm(dim=1)) # batch size, num samples, state size
            # version which varies the input and samples the true model (all_controlled_next_state is an n x state size tensor containing the next state of the state given n settings of of the controllable feature(s))
            interaction_diversity_loss = interaction_likelihood * F.sigmoid(torch.max(self.delta(batchvals.values.next_state) - self.delta(batchvals.all_controlled_next_state), dim = 1).norm(dim=1)) # batch size, num samples, state size
            loss = (passive_loss + forward_loss + interaction_diversity_loss + interaction_loss).sum()
            self.optimizer.zero_grad()
            (loss).backward()
            torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), self.args.max_grad_norm)

    def assess_losses(self, test_rollout):
        prediction_params = self.forward_model(self.gamma(test_rollout.values.state))
        interaction_likelihood = self.interaction_model(test_rollout.values.states)
        passive_prediction_params = self.passive_model(self.delta(test_rollout.values.state))
        passive_loss = - self.dist(passive_prediction_params).log_probs(self.delta(test_rollout.values.next_state))
        forward_loss = - self.dist(prediction_params).log_probs(self.delta(test_rollout.values.next_state)) * interaction_likelihood
        interaction_loss = (1-interaction_likelihood) * train_args.interaction_lambda + interaction_likelihood * torch.exp(passive_loss) # try to make the interaction set as large as possible, but don't penalize for passive dynamics states
        return passive_loss, forward_loss, interaction_loss

    def determine_active_set(self, rollouts):
        states, next_states = rollouts.get_values('state'), rollouts.get_values('next_state')
        test_diff = torch.max(torch.max(self.test_forward(states), dim=1), dim=0)
        v = torch.zeros(test_diff.shape)
        v[test_diff > self.active_epsilon] = 1 
        return delta.get_subset(v)

    def hypothesize(self, state):
        return self.interaction_model(state), self.forward_model(state)

