import numpy as np
import os, cv2, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from EnvironmentModels.environment_model import get_selection_list, FeatureSelector, ControllableFeature, sample_multiple
from EnvironmentModels.environment_normalization import hardcode_norm
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from DistributionalModels.InteractionModels.dummy_models import DummyModel
from DistributionalModels.InteractionModels.state_management import StateSet
from file_management import save_to_pickle, load_from_pickle
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from Networks.DistributionalNetworks.forward_network import forward_nets
from Networks.DistributionalNetworks.interaction_network import interaction_nets
from Networks.network import ConstantNorm, pytorch_model
from Networks.input_norm import InterInputNorm, PointwiseNorm
from Rollouts.rollouts import ObjDict
from tianshou.data import Collector, Batch, ReplayBuffer

def nflen(x):
    return ConstantNorm(mean= pytorch_model.wrap(sum([[84//2,84//2,0,0,0] for i in range(x // 5)], list())), variance = pytorch_model.wrap(sum([[84,84, 5, 5, 1] for i in range(x // 5)], list())), invvariance = pytorch_model.wrap(sum([[1/84,1/84, 1/5, 1/5, 1] for i in range(x // 5)], list())))

nf5 = ConstantNorm(mean=0, variance=5, invvariance=.2)

def default_model_args(predict_dynamics, model_class, norm_fn=None, delta_norm_fn=None):    
    model_args = ObjDict({ 'model_type': 'neural',
     'dist': "Gaussian",
     'passive_class': model_class,
     "forward_class": model_class,
     'interaction_class': model_class,
     'init_form': 'xnorm',
     'activation': 'relu',
     'factor': 8,
     'num_layers': 2,
     'use_layer_norm': False,
     'normalization_function': norm_fn,
     'delta_normalization_function': delta_norm_fn,
     'interaction_binary': [],
     'active_epsilon': .5,
     'base_variance': .0001 # TODO: this parameter is extremely sensitive, and that is a problem
     })
    return model_args

def load_hypothesis_model(pth):
    for root, dirs, files in os.walk(pth):
        for file in files:
            if file.find(".pt") != -1: # return the first pytorch file
                return torch.load(os.path.join(pth, file))

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # set input and output
        self.gamma = kwargs['gamma']
        self.delta = kwargs['delta']
        self.controllable = kwargs['controllable'] # controllable features USED FOR (BEFORE) training
        
        # environment model defines object factorization
        self.environment_model = kwargs['environment_model']
        self.env_name = self.environment_model.environment.name

        # construct the active model
        kwargs['post_dim'] = 0 # no post-state
        self.forward_model = forward_nets[kwargs['forward_class']](**kwargs)

        # set the passive model        
        norm_fn, num_inputs = kwargs['normalization_function'], kwargs['num_inputs']
        kwargs['normalization_function'], kwargs['num_inputs'] = kwargs['delta_normalization_function'], kwargs['num_outputs']
        fod = kwargs['first_obj_dim'] # there is no first object since the passive model only has one object
        kwargs['first_obj_dim'] = 0
        self.passive_model = forward_nets[kwargs['passive_class']](**kwargs)
        kwargs['normalization_function'], kwargs['num_inputs'], kwargs['first_obj_dim'] = norm_fn, num_inputs, fod

        # define the output for the forward and passive models
        if kwargs['dist'] == "Discrete":
            self.dist = Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "Gaussian":
            self.dist = torch.distributions.normal.Normal#DiagGaussian(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "MultiBinary":
            self.dist = Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else:
            raise NotImplementedError

        # construct the interaction model        
        odim = kwargs['output_dim']
        kwargs['output_dim'] = 1
        kwargs["num_outputs"] = 1
        self.interaction_model = interaction_nets[kwargs['interaction_class']](**kwargs)
        kwargs['output_dim'] = odim

        # set the values for what defines an interaction, in training and test
        self.interaction_binary = kwargs['interaction_binary']
        if len(self.interaction_binary) > 0:
            self.difference_threshold, self.forward_threshold, self.passive_threshold = self.interaction_binary
        self.interaction_prediction = kwargs['interaction_prediction']

        # Learn an interaction model which is based on distance to target
        # interaction_prediction remains the threshold for interaction
        self.interaction_distance = kwargs['interaction_distance']

        # output limits
        self.control_min, self.control_max = np.zeros(kwargs['num_outputs']), np.zeros(kwargs['num_outputs'])
        self.object_dim = kwargs["object_dim"]
        self.multi_instanced = kwargs["multi_instanced"]

        # assign norm values
        self.normalization_function = kwargs["normalization_function"]
        self.delta_normalization_function = kwargs["delta_normalization_function"]
        # note that self.output_normalization_function is defined in TRAIN, because that is wehre predict_dynamics is assigned
        self.active_epsilon = kwargs['active_epsilon'] # minimum l2 deviation to use the active values
        self.iscuda = kwargs["cuda"]
        # self.sample_continuous = True
        self.selection_binary = pytorch_model.wrap(torch.zeros((self.delta.output_size(),)), cuda=self.iscuda)
        if self.iscuda:
            self.cuda()
        self.reset_parameters()
        # parameters used to determine which factors to use
        self.sample_continuous = False
        self.predict_dynamics = False
        self.name = ""
        self.control_feature = None
        self.cfselectors = list() # control feature selectors which the model captures the control of these selectors AFTER training
        self.feature_selector = None
        self.selection_binary = None

        self.sample_able = StateSet()

    def save(self, pth):
        try:
            os.mkdir(pth)
        except OSError as e:
            pass
        em = self.environment_model 
        self.environment_model = None
        torch.save(self, os.path.join(pth, self.name + "_model.pt"))
        self.environment_model = em

    def bin_trace(self, names, tr):
        if self.multi_instanced: # returns a vector of length instances which is 1 where there is an interaction with the desired object
            return np.array([float(len([n for n in trace if n in names]) > 0) for trace in tr])
        else:
            trace = [t for it in tr for t in it] # flattens out instances
            if len([name for name in trace if name in names]) > 0:
                return 1
            return 0

    def _set_traces(self, flat_state, names, target_name):
        # sets trace for one state, if the target has an interaction with at least one name in names
        factored_state = self.environment_model.unflatten_state(pytorch_model.unwrap(flat_state), vec=False, instanced=False)
        self.environment_model.set_interaction_traces(factored_state)
        tr = self.environment_model.get_interaction_trace(target_name[0])
        return self.bin_trace(names, tr)

    def check_current_trace(self, source, target):
        tr = self.environment_model.get_interaction_trace(target)
        return pytorch_model.wrap(self.bin_trace([source], tr), cuda=self.iscuda)

    def generate_interaction_trace(self, rollouts, names, target_name):
        '''
        a trace basically determines if an interaction occurred, based on the "traces"
        in the environment model that record true object interactions
        '''
        traces = []
        for state in rollouts.get_values("state"):
            traces.append(self._set_traces(state, names, target_name))

            # locality = np.array([i - self.interaction_distance for i in range(int(self.interaction_distance * 2 + 1))])
            # locality[self.interaction_distance+1:] = 0
            # traces = np.convolve(traces, locality, 'same')
            # traces = self.interaction_distance - traces


        return pytorch_model.wrap(traces, cuda=self.iscuda)

    def _adjust_interaction_trace(self, traces):
        if self.interaction_distance > 0:
            traces = pytorch_model.unwrap(traces)
            locality = np.array([np.exp(-abs(i - self.interaction_distance) / 4) for i in range(int(self.interaction_distance * 2 + 1))])
            locality[self.interaction_distance+1:] = 0 # set the midpoint to be 1
            traces = np.convolve(traces, locality, 'same')
            print(traces[:100])
            print(traces[100:200])
        return pytorch_model.wrap(traces, cuda=self.iscuda)


    def cpu(self):
        super().cpu()
        self.forward_model.cpu()
        self.interaction_model.cpu()
        self.passive_model.cpu()
        self.normalization_function.cpu()
        self.delta_normalization_function.cpu()
        if hasattr(self, "output_normalization_function"):
            self.output_normalization_function.cpu()
        if self.selection_binary is not None:
            self.selection_binary = self.selection_binary.cpu()
        self.iscuda = False

    def cuda(self):
        super().cuda()
        self.forward_model.cuda()
        self.interaction_model.cuda()
        self.passive_model.cuda()
        if hasattr(self, "normalization_function"):
            self.normalization_function.cuda()
        if hasattr(self, "delta_normalization_function"):
            self.delta_normalization_function.cuda()
        if hasattr(self, "output_normalization_function"):
            self.output_normalization_function.cuda()
        if self.selection_binary is not None:
            self.selection_binary = self.selection_binary.cuda()
        self.iscuda = True

    def reset_parameters(self):
        self.forward_model.reset_parameters()
        self.interaction_model.reset_parameters()
        self.passive_model.reset_parameters()

    def compute_interaction(self, forward_mean, passive_mean, target):
        '''computes an interaction binary, which defines if the active prediction is high likelihood enough
        the passive is low likelihood enough, and the difference is sufficiently large
        TODO: there should probably be a mechanism that incorporates variance explicitly
        TODO: there should be a way of discounting components of state that are ALWAYS predicted with high probability
        '''
        
        # values based on log probability
        active_prediction = forward_mean < self.forward_threshold # the prediction must be good enough (negative log likelihood)
        not_passive = passive_mean > self.passive_threshold # the passive prediction must be bad enough
        difference = forward_mean - passive_mean < self.difference_threshold # the difference between the two must be large enough
        # return ((passive_mean > self.passive_threshold) * (forward_mean - passive_mean < self.difference_threshold) * (forward_mean < self.forward_threshold)).float() #(active_prediction+not_passive > 1).float()

        # values based on the absolute error between the mean and the target
        # forward_error = self.get_error(forward_mean, target, normalized=True)
        # passive_error = self.get_error(passive_mean, target, normalized=True)
        # passive_performance = passive_error > self.passive_threshold
        # forward_performance = forward_error < self.forward_threshold
        # difference = passive_error - forward_error > self.difference_threshold

        # forward threshold is used for the difference, passive threshold is used to determine that the accuracy is sufficient
        # return ((forward_error - passive_loss < self.forward_threshold) * (forward_error < self.passive_threshold)).float() #(active_prediction+not_passive > 1).float()
        # passive can't predict well, forward is better, forward predicts well
        potential = (active_prediction * difference).float()
        # print(passive_mean, forward_mean, difference)
        return ((not_passive) * (active_prediction) * (difference)).float(), potential #(active_prediction+not_passive > 1).float()

    def get_error(self, mean, target, normalized = False):
        rv = self.output_normalization_function.reverse
        nf = self.output_normalization_function
        # print((rv(mean) - target).abs().sum(dim=1).shape)
        if normalized:
            # print(mean, nf(target))
            return (mean - nf(target)).abs().sum(dim=1).unsqueeze(1)
        else:
            return (rv(mean) - target).abs().sum(dim=1).unsqueeze(1)

    def get_prediction_error(self, rollouts, active=False):
        pred_error = []
        nf = self.output_normalization_function
        done_flags = 1-rollouts.get_values("done")
        if active:
            dstate = self.gamma(rollouts.get_values("state"))
            model = self.forward_model
        else:
            dstate = self.delta(rollouts.get_values("state"))
            model = self.passive_model
        dnstate = self.delta(self.get_targets(rollouts))
        for i in range(int(np.ceil(rollouts.filled / 500))): # run 10k at a time
            dstate_part = dstate[i*500:(i+1)*500].cuda() if self.iscuda else dstate[i*500:(i+1)*500]
            dnstate_part = dnstate[i*500:(i+1)*500].cuda() if self.iscuda else dnstate[i*500:(i+1)*500]
            pred_error.append(pytorch_model.unwrap(self.get_error(model(dstate_part)[0], dnstate_part)) * pytorch_model.unwrap(done_flags[i*500:(i+1)*500]))
            # passive_error.append(self.dist(*self.passive_model(dstate[i*10000:(i+1)*10000])).log_prob(nf(dnstate[i*10000:(i+1)*10000])))
        return np.concatenate(pred_error, axis=0)

    def _get_target_mag(self, rollouts):
        targets = self.get_targets(rollouts)
        for i in range(int(np.ceil(rollouts.filled / 500))): # run 10k at a time
            l1mag.append(pytorch_model.unwrap(targets[i*500:(i+1)*500].norm(p=1, dim=1)))
        return np.concatenate(l1mag, axis=0)


    def get_interaction_vals(self, rollouts):
        interaction = []
        for i in range(int(np.ceil(rollouts.filled / 500))):
            inputs = rollouts.get_values("state")[i*500:(i+1)*500]
            ints = self.interaction_model(self.gamma(inputs))
            interaction.append(pytorch_model.unwrap(ints))
        return np.concatenate(interaction, axis=0)

    def _get_weights(self, err=None, ratio_lambda=2, passive_error_cutoff=2, passive_error_upper=10, weights=None, local=0):
        if err is not None:
            weights = err.squeeze()
            weights[weights<=passive_error_cutoff] = 0
            weights[weights>passive_error_upper] = 0
            weights[weights>passive_error_cutoff] = 1
        weights = pytorch_model.unwrap(weights)
        if local:
            locality = np.array([.4 for i in range(int(local))])
            locality[int(local // 2)] = 1 # set the midpoint to be 1
            weights = np.convolve(weights, locality, 'same') 
        total_live = np.sum(weights)
        total_dead = np.sum((weights + 1)) - np.sum(weights) * 2
        live_factor = total_dead / total_live * ratio_lambda
        use_weights = (weights * live_factor) + 1
        use_weights = use_weights / np.sum(use_weights)
        # use_weights = pytorch_model.unwrap(use_weights)
        # print(weights + 1, local)
        print("live, dead, factor", total_live, total_dead, live_factor)
        return weights, use_weights, total_live, total_dead, ratio_lambda

    
    def get_binaries(self, rollouts):
        bins = []
        rv = self.output_normalization_function.reverse
        fe, pe = list(), list()
        for i in range(int(np.ceil(rollouts.filled / 500))):
            inputs = rollouts.get_values("state")[i*500:(i+1)*500].cuda() if self.iscuda else rollouts.get_values("state")[i*500:(i+1)*500]
            targets = self.get_targets(rollouts)[i*500:(i+1)*500].cuda() if self.iscuda else self.get_targets(rollouts)[i*500:(i+1)*500]
            prediction_params = self.forward_model(self.gamma(inputs))
            interaction_likelihood = self.interaction_model(self.gamma(inputs))
            passive_prediction_params = self.passive_model(self.delta(inputs))
            target = self.output_normalization_function(self.delta(targets))
            passive_loss = - self.dist(*passive_prediction_params).log_prob(target)
            forward_error = - self.dist(*prediction_params).log_prob(target)
            if self.multi_instanced: passive_loss = self.split_instances(passive_loss).sum(dim=2)
            else: passive_loss = passive_loss.sum(dim=1).unsqueeze(1)
            if self.multi_instanced: forward_error = self.split_instances(forward_error).sum(dim=2)
            else: forward_error = forward_error.sum(dim=1).unsqueeze(1)
            interaction_binaries, potential = self.compute_interaction(forward_error, passive_loss, rv(target))
            # interaction_binaries, potential = self.compute_interaction(prediction_params[0].clone().detach(), passive_prediction_params[0].clone().detach(), rv(target))
            bins.append(pytorch_model.unwrap(interaction_binaries))
            fe.append(pytorch_model.unwrap(forward_error))
            pe.append(pytorch_model.unwrap(passive_loss))
        return np.concatenate(bins, axis=0), np.concatenate(fe, axis=0), np.concatenate(pe, axis=0)

    def get_targets(self, rollouts):
        # the target is whether we predict the state diff or the next state
        if self.predict_dynamics:
            targets = rollouts.get_values('state_diff')
        else:
            targets = rollouts.get_values('next_state')
        return targets

    def run_optimizer(self, train_args, optimizer, model, loss):
        optimizer.zero_grad()
        (loss.mean()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    def split_instances(self, delta_state, obj_dim=-1):
        # split up a state or batch of states into instances
        if obj_dim < 0:
            obj_dim = self.object_dim
        nobj = delta_state.shape[-1] // obj_dim
        if len(delta_state.shape) == 1:
            delta_state = delta_state.reshape(nobj, obj_dim)
        elif len(delta_state.shape) == 2:
            delta_state = delta_state.reshape(-1, nobj, obj_dim)
        return delta_state

    def flat_instances(self, delta_state):
        # change an instanced state into a flat state
        if len(delta_state.shape) == 2:
            delta_state = delta_state.flatten()
        elif len(delta_state.shape) == 3:
            batch_size = delta_state.shape[0]
            delta_state = delta_state.reshape(batch_size, delta_state.shape[1] * delta_state.shape[2])
        return delta_state

    def _train_combined(self, rollouts, test_rollout, train_args, batchvals, 
        trace, weights, use_weights, passive_error_all, interaction_schedule, ratio_lambda,
        active_optimizer, passive_optimizer, interaction_optimizer,
        idxes_sets=None, keep_outputs=False):
        outputs = list()
        inter_loss = nn.BCELoss()
        weight_lambda = ratio_lambda
        trw = trace
        if train_args.interaction_iters > 0:
            trw = torch.max(trace, dim=1)[0].squeeze() if self.multi_instanced else trace
        for i in range(train_args.num_iters):
            # passive failure weights
            if idxes_sets is None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=use_weights, existing=batchvals)
            else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
            # for debugging purposes only REMOVE:
            if train_args.interaction_iters > 0:
                true_binaries = trace[idxes].unsqueeze(1)
            if train_args.passive_weighting:
                pe = passive_error_all[idxes]
            # REMOVE above
            # for k, j in enumerate(idxes):
            #     print(trace[j], self.gamma(batchvals.values.state[k]), self.delta(batchvals.values.state_diff[k]))

            prediction_params = self.forward_model(self.gamma(batchvals.values.state))
            if self.multi_instanced: interaction_likelihood = self.interaction_model.instance_labels(self.gamma(batchvals.values.state))
            else: interaction_likelihood = self.interaction_model(self.gamma(batchvals.values.state))
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            # passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            target = self.output_normalization_function(self.delta(self.get_targets(batchvals)))
            # break up by instances to multiply with interaction_likelihood
            if self.multi_instanced:
                # handle passive
                passive_error = - self.dist(*passive_prediction_params).log_prob(target)
                passive_error = self.split_instances(passive_error).sum(dim=2)
                # handle active
                pmu, pvar, ptarget = self.split_instances(prediction_params[0]), self.split_instances(prediction_params[1]), self.split_instances(target)
                forward_error = - self.dist(pmu, pvar).log_prob(ptarget).squeeze().sum(dim=2)
            else:
                forward_error = - self.dist(*prediction_params).log_prob(target).sum(dim=1).unsqueeze(1)
                passive_error = - self.dist(*passive_prediction_params).log_prob(target).sum(dim=1).unsqueeze(1)

            # this is mislabeled, it is actually forward l1
            forward_l2 = (prediction_params[0] - target).abs().mean(dim=1).unsqueeze(1)
            # print(forward_error.shape, interaction_likelihood.shape)
            forward_loss = forward_error * interaction_likelihood.clone().detach() # detach so the interaction is trained only through discrimination 
            passive_loss = passive_error
            # might consider scaling passive error
            # print(prediction_params[0].shape, target.shape, forward_error.shape, interaction_likelihood.shape)
            active_diff = self.split_instances((prediction_params[0] - target)) if self.multi_instanced else (prediction_params[0] - target)
            if self.multi_instanced:
                broadcast_il = torch.stack([interaction_likelihood.clone().detach() for _ in range(active_diff.shape[-1])], dim=2)
                active_l2 = active_diff * broadcast_il
            else:
                active_l2 = (active_diff * interaction_likelihood).norm(dim=1, p=1).clone().detach()
            passive_l2 = (passive_prediction_params[0] - target).norm(dim=1, p=1)

            # append outputs
            if keep_outputs: outputs.append((prediction_params, passive_prediction_params, interaction_likelihood))

            # assign done flags
            done_flags = 1-batchvals.values.done

            # version with binarized loss
            # interaction_loss = torch.zeros((1,))
            # UNCOMMENT WHEN ACTUALLY RUNNING
            if train_args.interaction_iters <= 0:
                # interaction_binaries = self.compute_interaction(prediction_params[0].clone().detach(), passive_prediction_params[0].clone().detach(), self.rv(target))
                interaction_binaries, potential = self.compute_interaction(forward_error, passive_error, self.rv(target))
                # print(interaction_binaries.shape, forward_error.shape, passive_error.shape)
                interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
                # forward_bin_loss = forward_error * interaction_binaries
                # forward_max_loss = forward_error * torch.max(torch.cat([interaction_binaries, interaction_likelihood.clone(), potential.clone()], dim=1).detach(), dim = 1)[0]
                forward_max_loss = forward_error * torch.max(torch.cat([interaction_binaries, interaction_likelihood.clone()], dim=1).detach(), dim = 1)[0]
                self.run_optimizer(train_args, interaction_optimizer, self.interaction_model, interaction_loss) # no done flags because it is possible that the interaction here is valid
            else:
                # in theory, this works even in multiinstanced settings
                interaction_binaries, potential = self.compute_interaction(forward_error, passive_error, self.rv(target))
                interaction_loss = inter_loss(interaction_likelihood, interaction_binaries.detach())
                forward_max_loss = forward_loss
            # MIGHT require no-grad to step passive_error correctly
            # loss = (passive_error + forward_loss * interaction_schedule(i) + forward_error * (1-interaction_schedule(i))).sum() + interaction_loss.sum()
            loss = (forward_max_loss * interaction_schedule(i) + forward_error * (1-interaction_schedule(i))) * done_flags
            # self.run_optimizer(train_args, passive_optimizer, self.passive_model, passive_loss * done_flags)
            self.run_optimizer(train_args, active_optimizer, self.forward_model, loss)

            # PASSIVE OPTIMIZATION SEPARATE NO WEIGHTS CODE:
            # MIGHT 
            # pidxes, pbatchvals = rollouts.get_batch(train_args.batch_size, existing=pbatchvals)
            # passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            # ptarget = self.network_args.output_normalization_function(self.delta(self.get_targets(pbatchvals)))
            # passive_error = - self.dist(*passive_prediction_params).log_prob(target)
            # PASSIVE IGNORES ACTIVE SUCCESS
            # self.run_optimizer(train_args, self.passive_optimizer, self.passive_model, passive_error)
            # passive_error = passive_error * (1-interaction_binaries)
            # self.optimizer.zero_grad()
            # (loss).backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), train_args.max_grad_norm)
            # self.optimizer.step()
            

            if i % train_args.log_interval == 0:
                # print(i, ": pl: ", pytorch_model.unwrap(passive_error.norm()), " fl: ", pytorch_model.unwrap(forward_loss.norm()), 
                #     " il: ", pytorch_model.unwrap(interaction_loss.norm()), " dl: ", pytorch_model.unwrap(interaction_diversity_loss.norm()))
                print(i, i % (train_args.log_interval * 10), (train_args.log_interval * 10))
                # error
                if i % (train_args.log_interval * 10) == 0:
                    print("assessing full train")
                    self.assess_losses(rollouts)
                    # self.assess_error(rollouts, passive_error_cutoff=train_args.passive_error_cutoff)
                    print("assessing test rollouts")
                    self.assess_losses(test_rollout)
                    # self.assess_error(test_rollout, passive_error_cutoff=train_args.passive_error_cutoff)
                if self.multi_instanced: 
                    split_target = self.split_instances(target)
                    inp = self.gamma(batchvals.values.state)
                    inp = self.split_instances(inp)
                    active = self.split_instances(self.rv(prediction_params[0])).squeeze()
                    activeunnorm = self.split_instances(prediction_params[0]).squeeze()
                    activevar = self.split_instances(prediction_params[1]).squeeze()
                    print(self.rv(split_target).shape, self.rv(target).shape, self.rv(prediction_params[0]).shape)
                    adiff = self.split_instances(self.rv(target) - self.rv(prediction_params[0]))
                    pdiff = self.split_instances(self.rv(target) - self.rv(passive_prediction_params[0]))

                    obj_indices = pytorch_model.unwrap((trace[idxes] > 0).nonzero())
                    all_indices = []
                    for ti in obj_indices:
                        # print(self.rv(passive_prediction_params[0])[ti[0]])
                        all_indices.append(np.array([ti[0], ti[1]-2]))
                        all_indices.append(np.array([ti[0], ti[1]-1]))
                        all_indices.append(pytorch_model.unwrap(ti))
                        if ti[1]+1 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+1]))
                        if ti[1]+2 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+2]))
                        for _ in range(3):
                            all_indices.append(np.array([ti[0], np.random.randint(interaction_likelihood[ti[0]].shape[0])]))
                    for _ in range(20):
                        all_indices.append(np.array([np.random.randint(train_args.batch_size), np.random.randint(interaction_likelihood.shape[1])]))

                    obj_indices = np.array(all_indices)
                    print("iteration: ", i)
                    # print(trw[idxes], trace[idxes].sum(dim=1), idxes)
                    print(forward_loss[obj_indices[0][0]], interaction_likelihood[obj_indices[0][0]])
                    for a in obj_indices[:20]:
                        print("idx", a[0], a[1])
                        # print("has", trace[idxes[a[0]].sum())
                        print("trace: ", pytorch_model.unwrap(trace[idxes[a[0]], a[1]]),
                        "inter: ", pytorch_model.unwrap(interaction_likelihood[a[0], a[1]]),
                        "inp: ", pytorch_model.unwrap(inp[a[0], a[1]]),
                        "target: ", pytorch_model.unwrap(split_target[a[0], a[1]]),
                        "active: ", pytorch_model.unwrap(active[a[0], a[1]]),
                        "aun: ", pytorch_model.unwrap(activeunnorm[a[0], a[1]]),
                        "avar: ", pytorch_model.unwrap(activevar[a[0], a[1]]),
                        "adiff: ", pytorch_model.unwrap(adiff[a[0], a[1]]),
                        "pdiff: ", pytorch_model.unwrap(pdiff[a[0], a[1]]),
                        "floss: ", pytorch_model.unwrap(forward_loss[a[0], a[1]]))
                else:
                    likelihood_binaries = (interaction_likelihood > .5).float()
                    if train_args.interaction_iters > 0:
                        test_binaries = (interaction_binaries.squeeze() + true_binaries.squeeze() + likelihood_binaries.squeeze()).long().squeeze()
                        test_idxes = torch.nonzero(test_binaries)
                        # print(test_idxes.shape, test_binaries.shape)
                        # print([interaction_likelihood.shape, likelihood_binaries.shape, interaction_binaries.shape, true_binaries.shape, forward_error.shape, passive_error.shape])
                        intbint = torch.cat([interaction_likelihood, likelihood_binaries, interaction_binaries, true_binaries, forward_error, passive_error], dim=1).squeeze()
                    else:
                        passive_binaries = (passive_error > train_args.passive_error_cutoff).float()
                        test_binaries = (interaction_binaries.squeeze() + likelihood_binaries.squeeze() + passive_binaries.squeeze()).long().squeeze()
                        test_idxes = torch.nonzero(test_binaries)
                        intbint = torch.cat([interaction_likelihood, likelihood_binaries, interaction_binaries, forward_error, passive_error], dim=1).squeeze()
                    test_binaries[test_binaries > 1] = 1
                    inp = self.gamma(batchvals.values.state)
                    # print(inp.shape, batchvals.values.state.shape, prediction_params[0].shape, test_idxes, test_binaries, likelihood_binaries, interaction_binaries)
                    test_idxes = torch.tensor([i[0] for i in test_idxes if i < 30]).long()
                    print(test_idxes)
                    print("iteration: ", i,
                        "input", inp[:30].squeeze(),
                        # "\ninteraction", interaction_likelihood,
                        # "\nbinaries", interaction_binaries[:30],
                        # "\ntrue binaries", true_binaries,
                        "\nintbint", intbint[:30],
                        # "\naoutput", self.rv(prediction_params[0]),
                        # "\navariance", self.rv(prediction_params[1]),
                        # "\npoutput", self.rv(passive_prediction_params[0]),
                        # "\npvariance", self.rv(passive_prediction_params[1]),
                        # self.delta(batchvals.values.next_state[0]), 
                        # self.gamma(batchvals.values.state[0]),
                        "\ntarget: ", self.rv(target)[:30][test_idxes].squeeze(),
                        "\nactive", self.rv(prediction_params[0])[:30][test_idxes].squeeze(),
                        # "\ntadiff", (self.rv(target) - self.rv(prediction_params[0])) * test_binaries,
                        # "\ntpdiff", (self.rv(target) - self.rv(passive_prediction_params[0])) * test_binaries,
                        "\ntadiff", (self.rv(target) - self.rv(prediction_params[0]))[:30][test_idxes].squeeze(),
                        "\ntpdiff", (self.rv(target) - self.rv(passive_prediction_params[0]))[:30][test_idxes].squeeze(),
                        "\nal2: ", active_l2.mean(dim=0),
                        "\npl2: ", passive_l2.mean(dim=0),)
                print(
                    "\nae: ", (forward_error * done_flags).mean(dim=0),
                    "\nal: ", (forward_loss * done_flags).sum(dim=0) / interaction_likelihood.sum(),
                    "\npl: ", (passive_error * done_flags).mean(dim=0),
                    "\ninter_lambda: ", interaction_schedule(i)
                    )
                # REWEIGHTING CODE
                if train_args.interaction_iters > 0:
                    weight_lambda = train_args.interaction_weight * np.exp(-i/train_args.num_iters * 5)
                    print(weight_lambda)
                    _, use_weights, live, dead, ratio_lambda = self._get_weights(ratio_lambda=weight_lambda, weights=trw, local=train_args.interaction_local)
                elif train_args.passive_weighting:
                    weight_lambda = train_args.passive_weighting * np.exp(-i/train_args.num_iters * 3)
                    print("weight lambda", weight_lambda)
                    _, use_weights, live, dead, ratio_lambda = self._get_weights(passive_error_all, ratio_lambda=weight_lambda, passive_error_cutoff=train_args.passive_error_cutoff, local=train_args.interaction_local)
        return outputs
        # torch.save(self.forward_model, "data/active_model.pt")
        # torch.save(self.interaction_model, "data/train_int.pt")


    def _train_interaction(self, rollouts, train_args, batchvals, trace, trace_targets, interaction_optimizer, idxes_sets=None, keep_outputs=False):
        outputs = list()
        inter_loss = nn.BCELoss()
        if train_args.interaction_iters > 0:
            # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
            trw = torch.max(trace, dim=1)[0].squeeze() if self.multi_instanced else trace
            print(trw.sum())
            # weights the values
            _, weights, live, dead, ratio_lambda = self._get_weights(ratio_lambda=train_args.interaction_weight, weights=trw, local=train_args.interaction_local)
            for i in range(train_args.interaction_iters):
                # get the input and target values
                if idxes_sets is not None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
                else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=pytorch_model.unwrap(weights), existing=batchvals)
                target = trace_targets[idxes]# if self.multi_instanced else trace[idxes]
                
                # get the network outputs
                # multi-instanced will have shape [batch, num_instances]
                if self.multi_instanced: interaction_likelihood = self.interaction_model.instance_labels(self.gamma(batchvals.values.state))
                else: interaction_likelihood = self.interaction_model(self.gamma(batchvals.values.state))
                
                # keep outputs only for unit testing
                if keep_outputs: outputs.append(interaction_likelihood)

                # compute loss
                trace_loss = inter_loss(interaction_likelihood.squeeze(), target)
                self.run_optimizer(train_args, interaction_optimizer, self.interaction_model, trace_loss)
                
                if i % train_args.log_interval == 0:
                    obj_indices = list(range(30))
                    inp = self.gamma(batchvals.values.state)
                    if self.multi_instanced: 
                        # print out only the interaction instances which are true
                        # TODO: a ton of logging code that I no longer understand
                        # target = target
                        inp = self.split_instances(inp, self.object_dim)
                        obj_indices = pytorch_model.unwrap((trace[idxes] > 0).nonzero())
                        objective = self.delta(self.get_targets(batchvals))
                        all_indices = []
                        for ti in obj_indices:
                            all_indices.append(np.array([ti[0], ti[1]-2]))
                            all_indices.append(np.array([ti[0], ti[1]-1]))
                            all_indices.append(pytorch_model.unwrap(ti))
                            if ti[1]+1 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+1]))
                            if ti[1]+2 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+2]))
                            for i in range(3):
                                all_indices.append(np.array([ti[0], np.random.randint(interaction_likelihood[ti[0]].shape[0])]))
                        for _ in range(20):
                            all_indices.append(np.array([np.random.randint(train_args.batch_size), np.random.randint(interaction_likelihood.shape[1])]))
                        obj_indices = np.array(all_indices)
                        # print (obj_indices)
                        print(i, ": tl: ", trace_loss)
                        # print(target.shape)
                        print(target[obj_indices[0][0]], interaction_likelihood[obj_indices[0][0]])
                        for a in obj_indices:
                            print("state:", pytorch_model.unwrap(inp)[a[0], a[1]],
                            "inter: ", interaction_likelihood[a[0], a[1]],
                            "target: ", target[a[0], a[1]])
                    else:
                        print(i, ": tl: ", trace_loss)
                        print("\nstate:", pytorch_model.unwrap(inp)[obj_indices],
                            "\ntraining: ", pytorch_model.unwrap(interaction_likelihood[obj_indices]),
                            "\ntarget: ", pytorch_model.unwrap(target[obj_indices]))
                    weight_lambda = train_args.interaction_weight * np.exp(-i/(train_args.interaction_iters) * 5)
                    _, weights, live, dead, ratio_lambda = self._get_weights(ratio_lambda=weight_lambda, weights=trw, local=train_args.interaction_local)
            self.interaction_model.needs_grad=False # no gradient will pass through the interaction model
            # if train_args.save_intermediate:
            #     torch.save(self.interaction_model, "data/temp/interaction_model.pt")
        # if train_args.load_intermediate:
        #     self.interaction_model = torch.load("data/temp/interaction_model.pt")
        return outputs

    def _train_passive(self, rollouts, train_args, batchvals, active_optimizer, passive_optimizer, idxes_sets=None, keep_outputs=False):
        outputs = list()
        for i in range(train_args.pretrain_iters):
            # get input-output values
            if idxes_sets is not None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
            else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals)
            target = self.nf(self.delta(self.get_targets(batchvals)))

            # compute network values
            passive_prediction_params = self.passive_model(self.delta(batchvals.values.state))
            
            # compute losses
            done_flags = 1-batchvals.values.done
            passive_loss = - self.dist(*passive_prediction_params).log_prob(target)
            if self.multi_instanced: passive_loss = self.split_instances(passive_loss).sum(dim=2) * done_flags
            else: passive_loss = passive_loss.sum(dim=1).unsqueeze(1) * done_flags

            # only for unit testing
            if keep_outputs: outputs.append((prediction_params, passive_prediction_params))

            # optimize active and passive models
            if not train_args.no_pretrain_active:
                prediction_params = self.forward_model(self.gamma(batchvals.values.state))
                active_loss = - self.dist(*prediction_params).log_prob(target)
                if self.multi_instanced: active_loss = self.split_instances(active_loss).sum(dim=2) * done_flags
                else: active_loss = - self.dist(*prediction_params).log_prob(target).sum(dim=1).unsqueeze(1) * done_flags
                self.run_optimizer(train_args, active_optimizer, self.forward_model, active_loss)
            self.run_optimizer(train_args, passive_optimizer, self.passive_model, passive_loss)
            if i % train_args.log_interval == 0:
                if self.multi_instanced:
                    for j in range(2):
                        print(
                            # self.network_args.normalization_function.reverse(passive_prediction_params[0][0]),
                            # self.network_args.normalization_function.reverse(passive_prediction_params[1][0]), 
                            "input", pytorch_model.unwrap(self.gamma(batchvals.values.state)[j]),
                            "\npinput", pytorch_model.unwrap(self.delta(batchvals.values.state[j])),
                            "ninput", pytorch_model.unwrap(self.inf(self.gamma(batchvals.values.state)[j])),
                            "\npninput", pytorch_model.unwrap(self.dnf(self.delta(batchvals.values.state[j]))),
                            # "\naoutput", pytorch_model.unwrap(self.rv(prediction_params[0])[j]),
                            # "\navariance", self.rv(prediction_params[1]),
                            "\npoutput", pytorch_model.unwrap(self.rv(passive_prediction_params[0])[j]),
                            "\npn_output", pytorch_model.unwrap(passive_prediction_params[0][j]),
                            "\npvariance", pytorch_model.unwrap(passive_prediction_params[1][j]),
                            # self.delta(batchvals.values.next_state[0]), 
                            # self.gamma(batchvals.values.state[0]),
                            "\ntarget: ", pytorch_model.unwrap(self.rv(target)[j]),
                            # "\nal: ", active_loss,
                            # "\npl: ", passive_loss
                            )

                else:
                    for j in range(train_args.batch_size):
                        if target[j].abs().sum() > 0:
                            print(
                                # self.network_args.normalization_function.reverse(passive_prediction_params[0][0]),
                                # self.network_args.normalization_function.reverse(passive_prediction_params[1][0]), 
                                "input", pytorch_model.unwrap(self.gamma(batchvals.values.state)[j]),
                                "\npinput", pytorch_model.unwrap(self.delta(batchvals.values.state[j])),
                                "ninput", pytorch_model.unwrap(self.inf(self.gamma(batchvals.values.state)[j])),
                                "\npninput", pytorch_model.unwrap(self.dnf(self.delta(batchvals.values.state[j]))),
                                # "\naoutput", pytorch_model.unwrap(self.rv(prediction_params[0])[j]),
                                # "\navariance", self.rv(prediction_params[1]),
                                "\npoutput", pytorch_model.unwrap(self.rv(passive_prediction_params[0])[j]),
                                "\npn_output", pytorch_model.unwrap(passive_prediction_params[0][j]),
                                "\npvariance", pytorch_model.unwrap(passive_prediction_params[1][j]),
                                # self.delta(batchvals.values.next_state[0]), 
                                # self.gamma(batchvals.values.state[0]),
                                "\ntarget: ", pytorch_model.unwrap(self.rv(target)[j]),
                                # "\nal: ", active_loss,
                                # "\npl: ", passive_loss
                                )
                print("Iters", i, ", pl: ", passive_loss.mean().detach().cpu().numpy(),
                    # ", al: ", active_loss.mean().detach().cpu().numpy()
                    )
        return outputs

    def train(self, rollouts, test_rollout, train_args, control, controllers, target_name):
        '''
        Train the passive model, interaction model and active model
        @param control is the name of the object that we have control over
        @param controllers is the list of corresponding controllable feature selectors for object @param control 
        @param target_name is the name of the object that we want to control using @param control
        '''
        # define names
        self.control_feature = control # the name of the controllable object
        self.controllers = controllers
        control_name = self.control_feature
        self.target_name = target_name
        self.name = control + "->" + target_name
        self.predict_dynamics = train_args.predict_dynamics
        
        # initialize the optimizers
        active_optimizer = optim.Adam(self.forward_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        passive_optimizer = optim.Adam(self.passive_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        interaction_optimizer = optim.Adam(self.interaction_model.parameters(), train_args.critic_lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        
        # compute maximum and minimum of target values
        minmax = self.delta(rollouts.get_values('state'))
        self.control_min = np.amin(pytorch_model.unwrap(minmax), axis=1)
        self.control_max = np.amax(pytorch_model.unwrap(minmax), axis=1)

        # Computes the target normalization value, get normalization values
        if train_args.multi_instanced:
            output_norm_fun = PointwiseNorm(object_dim = self.object_dim)
        else:
            output_norm_fun = InterInputNorm()
        if len(train_args.hardcode_norm) > 0:
            if self.predict_dynamics:
                target_norm = hardcode_norm(train_args.hardcode_norm[0], ["Relative" + target_name])
                output_norm_fun.assign_mean_var(*target_norm)
            else:
                target_norm = hardcode_norm(train_args.hardcode_norm[0], [target_name])
                output_norm_fun.assign_mean_var(*target_norm)
        else:
            output_norm_fun.compute_input_norm(self.delta(self.get_targets(rollouts)))
        self.output_normalization_function = output_norm_fun
        self.inf = self.normalization_function
        self.irv = self.normalization_function.reverse
        self.dnf = self.delta_normalization_function
        self.drv = self.delta_normalization_function.reverse
        self.nf = self.output_normalization_function # temporarily to save length
        self.rv = self.output_normalization_function.reverse # same as above
        if train_args.cuda:
            self.inf.cuda()
            self.dnf.cuda()
            self.nf.cuda()

        # pre-initialize batches because it accelerates time
        batchvals = type(rollouts)(train_args.batch_size, rollouts.shapes)
        pbatchvals = type(rollouts)(train_args.batch_size, rollouts.shapes)

        self._train_passive(rollouts, train_args, batchvals, active_optimizer, passive_optimizer)

        if train_args.save_intermediate:
            torch.save(self.passive_model, "data/temp/passive_model.pt")
            # torch.save(self.forward_model, "data/temp/active_model.pt")

        # generate the trace
        trace = None
        trace_targets = None
        if train_args.interaction_iters > 0:
            if train_args.env != "RoboPushing":
                if train_args.load_intermediate: trace = load_from_pickle("data/temp/trace.pkl").cpu().cuda()
                else: trace = self.generate_interaction_trace(rollouts, [control_name], [target_name])
                trace_targets = self._adjust_interaction_trace(trace)
                if train_args.save_intermediate:
                    save_to_pickle("data/temp/trace.pkl", trace)

        # train the interaction model with true interaction "trace" values
        self._train_interaction(rollouts, train_args, batchvals, trace, trace_targets, interaction_optimizer)

        if train_args.save_intermediate:
            torch.save(self.interaction_model, "data/temp/interaction_model.pt")

        if train_args.load_intermediate:
            if train_args.interaction_iters > 0:
                trace = load_from_pickle("data/temp/trace.pkl").cpu().cuda()
                self.interaction_model = torch.load("data/temp/interaction_model.pt")
                print("loading interaction model")
            self.passive_model = torch.load("data/temp/passive_model.pt")
        #     self.forward_model = torch.load("data/temp/active_model.pt")
        #     self.passive_model.cpu()
        #     self.passive_model.cuda()
        #     self.forward_model.cpu()
        #     self.forward_model.cuda()

        # initialize the interaction schedule, which is degree to which the interactions affect the forward loss
        if train_args.epsilon_schedule <= 0: interaction_schedule = lambda i: 1
        else: interaction_schedule = lambda i: np.power(0.5, (i/train_args.epsilon_schedule))

        # sampling weights, either wit hthe passive error or if we can upweight the true interactions
        if train_args.passive_weighting > 0:
            passive_error_all = self.get_prediction_error(rollouts)
            print(passive_error_all[:100])
            print(passive_error_all[100:200])
            print(passive_error_all[200:300])
            # passive_error_all = self.interaction_model(self.gamma(rollouts.get_values("state")))
            # passive_error = pytorch_model.wrap(trace)
            weights, use_weights, total_live, total_dead, ratio_lambda = self._get_weights(passive_error_all, ratio_lambda = train_args.passive_weighting, passive_error_cutoff=train_args.passive_error_cutoff, local=train_args.interaction_local)
            print(use_weights[:100])
            print(use_weights[100:200])
            print(use_weights[200:300])
        elif train_args.interaction_iters > 0:
            print("compute values")
            passive_error_all = trace.clone()
            trw = torch.max(trace, dim=1)[0].squeeze() if self.multi_instanced else trace
            print(trw.sum())
            weights, use_weights, total_live, total_dead, ratio_lambda = self._get_weights(ratio_lambda=train_args.interaction_weight, weights=trw, local=train_args.interaction_local)
            use_weights =  copy.deepcopy(use_weights)
            print(use_weights.shape)
        elif train_args.change_weighting > 0:
            target_mag = self._get_target_mag(rollouts)
            weights, use_weights, total_live, total_dead, ratio_lambda = self._get_weights(target_mag, ratio_lambda = train_args.change_weighting, passive_error_cutoff=train_args.passive_error_cutoff)

        else: # no weighting o nthe samples
            passive_error_all = torch.ones(rollouts.filled)
            weights, use_weights = np.ones(rollouts.filled) / rollouts.filled, np.ones(rollouts.filled) / rollouts.filled
            print(rollouts.filled, weights.shape, weights)
            total_live, total_dead = 0, 0
            ratio_lambda = 1

        # handling boosting the passive operator to work with upweighted states
        # boosted_passive_operator = copy.deepcopy(self.passive_model)
        # true_passive = self.passive_model
        # self.passive_model = boosted_passive_operator
        passive_optimizer = optim.Adam(self.passive_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
        self._train_combined(rollouts, test_rollout, train_args, batchvals, 
            trace, weights, use_weights, passive_error_all, interaction_schedule, ratio_lambda,
            active_optimizer, passive_optimizer, interaction_optimizer)        # if args.save_intermediate:
        self.save(train_args.save_dir)
        if train_args.interaction_iters > 0:
            self.compute_interaction_stats(rollouts, trace = trace, passive_error_cutoff=train_args.passive_error_cutoff)
        del self.nf, self.rv


    def compute_interaction_stats(self, rollouts, trace=None, passive_error_cutoff=2):
        ints = self.get_interaction_vals(rollouts)
        bins, fe, pe = self.get_binaries(rollouts)
        if trace is None:
            trace = self.generate_interaction_trace(rollouts, [self.control_feature], [self.target_name])
        if self.multi_instanced: trace = torch.max(trace, dim=1)[0].squeeze()
        trace = pytorch_model.unwrap(trace)
        passive_error = self.get_prediction_error(rollouts)
        weights, use_weights, total_live, total_dead, ratio_lambda = self._get_weights(passive_error, ratio_lambda=1, passive_error_cutoff=passive_error_cutoff)     
        print(ints.shape, bins.shape, trace.shape, fe.shape, pe.shape)
        pints, ptrace = np.zeros(ints.shape), np.zeros(trace.shape)
        pints[ints > .7] = 1
        ptrace[trace > 0] = 1
        print_weights = (weights + pints.squeeze() + ptrace).squeeze()
        print_weights[print_weights > 1] = 1

        print(ints.shape, bins.shape, np.expand_dims(trace, 1).shape, fe.shape, pe.shape)
        comb = np.concatenate([ints, bins, np.expand_dims(trace, 1), fe, pe], axis=1)
        
        bin_error = bins.squeeze()-trace.squeeze()
        bin_false_positives = np.sum(bin_error[bin_error > 0])
        bin_false_negatives = np.sum(np.abs(bin_error[bin_error < 0]))

        int_bin = ints.copy()
        int_bin[int_bin >= .5] = 1
        int_bin[int_bin < .5] = 0
        int_error = int_bin.squeeze() - trace.squeeze()
        int_false_positives = np.sum(int_error[int_error > 0])
        int_false_negatives = np.sum(np.abs(int_error[int_error < 0]))

        comb_error = bins.squeeze() + int_bin.squeeze()
        comb_error[comb_error > 1] = 1
        comb_error = comb_error - trace.squeeze()
        comb_false_positives = np.sum(comb_error[comb_error > 0])
        comb_false_negatives = np.sum(np.abs(comb_error[comb_error < 0]))

        print("bin fp, fn", bin_false_positives, bin_false_negatives)
        print("int fp, fn", int_false_positives, int_false_negatives)
        print("com fp, fn", comb_false_positives, comb_false_negatives)
        print("total, tp", trace.shape[0], np.sum(trace))
        del bins
        del fe
        del pe
        del pints
        del ptrace
        del comb
        del bin_error
        del bin_false_positives
        del bin_false_negatives
        del comb_error


    def assess_losses(self, test_rollout):
        forward_loss, passive_loss = list(), list()
        for i in range(int(np.ceil(test_rollout.filled / 300))):
            state, target = test_rollout.get_values("state")[i*300:(i+1)*300], self.get_targets(test_rollout)[i*300:(i+1)*300]
            prediction_params = self.forward_model(self.gamma(state))
            interaction_likelihood = self.interaction_model(self.gamma(state))
            passive_prediction_params = self.passive_model(self.delta(state))
            passive_loss.append(pytorch_model.unwrap(- self.dist(*passive_prediction_params).log_prob(self.delta(target)) * interaction_likelihood))
            forward_loss.append(pytorch_model.unwrap(- self.dist(*prediction_params).log_prob(self.delta(target)) * interaction_likelihood))
        pl, fl = np.concatenate(passive_loss, axis=0).mean(), np.concatenate(forward_loss, axis=0).mean()
        print("passive, forward losses", pl, fl)
        return pl, fl

    def assess_error(self, test_rollout, passive_error_cutoff=2):
        print("assessing_error", test_rollout.filled)
        if self.env_name != 'RobosuitePushing':
            self.compute_interaction_stats(test_rollout, passive_error_cutoff=passive_error_cutoff)
        rv = self.output_normalization_function.reverse
        states = test_rollout.get_values("state")
        interaction, forward, passive = list(), list(), list()
        for i in range(int(np.ceil(test_rollout.filled / 300))):
            inter, f, p = self.hypothesize(states[i*300:(i+1)*300])
            interaction.append(pytorch_model.unwrap(inter)), forward.append(pytorch_model.unwrap(f)), passive.append(pytorch_model.unwrap(p))
        interaction, forward, passive = np.concatenate(interaction, axis=0), np.concatenate(forward, axis=0), np.concatenate(passive, axis=0)
        targets = self.get_targets(test_rollout)
        dtarget = self.split_instances(self.delta(targets)) if self.multi_instanced else self.delta(targets)
        axis = 2 if self.multi_instanced else 1
        print(forward.shape, dtarget.shape, interaction.shape)
        inter_bin = interaction.copy()
        inter_bin[interaction >= self.interaction_prediction] = 1
        inter_bin[interaction < self.interaction_prediction] = 0
        sfe = np.linalg.norm(forward - pytorch_model.unwrap(dtarget), ord =1, axis=axis) * interaction.squeeze() # per state forward error
        spe = np.linalg.norm(passive - pytorch_model.unwrap(dtarget), ord =1, axis=axis) * interaction.squeeze() # per state passive error
        # print(self.output_normalization_function.mean, self.output_normalization_function.std)
        cat_ax = 1
        if self.multi_instanced:
            interaction = np.expand_dims(interaction, axis=2)
            cat_ax = 2
        print(forward[:100].shape, (forward - pytorch_model.unwrap(dtarget))[:100].shape, (passive - pytorch_model.unwrap(dtarget))[:100].shape, pytorch_model.unwrap(dtarget)[:100].shape, pytorch_model.unwrap(interaction)[:100].shape)
        print("forward, passive, interaction", np.concatenate([forward[:100], (forward - pytorch_model.unwrap(dtarget))[:100], (passive - pytorch_model.unwrap(dtarget))[:100], pytorch_model.unwrap(dtarget)[:100], pytorch_model.unwrap(interaction)[:100]], axis=cat_ax))
        sfeat = sfe[interaction.squeeze() > .5]
        speat = spe[interaction.squeeze() > .5]
        dtargetat = dtarget[interaction.squeeze() > .5]
        interat = interaction[interaction.squeeze() > .5]
        print([np.expand_dims(sfeat[:100], 1).shape, np.expand_dims(speat[:100], 1).shape, pytorch_model.unwrap(dtargetat)[:100].shape, interat[:100].shape])
        print("forward, passive, inter-normalized", np.concatenate([np.expand_dims(sfeat[:100], 1), np.expand_dims(speat[:100], 1), pytorch_model.unwrap(dtargetat)[:100], interat[:100]], axis=1))
        print("inputs", self.gamma(test_rollout.get_values("state"))[:100])
        print("targets", self.delta(targets)[:100])
        forward_error = np.sum(np.abs(sfe)) / np.sum(interaction)
        passive_error = np.sum(np.abs(spe)) / np.sum(interaction)
        print("comparison", forward_error, passive_error)
        return forward_error, passive_error

    def _wrap_state(self, state, tar_state=None):
        # takes in a state, either a full state (dict, tensor or array), or 
        # state = input state (gamma), tar_state = target state (delta)
        # print(type(state), state.shape[-1], self.environment_model.state_size)
        if type(state) == np.ndarray:
            if state.shape[-1] == self.environment_model.state_size:
                inp_state = pytorch_model.wrap(self.gamma(state), cuda=self.iscuda)
                tar_state = pytorch_model.wrap(self.delta(state), cuda=self.iscuda)
            else:
                inp_state = pytorch_model.wrap(state, cuda=self.iscuda)
                tar_state = pytorch_model.wrap(tar_state, cuda=self.iscuda) if tar_state else None
        elif type(state) == torch.tensor:
            inp_state = self.gamma(state)
            tar_state = self.delta(state)
        else: # assumes that state is a batch or dict
            if (type(state) == Batch or type(state) == dict) and ('factored_state' in state): state = state['factored_state'] # use gamma on the factored state
            inp_state = pytorch_model.wrap(self.gamma(state), cuda=self.iscuda)
            tar_state = pytorch_model.wrap(self.delta(state), cuda=self.iscuda)
        return inp_state, tar_state

    def predict_next_state(self, state):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        inp_state, tar_state = self._wrap_state(state)

        rv = self.output_normalization_function.reverse
        inter = self.interaction_model(inp_state)
        intera = inter.clone()
        intera[inter > self.interaction_prediction] = 1
        intera[inter <= self.interaction_prediction] = 0
        if self.predict_dynamics:
            fpred, ppred = tar_state + rv(self.forward_model(inp_state)[0]), tar_state + rv(self.passive_model(tar_state)[0])
        else:
            fpred, ppred = rv(self.forward_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        if len(state.shape) == 1:
            return (inter, fpred) if pytorch_model.unwrap(inter) > self.interaction_prediction else (inter, ppred)
        else:
            # inter_assign = torch.cat((torch.arange(state.shape[0]).unsqueeze(1), intera), dim=1).long()
            pred = torch.stack((ppred, fpred), dim=1)
            # print(inter_assign.shape, pred.shape)
            intera = pytorch_model.wrap(intera.squeeze().long(), cuda=self.iscuda)
            # print(intera, self.interaction_prediction)
            pred = pred[torch.arange(pred.shape[0]).long(), intera]
        # print(pred, inter, self.predict_dynamics, rv(self.forward_model(inp_state)[0]))
        return inter, pred

    def test_forward(self, states, next_state, interact=True):
        # gives back the difference between the prediction mean and the actual next state for different sampled feature values
        rv = self.output_normalization_function.reverse
        checks = list()
        print(np.ceil(len(states)/2000))
        batch_pred, inters = list(), list()
        # painfully slow when states is large, so an alternative might be to only look at where inter.sum() > 1
        for state in states:
            # print(self.gamma(self.control_feature.sample_feature(state)))
            control = self.controllers
            if type(self.controllers) == list and len(self.controllers) == 1:
                control = self.controllers[0]
            if type(control) == list: # multiple control features
                sampled_feature = sample_multiple(self.controllers, pytorch_model.unwrap(state))
            else:
                sampled_feature = control.sample_feature(state)
            # print(self.gamma(sampled_feature))
            # print(sampled_feature.shape, print(type(self.control_feature)))
            sampled_feature = pytorch_model.wrap(sampled_feature, cuda=self.iscuda)
            inter, pred_states = self.predict_next_state(sampled_feature)
            # if inter.sum() > .7:
            #     # print(inter.shape, pred_states.shape, sampled_feature.shape, inter > 0)
            #     print(inter[inter.squeeze() > 0.7], pred_states[inter.squeeze() > 0.7], self.gamma(sampled_feature[inter.squeeze() > 0.7]), self.control_feature.object())
            # if inter.sum() >= 1:
            #     print('int', pytorch_model.unwrap(inter))
            #     print('sam', pytorch_model.unwrap(self.gamma(sampled_feature)))
            #     print('pred_states', pytorch_model.unwrap(pred_states))
            batch_pred.append(pred_states.cpu().clone().detach()), inters.append(inter.cpu().clone().detach())
            del pred_states
            del inter
        batch_pred, inters = torch.stack(batch_pred, dim=0), torch.stack(inters, dim=0) # batch x samples x state, batch x samples x 1
        next_state_broadcast = pytorch_model.wrap(torch.stack([self.delta(next_state).clone().cpu() for _ in range(batch_pred.size(1))], dim=1)).cpu()
        # compare predictions with the actual next state to make sure there are differences
        print(int(np.ceil(len(states)/2000)), batch_pred, next_state_broadcast)
        state_check = (next_state_broadcast - batch_pred).abs()
        print(state_check[:10])
        # should be able to predict at least one of the next states accurately
        match = state_check.min(dim=1)[0]
        match_keep = match.clone()
        print(match[:10], self.interaction_prediction)
        match_keep[match <= self.interaction_prediction] = 1
        match_keep[match > self.interaction_prediction] = 0
        if interact: # if the interaction value is less, assume there is no difference because the model is flawed
            inters[inters > self.interaction_prediction] = 1
            inters[inters <= self.interaction_prediction] = 0
            checks.append((state_check * match_keep.unsqueeze(1)) * inters) # batch size, num samples, state size
        else:
            checks.append(state_check * match_keep.unsqueeze(1))
        return torch.cat(checks, dim=0)

    def determine_active_set(self, rollouts, use_hardcoded_rng=None, feature_step=1):
        states = rollouts.get_values('state')
        next_states = rollouts.get_values('state')
        targets = self.get_targets(rollouts)
        # create a N x num samples x state size of the nth sample tested for difference on the num samples of assignments of the controllable feature
        # then take the largest difference along the samples
        sample_diffs = torch.max(self.test_forward(states, next_states), dim=1)[0]
        # take the largest difference at any given state
        test_diff = torch.max(sample_diffs, dim=0)[0]
        v = torch.zeros(test_diff.shape)
        # if the largest difference is larger than the active_epsilon, assign it
        print("test_diff", test_diff)
        v[test_diff > self.active_epsilon] = 1
        # collect by instance and determine
        v = self.split_instances(v)
        print(v.shape)
        v = torch.max(v, dim=0)[0]

        print("act set", v, v.sum())
        if v.sum() == 0:
            return None, None

        # create a feature selector to match that
        self.selection_binary = pytorch_model.wrap(v, cuda=self.iscuda)
        self.feature_selector, self.reverse_feature_selector = self.environment_model.get_subset(self.delta, v)

        # create a controllable feature selector for each controllable feature
        self.cfselectors = list()
        for ff in self.feature_selector.flat_features:
            factored = self.environment_model.flat_to_factored(ff)
            print(factored)
            single_selector = FeatureSelector([ff], {factored[0]: factored[1]}, {factored[0]: np.array([factored[1], ff])}, [factored[0]])
            rng = self.determine_range(rollouts, single_selector, use_hardcoded_rng = use_hardcoded_rng)
            print(rng)
            self.cfselectors.append(ControllableFeature(single_selector, rng, feature_step, self))
        self.selection_list = get_selection_list(self.cfselectors)
        self.control_min = [cfs.feature_range[0] for cfs in self.cfselectors]
        self.control_max = [cfs.feature_range[1] for cfs in self.cfselectors]
        return self.feature_selector, self.cfselectors

    def get_active_mask(self):
        return pytorch_model.unwrap(self.selection_binary.clone())

    def determine_range(self, rollouts, active_delta, use_hardcoded_rng):
        # Even if we are predicting the dynamics, we determine the active range with the states
        # TODO: However, using the dynamics to predict possible state range ??
        # if self.predict_dynamics:
        #     state_diffs = rollouts.get_values('state_diff')
        #     return active_delta(state_diffs).min(dim=0)[0], active_delta(state_diffs).max(dim=0)[0]
        # else:
        if use_hardcoded_rng is not None:
            return float(pytorch_model.unwrap(active_delta(use_hardcoded_rng[0]).squeeze())), float(pytorch_model.unwrap(active_delta(use_hardcoded_rng[0]).squeeze()))
        else:
            states = rollouts.get_values('state')
            return float(pytorch_model.unwrap(active_delta(states).min(dim=0)[0].squeeze())), float(pytorch_model.unwrap(active_delta(states).max(dim=0)[0].squeeze()))

    def hypothesize(self, state):
        # takes in a state, either a full state (dict, tensor or array), or 
        # state = input state (gamma), tar_state = target state (delta)
        # computes the interaction value, the mean of forward model, the mean of the passive model if the target state is not None
        inp_state, tar_state = self._wrap_state(state)
        rv = self.output_normalization_function.reverse
        if self.multi_instanced:
            return self.interaction_model.instance_labels(inp_state), self.split_instances(rv(self.forward_model(inp_state)[0])), self.split_instances(rv(self.passive_model(tar_state)[0])) if tar_state is not None else None
        else:
            return self.interaction_model(inp_state), rv(self.forward_model(inp_state)[0]), rv(self.passive_model(tar_state)[0]) if tar_state is not None else None

    def check_interaction(self, inter):
        return inter > self.interaction_prediction

    def collect_samples(self, rollouts, use_trace=False):
        self.sample_able = StateSet()
        for state,next_state in zip(rollouts.get_values("state"), rollouts.get_values("next_state")):
            if use_trace:
                inter = self._set_traces(state, [self.control_feature], [self.target_name])
            else:
                inter = self.interaction_model(self.gamma(state))
            if inter > self.interaction_prediction:
                inputs, targets = [self.gamma(state)], [self.delta(next_state)]
                if self.multi_instanced:
                    inter_bin = self.interaction_model.instance_labels(self.gamma(state))
                    inter_bin[inter_bin<.2] = 0
                    idxes = inter_bin.nonzero()
                    mvtg = self.split_instances(self.delta(next_state))
                    inputs, targets = list(), list()
                    # print(inter_bin.shape)
                    for idx in idxes:
                        # print(inter_bin[0, idx[1]])
                        # print(idx, mvtg.shape, inter_bin.shape)
                        targets.append(mvtg[idx[1]])
                print("sample", inter, inputs, targets)
                for tar in targets:
                    sample = pytorch_model.unwrap(tar) * pytorch_model.unwrap(self.selection_binary)
                    self.sample_able.add(sample)
        # if self.iscuda:
        #     self.sample_able.cuda()
        print(self.sample_able.vals)

    def sample(self, states):
        if self.sample_continuous: # TODO: states should be a full environment state, so need to apply delta to get the appropriate parts
            weights = np.random.random((len(self.cfselectors,))) # random weight vector
            lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
            len_cfs = np.array([j-i for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])
            edited_features = lower_cfs + len_cfs * weights
            new_states = copy.deepcopy(states)
            for f, cfs in zip(edited_features, self.cfselectors):
                cfs.assign_feature(new_states, f)
            if len(new_states.shape) > 1: # if a stack, duplicate mask for all
                return self.delta(new_states), pytorch_model.wrap(torch.stack([self.selection_binary.clone() for _ in range(new_states.size(0))], dim=0), cuda=self.iscuda)
            return self.delta(new_states), self.selection_binary.clone()
        else: # sample discrete with weights, only handles single item sampling
            value = np.random.choice(self.sample_able.vals)
            return value.clone(), self.selection_binary.clone()


interaction_models = {'neural': NeuralInteractionForwardModel, 'dummy': DummyModel}