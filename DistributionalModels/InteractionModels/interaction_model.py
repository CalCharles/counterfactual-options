import numpy as np
import os, cv2, time, copy, psutil
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
from Rollouts.rollouts import ObjDict, merge_rollouts
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
        self.zeta = kwargs['zeta']
        self.controllable = kwargs['controllable'] # controllable features USED FOR (BEFORE) training
        
        # environment model defines object factorization
        self.environment_model = kwargs['environment_model']
        self.env_name = self.environment_model.environment.name

        # construct the active model
        kwargs['post_dim'] = 0 # no post-state
        print(kwargs['num_outputs'], kwargs['aggregate_final'])
        self.forward_model = forward_nets[kwargs['forward_class']](**kwargs)
        print(self.forward_model)

        # set the passive model
        norm_fn, num_inputs = kwargs['normalization_function'], kwargs['num_inputs']
        kwargs['normalization_function'], kwargs['num_inputs'] = kwargs['delta_normalization_function'], kwargs['num_outputs']
        self.first_obj_dim = kwargs['first_obj_dim']
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

        # The threshold for the interaction model output to predict an interaction
        self.interaction_prediction = kwargs['interaction_prediction']

        # Learn an interaction model which is based on distance to target
        # interaction_prediction remains the threshold for interaction
        self.interaction_distance = kwargs['interaction_distance']

        # output limits
        self.control_min, self.control_max = np.zeros(kwargs['num_outputs']), np.zeros(kwargs['num_outputs'])
        self.object_dim = kwargs["object_dim"]
        self.multi_instanced = kwargs["multi_instanced"] # for when the TARGET is multi instanced
        self.instanced_additional = kwargs["instanced_additional"] # for when the ADDITIONAL is multi instanced
        # assign norm values
        self.normalization_function = kwargs["normalization_function"]
        self.delta_normalization_function = kwargs["delta_normalization_function"]
        # note that self.output_normalization_function is defined in TRAIN, because that is wehre predict_dynamics is assigned
        self.active_epsilon = kwargs['active_epsilon'] # minimum l2 deviation to use the active values
        self.iscuda = kwargs["cuda"]
        # self.sample_continuous = True
        self.selection_binary = pytorch_model.wrap(torch.zeros((self.delta.output_size(),)), cuda=self.iscuda)
        print("precuda", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

        if self.iscuda:
            self.cuda()
        print("postcuda", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

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
        # print(torch.cat([forward_mean, passive_mean, difference, ((not_passive) * (active_prediction) * (difference)).float()], dim=1)[:10], self.forward_threshold, self.passive_threshold, self.difference_threshold)
        return ((not_passive) * (active_prediction) * (difference)).float(), potential #(active_prediction+not_passive > 1).float()

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

    def get_active_mask(self):
        return pytorch_model.unwrap(self.selection_binary.clone())


interaction_models = {'neural': NeuralInteractionForwardModel, 'dummy': DummyModel}