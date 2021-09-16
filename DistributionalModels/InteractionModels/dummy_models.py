import numpy as np
import torch
from EnvironmentModels.environment_model import FeatureSelector, ControllableFeature
from DistributionalModels.InteractionModels.state_management import StateSet
from Networks.network import pytorch_model

class DummyModel():
    def __init__(self,**kwargs):
        self.environment_model = kwargs['environment_model']
        self.gamma = self.environment_model.get_raw_state
        self.delta = self.environment_model.get_object
        self.controllable = list()
        self.name = "State->Reward"
        self.selection_binary = torch.ones([1])
        self.interaction_model = None
        self.interaction_prediction = None
        self.predict_dynamics = False
        self.iscuda = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def sample(self, states):
        return self.environment_model.get_param(states), self.selection_binary

    def get_active_mask(self):
        return self.selection_binary.clone()


class DummyBlockDatasetModel():
    def __init__(self, environment_model, multi_instanced = False):
        self.interaction_prediction = .3
        self.interaction_minimum = .9
        self.multi_instanced = multi_instanced
        self.gamma = environment_model.create_entity_selector(["Ball", "Block"])
        self.delta = environment_model.create_entity_selector(["Block"])
        fs = FeatureSelector([19], {"Block": 4}, {"Block": np.array([4, 19])}, ["Block"])
        rng = np.array([0,1])
        self.cfselectors = [ControllableFeature(fs, rng, 1, self)]
        self.sample_able = StateSet([np.array([0,0,0,0,0])])
        self.selection_binary = pytorch_model.wrap(np.array([0,0,0,0,1]))
        self.name = "Ball->Block"
        self.control_min, self.control_max = 0, 1
        self.iscuda = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def hypothesize(self, state): # gives an "interaction" when the Ball state is within a certain margin of the block
        if abs(state[0] - state[5]) < 9 and abs(state[1] - state[6] < 6): 
            return np.array(1), np.array(0), np.array(0)
        return np.array(0), np.array(0), np.array(0)

class DummyNegativeRewardDatasetModel():
    def __init__(self, environment_model, multi_instanced = False):
        self.interaction_prediction = .3
        self.interaction_minimum = .9
        self.multi_instanced = multi_instanced
        self.gamma = environment_model.create_entity_selector(["Block", "Obstacle"])
        self.delta = environment_model.create_entity_selector(["Block"])
        fs1 = FeatureSelector([7], {"Block": 0}, {"Block": np.array([0, 7])}, ["Block"])
        fs2 = FeatureSelector([8], {"Block": 1}, {"Block": np.array([1, 8])}, ["Block"])
        rng1, rng2 = np.array([-.3,.1]), np.array([-.2, .2])
        self.cfselectors = [ControllableFeature(fs1, rng1, 1, self), ControllableFeature(fs2, rng2, 1, self)]
        self.sample_able = StateSet([np.array([1])])
        self.selection_binary = pytorch_model.wrap(np.array([1]))
        self.name = "Block+Obstacle->Target" # name needs this form for network initializaiton
        self.control_min, self.control_max = -100, 1
        self.iscuda = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def hypothesize(self, state): # gives an "interaction" when the Ball state is within a certain margin of the block
        return np.array(0), np.array(0), np.array(0)
