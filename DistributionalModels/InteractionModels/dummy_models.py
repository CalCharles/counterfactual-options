import numpy as np
from EnvironmentModels.environment_model import FeatureSelector, ControllableFeature
from DistributionalModels.InteractionModels.interaction_model import StateSet
from Networks.network import pytorch_model


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
        self.control_min, self.control_max = 0, 3
        self.iscuda = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def hypothesize(self, state): # gives an "interaction" at some random locations
        if abs(state['factored_state']['Ball'][0] - state['factored_state']['Block'][0]) < 15 and abs(state['factored_state']['Ball'][1] - state['factored_state']['Block'][1] < 10): 
            return np.array(1), np.array(0), np.array(0)
        return np.array(0), np.array(0), np.array(0)

class DummyModel():
    def __init__(self,**kwargs):
        self.environment_model = kwargs['environment_model']
        self.gamma = self.environment_model.get_raw_state
        self.delta = self.environment_model.get_object
        self.controllable = list()
        self.name = "RawModel"
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
