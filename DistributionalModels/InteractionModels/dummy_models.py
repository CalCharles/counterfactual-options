import numpy as np
import torch
import copy
from EnvironmentModels.environment_model import FeatureSelector, ControllableFeature
from Environments.SelfBreakout.breakout_objects import intersection
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
        self.multi_instanced = False

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
        self.multi_instanced = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def hypothesize(self, state): # gives an "interaction" when the Ball state is within a certain margin of the block
        if abs(state[0] - state[5]) < 9 and abs(state[1] - state[6] < 6): 
            return np.array(1), np.array(0), np.array(0)
        return np.array(0), np.array(0), np.array(0)


def intersection(a, b, awh, bwh):
    return (abs(a[1] - b[1]) * 2 < (awh[0] + bwh[0])) and (abs(a[0] - b[0]) * 2 < (awh[1] + bwh[1]))


class DummyMultiBlockDatasetModel():
    def __init__(self, environment_model, multi_instanced = True):
        self.interaction_prediction = .3
        self.interaction_minimum = .9
        self.multi_instanced = multi_instanced
        self.gamma = environment_model.create_entity_selector(["Ball", "Block"])
        self.delta = environment_model.create_entity_selector(["Block"])
        self.num_blocks = environment_model.environment.num_blocks
        fs = FeatureSelector([19+ i * 5 for i in range(environment_model.environment.num_blocks)], {"Block": 4}, {"Block": np.array([[4, 19 + i * 5] for i in range(environment_model.environment.num_blocks)])}, ["Block"])
        rng = np.array([0,1])
        self.cfselectors = [ControllableFeature(fs, rng, 1, self)]
        self.sample_able = StateSet([np.array([0,0,0,0,0])])
        self.selection_binary = pytorch_model.wrap(np.array([0,0,0,0,1]))
        self.name = "Ball->Block"
        self.block_width = environment_model.environment.block_width
        self.block_height = environment_model.environment.block_height
        self.ball_width = environment_model.environment.ball.height
        self.ball_height = environment_model.environment.ball.height
        self.control_min, self.control_max = 0, 1
        self.iscuda = False
        self.multi_instanced = True

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def split_instances(self, state):
        cat_state = [state[...,i*5:(i+1)*5] for i in range(self.num_blocks)]
        if len(state.shape) == 1:
            return np.stack(cat_state, axis=0) 
        return np.concatenate(cat_state, axis=-2)

    def hypothesize(self, state): # gives an "interaction" when the Ball state is within a certain margin of the block
        ball_pos = state[:2]
        ball_next_pos = ball_pos + state[2:4]
        for i in range(self.num_blocks):
            block_pos = state[5 + i * 5: 5 + i * 5 + 2]
            block_attr = state[9 + i*5]
            # print(ball_next_pos, block_pos, intersection(ball_next_pos, block_pos, (self.ball_width, self.ball_height), (self.block_width, self.block_height)), block_attr)
            if intersection(ball_next_pos, block_pos, (self.ball_width, self.ball_height), (self.block_width, self.block_height)) and block_attr > 0:
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
        fs3 = FeatureSelector([9], {"Block": 2}, {"Block": np.array([2, 9])}, ["Block"])
        rng1, rng2, rng3 = np.array([-.26,.07]), np.array([-.17, .17]), np.array([.82, .83])
        self.cfselectors = [ControllableFeature(fs1, rng1, 1, self), ControllableFeature(fs2, rng2, 1, self)]
        self.cfnonselector = [ControllableFeature(fs3, rng3, 1, self)]
        self.sample_able = StateSet([np.array([1])])
        self.selection_binary = pytorch_model.wrap(np.array([1]))
        self.name = "Block+Obstacle->Target" # name needs this form for network initializaiton
        self.control_min, self.control_max = np.array([-.25, -.15]), np.array([.05, .15])
        self.iscuda = False
        self.multi_instanced = False

    def cuda(self):
        self.iscuda = True

    def cpu(self):
        self.iscuda = False

    def hypothesize(self, state): # gives an "interaction" when the Ball state is within a certain margin of the block
        return np.array(0), np.array(0), np.array(0)
