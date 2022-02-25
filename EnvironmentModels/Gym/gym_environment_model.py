import numpy as np
import os, cv2, time, torch
from EnvironmentModels.environment_model import EnvironmentModel

class GymEnvironmentModel(EnvironmentModel):
    def __init__(self, gym_environment):
        super().__init__(gym_environment)
        action_shape = gym_environment.action_space.shape
        self.action_shape = action_shape
        state_shape = gym_environment.observation_space.shape
        self.state_shape = state_shape
        state = gym_environment.get_state()
        self.object_names = ["State", "Frame", "Object", "Action", 'Done', "Reward"] # TODO: Reward missing from the objects
        self.object_sizes = {"Action": int(np.prod(action_shape)), "State": int(np.prod(state_shape)), "Frame": int(np.prod(state_shape)), "Object": int(np.prod(state_shape)), 'Done': 1, "Reward": 1}
        self.object_num = {"Action": 1, "State": 1, "Frame": 1, "Object": 1, 'Done': 1, "Reward": 1}
        self.state_size = int(np.prod(state_shape))
        self.shapes_dict = {"state": [self.state_size], "next_state": [self.state_size], "state_diff": [self.state_size], "action": action_shape, "done": [1]}
        self.enumeration = {"Action": [0,1], "State": [1,2], "Frame": [2,3], "Object": [3,4], "Done": [4,5], "Reward": [5,6]}
        self.param_size = self.state_size
        self.set_indexes()
        self.flat_rel_space = gym_environment.state_space
        self.reduced_flat_rel_space = gym_environment.state_space

    def get_HAC_flattened_state(self, full_state, instanced=False, use_instanced=True):
        return self.flatten_factored_state(full_state['factored_state'], names=self.object_names[:1])

    def get_raw_state(self, full_state):
        # print(state)
        return full_state['raw_state']
        # if type(state) == dict:
        #     return state["State"]
        # else:
        #     return state[self.object_sizes["Action"]:self.object_sizes['State'] + self.object_sizes["Action"]]

    def get_param(self, full_state):
        raw_state = self.get_raw_state(full_state)
        # if len(raw_state.shape) == 1:
        #     raw_state = raw_state.reshape(1, raw_state.shape[0])
        return raw_state

    def get_factored_state(self, instanced = False): # "instanced" indicates if a single type can have multiple instances (true), or if all of the same type is grouped into a single vector
        return self.environment.extracted_state_dict()

    def get_object(self, state): # "instanced" indicates if a single type can have multiple instances (true), or if all of the same type is grouped into a single vector
        return self.get_raw_state(state)


    def set_from_factored_state(self, factored_state, instanced = False, seed_counter=-1):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        raise NotImplementedError("I don't know Gym well enough yet")

