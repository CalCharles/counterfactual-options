import numpy as np
import os, cv2, time, torch
from EnvironmentModels.environment_model import EnvironmentModel


class Nav2DEnvironmentModel(EnvironmentModel):
    def __init__(self, nav2d):
        super().__init__(nav2d)
        self.object_names = ["Action", "Pos", "Target", 'Frame', 'Done', "Reward"] # TODO: Reward missing from the objects
        self.object_sizes = {"Action": 3, "Pos": 3, "Target": 3, 'Frame': (self.environment.N**2)*3, 'Done': 1, "Reward": 1}
        self.object_num = {"Action": 1, "Pos": 1, "Target": 1, 'Frame': 1, 'Done': 1, "Reward": 1}
        self.state_size = (self.environment.N**2)*3# sum([self.object_sizes[n] * self.object_num[n] for n in self.object_names])
        self.shapes_dict = {"state": [self.state_size], "next_state": [self.state_size], "state_diff": [self.state_size], "action": [1], "done": [1]}
        self.enumeration = {"Action": [0,1], "Pos": [1,2], "Target": [2,3], 'Frame': [3,4]}
        self.param_size = self.environment.N**2
        self.set_indexes()

    # def get_interaction_trace(self, name):
    #     trace = []
    #     for i in range(*self.enumeration[name]):
    #         # print(name, self.environment.objects[i].interaction_trace)
    #         trace.append(self.environment.objects[i].interaction_trace)
    #     return trace

    # def set_interaction_traces(self, factored_state):
    #     self.set_from_factored_state(factored_state)
    #     self.environment.step(factored_state["Action"][-1])
    #     self.set_from_factored_state(factored_state)
    def get_raw_state(self, state):
        if type(state) == dict:
            return state["Frame"]
        else:
            return state[self.enumeration['Frame'][0]*3:self.object_sizes['Frame'] + (self.enumeration['Frame'][0]*3)]

    def get_param(self, factored_state):
        raw_state = self.get_raw_state(factored_state)
        if len(raw_state.shape) == 1:
            raw_state = raw_state.reshape(*self.environment.reshape)
        elif len(raw_state.shape) == 2:
            raw_state = raw_state.reshape(-1,*self.environment.reshape)
        if len(raw_state.shape) == 2:
            return raw_state[:,:,:,2].flatten()
        else:
            return raw_state[:,:,2].flatten()

    def get_factored_state(self, instanced = False): # "instanced" has no use here
        factored_state = {n: [] for n in self.object_names}
        factored_state["Action"] = [0,0,self.environment.action]
        factored_state["Pos"] = self.environment.pos.tolist() + [1]
        factored_state["Target"] = self.environment.target.tolist() + [1]
        factored_state["Frame"] = self.environment.frame.flatten()
        factored_state["Done"] = [float(self.environment.done)]
        factored_state["Reward"] = [float(self.environment.reward)]
        return factored_state

    def flatten_factored_state(self, factored_state, instanced=False, names=None):
        if names is None:
            names = self.object_names
        if type(factored_state) == np.ndarray or type(factored_state) == torch.Tensor: # already flattened
            return factored_state
        if type(factored_state) == list:
            flattened_state = np.array([np.concatenate([factored_state[i][f] for f in names], axis=1) for i in range(factored_state)])
        else:
            flattened_state = np.array(np.concatenate([factored_state[f] for f in names], axis=0))
        return flattened_state

    def unflatten_state(self, flattened_state, vec=False, instanced=False, names=None):
        if names is None:
            names = self.object_names
        def unflatten(flattened):
            at = 0
            factored = dict()
            for name in self.object_names:
                factored[name] = flattened[at:at+(self.object_sizes[name]*self.object_num[name])]
                at += (self.object_sizes[name]*self.object_num[name])
            return factored
        if len(flattened_state.shape) == 2:
            if vec:
                factored = unflatten(flattened_state)
            else:
                factored = []
                for i in range(flattened_state.shape[0]):
                    factored.append(unflatten(flattened_state[i]))
        else: # assumes state is a vector
            factored = unflatten(flattened_state)
        return factored

    def set_from_factored_state(self, factored_state, instanced = False, seed_counter=-1):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.environment.seed_counter = seed_counter
        op = self.environment.pos
        self.environment.pos = np.array(factored_state["Pos"][:2])
        ot = self.environment.target
        self.environment.target = np.array(factored_state["Target"][:2])
        self.environment.frame = factored_state["Frame"].reshape((self.environment.N, self.environment.N, 3))
        self.environment.action = factored_state["Action"][-1]
        self.environment.done, self.environment.reward = factored_state["Done"], factored_state["Reward"]
        
