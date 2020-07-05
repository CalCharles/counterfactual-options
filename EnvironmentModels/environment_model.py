import numpy as np
import os, cv2, time
import torch
from Rollouts.rollouts import Rollouts


class EnvironmentModel():
    def __init__(self, environment):
        self.environment = environment
        self.object_names = [] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = {} # must be initialized, a dictionary of name to length of the state
        self.object_num = {} # must be initialized, a dictionary of name to number of instances of the object (max if not fixed)

    def get_factored_state(self):
        '''
        gets the factored state for the current environment
        factored states have the format: {"name": state, ...,}
        '''
        return

    def flatten_factored_state(self, factored_state, names=None):
        ''' 
        generates an nxdim state from a list of factored states. Overloaded to accept single factored states as well 
        This is in the environment model because the order shoud follow the order of the object names
        '''

    def unflatten_state(self, flattened_state, vec=False, typed=False, names=None):
        ''' 
        generates a list of factored states from an nxdim state. Overloaded to accept length dim vector as well 
        This is in the environment model because the order shoud follow the order of the object names
        '''


    def set_from_factored_state(self, factored_state):
        '''
        from the factored state, sets the environment.
        If the factored state is not complete, then this function does as good a reconstruction as possible
        '''
        pass

    def get_done(self, factored_state):
        return factored_state['Done'][-1]

    def get_reward(self, factored_state):
        return factored_state['Reward'][-1]

    def get_action(self, factored_state):
        return factored_state['Action'][-1]


    def step(self, action):
        '''
        steps the environment. in general, this should just defer to calling the environment step.
        if the model is learned, however, this could be more involved
        '''
        return self.environment.step(action)

    def save_factored_state(self, factored_state, save_raw = False):
        factored_str = ""
        for name, state in factored_state.items():
            factored_str = name + ":" + " ".join(map(str, state)) + "\t"
        self.environment.write_objects(factored_str, save_raw=save_raw)

    def get_insert_dict(self, factored_state, last_state=None, typed=False):
        if last_state is None:
            last_state = torch.zeros(self.flatten_factored_state(factored_state, typed=typed).shape)
        state = torch.tensor(self.flatten_factored_state(factored_state, typed=typed)).float()
        insert_dict = {'state': state, 'state_diff': state-last_state, 'done': self.get_done(factored_state), 'action': self.get_action(factored_state)}
        return insert_dict, state

class ModelRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is 1 for discrete, needs to be specified for continuous
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "state_diff", "action", "done"]
        self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})

    def state_equals(self, other, at=-1):
        EPSILON = 1e-10
        if at > 0:
            return (self.values["state"][at] - other.values["state"][at]).norm(p=1) <= EPSILON
        return (self.values["state"] - other.values["state"]).norm(p=1) <= EPSILON