import numpy as np
import os, cv2, time
import torch
from Rollouts.rollouts import Rollouts, ObjDict


class EnvironmentModel():
    def __init__(self, environment):
        self.environment = environment
        self.frameskip = environment.frameskip
        self.object_names = [] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = {} # must be initialized, a dictionary of name to length of the state
        self.object_num = dict() # must be initialized, a dictionary of name to number of instances of the object (max if not fixed)
        self.enumeration = dict() # the range of instance numbers
        self.indexes = dict() # the range of indexes in a flattened state

    def get_num(self, instanced=False):
        if instanced:
            return sum(self.object_num.values())
        else:
            return len(self.object_num.keys())

    def set_indexes(self):
        idx = 0
        for name in self.object_names:
            step = self.object_sizes[name] * self.object_num[name]
            self.indexes[name] = [idx, idx+step]
            idx += step


    def get_factored_state(self):
        '''
        gets the factored state for the current environment
        factored states have the format: {"name": state, ...,}
        '''
        return

    def get_factored_zero_state(self, instanced = False): # "instanced" indicates if a single type can have multiple instances (true), or if all of the same type is grouped into a single vector
        # gets a factored state with all zeros
        factored_state = self.get_factored_state()
        zero_state = dict()
        for n in factored_state.keys():
            zero_state[n] = torch.zeros(factored_state[n].shape)
        return zero_state

    def cuda_factored(self, factored_state):
        return {n: [state.cuda() for state in factored_state[n]] for n in factored_state.keys()}

    def get_flattened_state(self, names=None):
        return self.flatten_factored_state(self.get_factored_state(), names=names)

    def flatten_factored_state(self, factored_state, names=None):
        ''' 
        generates an nxdim state from a list of factored states. Overloaded to accept single factored states as well 
        This is in the environment model because the order shoud follow the order of the object names
        '''

    def unflatten_state(self, flattened_state, vec=False, instanced=False, names=None):
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

    def get_insert_dict(self, factored_state, next_factored_state, last_state=None, instanced=False):
        if last_state is None:
            last_state = torch.zeros(self.flatten_factored_state(factored_state, instanced=instanced).shape)
        state = torch.tensor(self.flatten_factored_state(factored_state, instanced=instanced)).float()
        next_state = torch.tensor(self.flatten_factored_state(next_factored_state, instanced=instanced)).float()
        insert_dict = {'state': state, 'next_state': next_state, 'state_diff': state-last_state, 'done': self.get_done(factored_state), 'action': self.get_action(factored_state)}
        return insert_dict, state

    def create_entity_selector(self, names):
        flat_features = list()
        factored_features = dict()
        for name in names:
            flat_features += list(range(self.indexes[name]))
            factored_features[name] = np.arange(self.indexes[name]).dtype('int')
        return FeatureSelector(flat_features, factored_features)

class ControllableFeature():
    def __init__(self, feature_selector, feature_range, feature_step ):
        self.feature_selector = feature_selector
        self.feature_range = feature_range # two element list
        self.feature_step = feature_step

    def get_indexes(self):
        return list(self.feature_selector.values())[0]

    def get_num_steps(self):
        return (cfs.feature_range[1] - cfs.feature_range[0]) // cfs.feature_step

    def get_steps(self):
        return [i * cfs.feature_step + cfs.feature_range[0] for i in range(self.get_num_steps)]

class FeatureSelector():
    def __init__(self, flat_features, factored_features):
        '''
        flat features: a list of indexes which are the features when flat
        factored_features: a dict of the entity name and the feature indices as a numpy array, with only one index per tuple
        For the hash function to work, feature selectors must be immutable, and the hash is the tuple constructed by the flat_features
        To ensure immutability, the flat features must be sorted, and the factored features must be sorted to match the flat features
        '''
        self._flat_features = np.array(flat_features).dtype("int")
        self.factored_features = factored_features
        self.clean_factored_features()

    def __hash__(self):
        return tuple(self.flat_features)

    def __eq__(self, other):
        return tuple(self.flat_features) = tuple(other.flat_features)
    
    def get_entity(self):
        return list(self.factored_features.keys())

    def clean_factored_features(self): # only cleaned once at the start
        new_factored = dict()
        for name in self.factored_features.keys():
            if name in new_factored:
                new_factored[name].append(self.factored_features[name])
            else:
                new_factored[name] = [self.factored_features[name]]
        for name in self.factored_features.keys():
            self.factored_features[name] = np.array(self.factored_features[name]).dtype("int")

    def __call__(self, states):
        if type(states) is dict: # factored features
            return {name: states[name][idxes] for name, idxes in self.factored_features}
        elif len(states.shape) == 1: # a single flattened state
            return states[self.flat_features]
        elif len(states.shape) == 2: # a batch of flattened state
            return states[:, self.flat_features]

def assign_feature(self, states, assignment): # assignment is a tuple assignment keys (tuples or indexes), and assignment values
    if type(states) is dict: # factored features
        states[assignment[0][0]][assignment[0][1]] = assignment[1]
    elif len(states.shape) == 1: # a single flattened state
        return states[assignment[0]] = assignment[1]
    elif len(states.shape) == 2: # a batch of flattened state
        return states[:, assignment[0]] = assignment[1]

class ModelRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is action_chain x 1 for discrete, needs to be specified for continuous action_chain x max_action_dim
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "state_diff", "next_state", "action", "done"]
        if "all_state" in list(shapes_dict.shapes_dict.keys()):
            self.names.append("all_state_next")
        print(self.shapes)
        self.initialize_shape(self.shapes, create=True)
        # self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})

    def state_equals(self, other, at=-1):
        EPSILON = 1e-10
        if at > 0:
            return (self.values["state"][at] - other.values["state"][at]).norm(p=1) <= EPSILON
        return (self.values["state"] - other.values["state"]).norm(p=1) <= EPSILON