import numpy as np
import os, cv2, time, copy, itertools
import torch
from Rollouts.rollouts import Rollouts, ObjDict
from tianshou.data import Batch
from Networks.network import ConstantNorm

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
        
    def get_state(self):
        return self.environment.get_state()

    def get_object(self, state):
        return state["raw_state"]

    def get_raw_state(self, state): # get the raw form of the state, assumed to be frame, define in subclass if otherwise
        if type(state) == dict:
            return state["raw_state"]
        else:
            at = 0
            for n in self.object_names:
                if n != "Frame":
                    at += self.object_sizes[n] * self.object_num[n]
            return state[at:self.object_sizes['Frame'] + at]

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

    def append_shapes(self, addv):
        if type(addv) != np.ndarray:
            addv = np.array(addv)
        addv=addv.squeeze() # TODO: highly problamatic line but I don't know how to fix it
        # print(addv.shape, addv.tolist())
        if len(addv.shape) == 0:
            return [addv.tolist()]
        return addv.tolist()

    def flatten_factored_state(self, factored_state, instanced=False, names=None):
        ''' 
        generates an nxdim state from a list of factored states. Overloaded to accept single factored states as well 
        This is in the environment model because the order shoud follow the order of the object names
        if the input state is not flattened, return
        '''
        if names is None:
            names = self.object_names
        if type(factored_state) == np.ndarray or type(factored_state) == torch.Tensor: # already flattened
            return factored_state
        if instanced:
            if type(factored_state) == list:
                flattened_state = list()
                for f in factored_state:
                    flat = list()
                    for n in names:
                        if self.object_num[n] > 1:
                            for i in range(self.object_num[n]):
                                flat += self.append_shapes(f[n+str(i)])
                    flattened_state += flat
                flattened_state = np.array(flattened_state)
            else:
                flattened_state = list()
                for n in names:
                    if self.object_num[n] > 1:
                        for i in range(self.object_num[n]):
                            flattened_state += self.append_shapes(factored_state[n+str(i)])
                    else:
                        flattened_state += self.append_shapes(factored_state[n])
                flattened_state = np.array(flattened_state)
        else:
            if type(factored_state) == list:
                flattened_state = np.array([np.concatenate([factored_state[i][f] for f in names], axis=1) for i in range(factored_state)])
            else:
                # print(factored_state)
                flattened_state = np.array(np.concatenate([factored_state[f] for f in names], axis=0))
        return flattened_state


    def unflatten_state(self, flattened_state, vec=False, instanced=False, names=None):
        ''' 
        generates a list of factored states from an nxdim state. Overloaded to accept length dim vector as well 
        This is in the environment model because the order shoud follow the order of the object names
        '''
        if names is None:
            names = self.object_names
        def unflatten(flattened):
            at = 0
            factored = dict()
            for name in self.object_names:
                if instanced: #factor each object, even those of the same type 
                    for k in range(self.object_num[name]):
                        usename = name
                        if self.object_num[name] > 1:
                            usename = name+str(k)
                        if vec:
                            factored[name] = flattened[:, at:at+self.object_sizes[name]]
                        else: # a single state at a time
                            factored[name] = flattened[at:at+self.object_sizes[name]]
                        at += self.object_sizes[name]
                else: # factor each object, grouping objects of the same type
                    if vec:
                        factored[name] = flattened[:, at:at+(self.object_sizes[name]*self.object_num[name])]
                    else: # a single state at a time
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

    def set_from_factored_state(self, factored_state):
        '''
        from the factored state, sets the environment.
        If the factored state is not complete, then this function does as good a reconstruction as possible
        '''
        pass

    def get_param(self, factored_state): # define in the subclass function
        return

    def get_done(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state['Done'][-1]

    def get_reward(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state['Reward'][-1]

    def get_action(self, full_state):
        factored_state = full_state['factored_state']
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

    def get_insert_dict(self, factored_state, next_factored_state, last_state=None, instanced=False, action_shift=False):
        if last_state is None:
            last_state = torch.zeros(self.flatten_factored_state(factored_state, instanced=instanced).shape)
        if action_shift:
            factored_state['Action'] = copy.copy(next_factored_state['Action'])
        state = torch.tensor(self.flatten_factored_state(factored_state, instanced=instanced)).float()
        next_state = torch.tensor(self.flatten_factored_state(next_factored_state, instanced=instanced)).float()
        insert_dict = {'state': state, 'next_state': next_state, 'state_diff': next_state-state, 'done': self.get_done({"factored_state": factored_state}), 'action': self.get_action({"factored_state": factored_state})}
        return insert_dict, state

    def create_entity_selector(self, names):
        flat_features = list()
        factored_features = dict()
        feature_match = dict()
        for name in names:
            flat_features += list(range(*self.indexes[name]))
            factored_features[name] = np.arange(self.object_sizes[name]).astype('int')
            feature_match[name] = np.array([[fac, flt] for fac,flt in zip(factored_features[name], list(range(*self.indexes[name])))])
        return FeatureSelector(flat_features, factored_features, feature_match, names)

    def flat_to_factored(self, idx):
        for name in self.indexes.keys():
            if self.indexes[name][0] <= idx < self.indexes[name][1]: 
                return name, idx - self.indexes[name][0]

    def get_subset(self, entity_selector, flat_bin):
        # flat_bin is a binary vector that is 1 when the feature is chosen and 0 otherwise
        # flat_bin[2] = 1
        a = entity_selector.flat_features[torch.nonzero((flat_bin == 1).long())].flatten()
        fs = dict()
        ad = dict()
        names = list()
        for i, (n, idx) in zip(a, [self.flat_to_factored(i) for i in a]):
            if n in ad:
                fs[n].append([i,idx])
                ad[n].append(idx)
            else:
                fs[n] = [[i, idx]]
                ad[n] = [idx]
            names.append(n)
        lastn = names[0]
        clean_names = [lastn]
        for n in names:
            if n != lastn:
                clean_names.append(n)
        fs = {n: np.array(fs[n]) for n in fs.keys()}
        return FeatureSelector(a, ad, fs, clean_names)

    def construct_action_selector(self):
        if self.environment.discrete_actions:
            return FeatureSelector([self.indexes['Action'][1] - 1], {'Action': self.object_sizes['Action'] - 1}, {'Action': np.array([self.indexes['Action'][1] - 1, self.object_sizes['Action'] - 1])}, ['Action'])
        else:
            return [FeatureSelector([self.indexes['Action'][0] + i], {'Action': i}, {'Action': np.array([self.indexes['Action'][0] + i, i])}, ['Action']) for i in range( self.object_sizes['Action'])]

def get_selection_list(cfss): # TODO: put this inside ControllableFeature as a static function

    possibility_lists = list()
    for cfs in cfss: # order of cfs matter
        fr, s = list(), cfs.feature_range[0]
        while s <= cfs.feature_range[1]: # allows for single value controllable features to avoid bugs
            fr.append(s)
            s += cfs.feature_step
        possibility_lists.append(fr)
    print( cfs.feature_range, cfs.feature_step)
    return itertools.product(*possibility_lists)

def sample_multiple(controllable_features, states):
    linspaces = list()
    cf = controllable_features[0]
    # print(len(controllable_features), (cf.feature_range[1] - cf.feature_range[0] ), cf.feature_step)
    for cf in controllable_features:
        num = int((cf.feature_range[1] - cf.feature_range[0] ) / cf.feature_step) + 1
        linspaces.append(np.linspace(cf.feature_range[0], cf.feature_range[1], num))
    ranges = np.meshgrid(*linspaces)
    ranges = [r.flatten() for r in ranges]
    vals = np.array(ranges).T
    MAX_NUM = 200
    if len(vals) > 0:
        vals = vals[np.random.choice(np.arange(len(vals)), size=(200,), replace=False)]
    all_states = list()
    for control_values in vals:
        assigned_states = states.clone()
        for cf, f in zip(controllable_features, control_values):
            cf.assign_feature(assigned_states, f)
        all_states.append(assigned_states)
    if len(states.shape) == 1: # a single flattened state
        return torch.stack(all_states, dim=0) # if there are no batches, then this is the 0th dim
    return torch.stack(all_states, dim=1) # if we have a batch of states, then this is the 1st dim


class ControllableFeature():
    def __init__(self, feature_selector, feature_range, feature_step, feature_model=None):
        self.feature_selector = feature_selector
        self.feature_range = feature_range # two element list
        print("cf", feature_range)
        self.feature_step = feature_step
        self.feature_model = feature_model # if the control of this feature can be captured by a model (i.e. an interaction model or combination of models)

    def hypothesize(self, state):
        return self.feature_model.hypothesize(state)

    def object(self):
        return self.feature_selector.get_entity()[0]

    def get_indexes(self):
        # return list(self.feature_selector.values())[0]
        return self.feature_selector.flat_features[0]

    def sample_feature(self, states):
        all_states = []
        num = int((self.feature_range[1] - self.feature_range[0] ) / self.feature_step)
        # for f in np.linspace(*self.feature_range, num):
        f = self.feature_range[0]
        while f <= self.feature_range[1]:
            assigned_states = states.clone()
            self.assign_feature(assigned_states, f)
            # assign_feature(assigned_states, (self.feature_selector.flat_features[0], f))
            all_states.append(assigned_states)
            f += self.feature_step
        if len(states.shape) == 1: # a single flattened state
            return torch.stack(all_states, dim=0) # if there are no batches, then this is the 0th dim
        return torch.stack(all_states, dim=1) # if we have a batch of states, then this is the 1st dim

    def get_num_steps(self):
        return (self.feature_range[1] - self.feature_range[0]) // self.feature_step

    def get_steps(self):
        return [i * self.feature_step + self.feature_range[0] for i in range(self.get_num_steps())]

    def assign_feature(self, states, f, factored=False, edit=False, clipped=False):
        clip = self.feature_range if clipped else None
        if factored:
            assign_feature(states, (list(self.feature_selector.factored_features.items())[0], f), edit=edit, clipped=clip)
        else:
            assign_feature(states, (self.feature_selector.flat_features[0], f), edit=edit, clipped=clip)
        return states # assign feature should mutate states

class FeatureSelector():
    def __init__(self, flat_features, factored_features, feature_match, names):
        '''
        flat features: a list of indexes which are the features when flat
        factored_features: a dict of the entity name and the feature indices as a numpy array, with only one index per tuple
        For the hash function to work, feature selectors must be immutable, and the hash is the tuple constructed by the flat_features
        To ensure immutability, the flat features must be sorted, and the factored features must be sorted to match the flat features
        '''
        self.flat_features = np.array(flat_features).astype("int")
        self.factored_features = factored_features
        self.feature_match = feature_match # a dict of name: [[factored feature, flat feature]...]
        n = list(self.feature_match.keys())[0]
        self.names=names

        def find_idx(i, l2):
            for idx, k in enumerate(l2):
                if k == i:
                    return idx
            return -1

        def hash_names(n, on): # TODO: relative state defined by object order rather than ord order
            if n < on:
                return (n, on)
            return (on, n)

        self.relative_indexes = dict()
        self.relative_flat_indexes = dict()
        for n in self.names:
            for on in self.names: # for each name
                if n != on:
                    fac, flt = self.feature_match[n][:,0], self.feature_match[n][:,1]
                    fac2, flt2 = self.feature_match[on][:,0], self.feature_match[on][:,1]
                    for i, t in zip(fac, flt):
                        idx = find_idx(i, fac2)
                        if idx != -1:

                            hsh = hash_names(n, on)
                            if hsh not in self.relative_indexes:
                                self.relative_indexes[hsh] = list()
                                self.relative_flat_indexes[hsh] = list()
                            if i not in self.relative_indexes[hsh]:
                                self.relative_indexes[hsh].append(i)
                                self.relative_flat_indexes[hsh].append([t,flt2[idx]])
        # print(self.relative_indexes, self.relative_flat_indexes)
        if len(list(self.relative_flat_indexes.keys())) > 0:
            self.relative_flat_indexes = np.concatenate([np.array(self.relative_flat_indexes[hsh]) for hsh in self.relative_flat_indexes.keys()], axis=0)
        print(list(self.feature_match.keys()), self.names, self.relative_indexes, self.relative_flat_indexes)

        self.clean_factored_features()

    def __hash__(self):
        return tuple(self.flat_features)

    def __eq__(self, other):
        return tuple(self.flat_features) == tuple(other.flat_features)

    def get_entity(self):
        return list(self.factored_features.keys())

    def output_size(self):
        return len(self.flat_features)

    def clean_factored_features(self): # only cleaned once at the start
        for name in self.names:
            self.factored_features[name] = np.array(self.factored_features[name]).astype("int")
        for tup in self.relative_indexes.keys():
            self.relative_indexes[tup] = np.array(self.relative_indexes[tup]).astype("int")


    def get_relative(self, states):
        # print("states", states, self.relative_indexes)
        if type(states) is dict or type(states) is Batch: # factored features
            if type(states[self.names[0]]) == np.ndarray:
                cat = lambda x, a: np.concatenate(x, axis=a)
            elif type(states[self.names[0]]) == torch.Tensor:
                cat = lambda x, a: torch.cat(x, dim=a)
            if len(states[self.names[0]].shape) == 1: # only support internal dimension up to 2
                return cat([states[names[0]][i] - states[names[1]][i] for names, i in self.relative_indexes.items()], 0)
            if len(states[self.names[0]].shape) == 2: # only support internal dimension up to 2
                return cat([states[names[0]][:, i] - states[names[1]][:, i] for names, i in self.relative_indexes.items()], 1)
            # return [states[names[0]][i] - states[names[1]][i] for names, i in self.relative_indexes.items()]
        elif len(states.shape) == 1: # a single flattened state
            return states[self.relative_flat_indexes[:,0]] - states[self.relative_flat_indexes[:,1]]
        elif len(states.shape) == 2: # a batch of flattened state
            return states[:, self.relative_flat_indexes[:,0]] - states[:, self.relative_flat_indexes[:,1]]
        elif len(states.shape) == 3: # a batch of stacks of flattened state
            return states[:,:, self.relative_flat_indexes[:,0]] - states[:,:, self.relative_flat_indexes[:,1]]


    def __call__(self, states):
        if type(states) is dict or type(states) is Batch: # factored features, returns a flattened state to match the other calls
            # ks = self.factored_features.keys()
            # return {name: states[name][idxes] for name, idxes in self.factored_features}
            # TODO: Above lines are old code, if new code breaks revert back, otherwise remove
            if not hasattr(self, "names"): 
                pnames = list(self.factored_features.keys())
                self.names = [n for n in ["Action", "Paddle", "Ball", "Block", 'Done', "Reward"] if n in pnames] # TODO: above lines are hack, remove
            # print(states[self.names[0]], type(states[self.names[0]]) is torch.Tensor, type(states[self.names[0]]) == torch.tensor)
            if type(states[self.names[0]]) == np.ndarray:
                cat = lambda x, a: np.concatenate(x, axis=a)
            elif type(states[self.names[0]]) == torch.Tensor:
                cat = lambda x, a: torch.cat(x, dim=a)
            if len(states[self.names[0]].shape) == 1: # only support internal dimension up to 2
                if len(self.names) == 1 and len(self.factored_features[self.names[0]].shape) == 0: # TODO: should initialize self.feactored_features as arrays, mking this code wrong
                    if type(states[self.names[0]]) == np.ndarray:
                        cat = np.array
                    elif type(states[self.names[0]]) == torch.Tensor:
                        cat = torch.tensor
                    return cat([states[name][self.factored_features[name]] for name in self.names])
                return cat([states[name][self.factored_features[name]] for name in self.names], 0)
            if len(states[self.names[0]].shape) == 2: # only support internal dimension up to 2
                return cat([states[name][:, self.factored_features[name]] for name in self.names], 1)
        elif len(states.shape) == 1: # a single flattened state
            return states[self.flat_features]
        elif len(states.shape) == 2: # a batch of flattened state
            return states[:, self.flat_features]
        elif len(states.shape) == 3: # a batch of stacks of flattened state
            return states[:, :, self.flat_features]

    def reverse(self, delta_state, insert_state):
        if type(insert_state) == dict: # only works for dict to one object
            for name, idxes in self.factored_features.keys():
                insert_state[name][idxes] = delta_state
        elif len(insert_state.shape) == 1:
            insert_state[self.flat_features] = delta_state
        elif len(insert_state.shape) == 2: # a batch of flattened state
            insert_state[:, self.flat_features] = delta_state
        elif len(insert_state.shape) == 3: # a batch of flattened state
            insert_state[:, self.flat_features] = delta_state

def assign_feature(states, assignment, edit=False, clipped=None): 
# assignment is a tuple assignment keys (tuples or indexes), and assignment values
# edit means that the assignment is added 
    if type(states) is dict: # factored features
        states[assignment[0][0]][assignment[0][1]] = assignment[1] if not edit else states[assignment[0][0]][assignment[0][1]] + assignment[1]
        if clipped is not None:
            states[assignment[0][0]][assignment[0][1]] = states[assignment[0][0]][assignment[0][1]].clamp(clipped[0], clipped[1])
    elif len(states.shape) == 1: # a single flattened state
        states[assignment[0]] = assignment[1] if not edit else states[assignment[0]] + assignment[1]
        if clipped is not None:
            states[assignment[0]] = states[assignment[0]].clamp(clipped[0], clipped[1])
    elif len(states.shape) == 2: # a batch of flattened state
        states[:, assignment[0]] = assignment[1] if not edit else states[:, assignment[0]] + assignment[1]
        if clipped is not None:
            cstates = states[:, assignment[0]].clamp(clipped[0], clipped[1])
            states[:, assignment[0]] = cstates
    elif len(states.shape) == 3: # a batch of stacks of flattened state
        states[:, :, assignment[0]] = assignment[1] if not edit else states[:, :, assignment[0]] + assignment[1]
        if clipped is not None:
            cstates = states[:, :, assignment[0]].clamp(clipped[0], clipped[1])
            states[:, :, assignment[0]] = cstates

def discretize_actions(action_shape): # converts a continuous action space into a discrete one
    # takes action +- 1, 0 at each dimension, for every combination
    # creates combinatorially many combinations of this
    # action space is assumed to be the tuple shape of the space
    # TODO: assume action space of the form (n,)
    actions = list()
    def append_str(i, bs):
        if i == action_shape[0]: actions.append(bs)
        else:
            bsn1 = copy.copy(bs)
            bsn1.append(-1)
            append_str(i+1, bsn1)
            bs0 = copy.copy(bs)
            bs0.append(0)
            append_str(i+1, bs)
            bs.append(1)
            append_str(i+1,bs)
    append_str(0, list())
    return {i: actions[i] for i in range(len(actions))} # gives the ordering arrived at by -1, 0, 1 ordering interleaved

class ModelRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is action_chain x 1 for discrete, needs to be specified for continuous action_chain x max_action_dim
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "state_diff", "next_state", "action", "done"]
        if "all_state_next" in list(shapes_dict.keys()):
            self.names.append("all_state_next")
        # print(self.shapes)
        self.initialize_shape(self.shapes, create=True)
        # self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})

    def state_equals(self, other, at=-1):
        EPSILON = 1e-10
        if at > 0:
            return (self.values["state"][at] - other.values["state"][at]).norm(p=1) <= EPSILON
        return (self.values["state"] - other.values["state"]).norm(p=1) <= EPSILON