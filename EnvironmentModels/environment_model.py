import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from Rollouts.rollouts import Rollouts, ObjDict
from tianshou.data import Batch
from Networks.network import ConstantNorm, pytorch_model

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

    def get_insert_dict(self, factored_state, next_factored_state, last_state=None, instanced=False, action_shift=False, ignore_done=True):
        if last_state is None:
            last_state = torch.zeros(self.flatten_factored_state(factored_state, instanced=instanced).shape)
        if action_shift:
            factored_state['Action'] = copy.copy(next_factored_state['Action'])
        state = torch.tensor(self.flatten_factored_state(factored_state, instanced=instanced)).float()
        next_state = torch.tensor(self.flatten_factored_state(next_factored_state, instanced=instanced)).float()
        # skip = self.get_done({"factored_state": next_factored_state})
        skip = self.get_done({"factored_state": factored_state})
        info = self.get_info({"factored_state": factored_state})

        # print(skip, self.get_done({"factored_state": factored_stdate}))
        insert_dict = {'state': state, 'next_state': next_state, 'state_diff': next_state-state, 'done': self.get_done({"factored_state": factored_state}), 'action': self.get_action({"factored_state": factored_state}), 'info': info}
        return insert_dict, state, skip

    def match_factored_flat(self, f, name):
        # returns the corresponding flat features for factored feature @param f: int in name @param name
        return [self.indexes[name][0] + f + j * self.object_sizes[name] for j in range(self.object_num[name])]

    def create_entity_selector(self, names, add_relative=False):
        flat_features = list()
        factored_features = dict()
        feature_match = dict()
        for name in names:
            flat_features += list(range(*self.indexes[name]))
            factored_features[name] = np.arange(self.object_sizes[name]).astype('int')
            feature_match_flat = list()
            fac = factored_features[name]
            for i in range(len(factored_features[name])):
                flt = np.array(self.match_factored_flat(i, name)).astype(int)
                feature_match_flat.append(np.array([fac[i], flt]))
            feature_match[name] = np.array(feature_match_flat)
            print(name, flat_features, factored_features)
        return FeatureSelector(flat_features, factored_features, feature_match, names, add_relative=add_relative)

    def flat_to_factored(self, idx):
        for name in self.indexes.keys():
            if self.indexes[name][0] <= idx < self.indexes[name][1]: 
                return name, idx - self.indexes[name][0]

    def get_factored_subset(self, entity_selector, factored_bin):
        # assumes a single object entity selector
        # factored binary for a single instance of an object
        facfea = pytorch_model.unwrap(torch.nonzero((factored_bin == 1).long())).flatten()
        name = entity_selector.names[0]
        factored_features = {name: facfea}
        feature_match = dict()
        flat_features = list()
        names = [name]
        for f in facfea:
            flat_eq = self.match_factored_flat(f, name)
            feature_match[name] = np.array([f, np.array(flat_eq)])
            flat_features += flat_eq
        flat_features.sort()
        print(flat_features)
        control_selector = FeatureSelector(np.array(flat_features), factored_features, feature_match, names)

        facfea = pytorch_model.unwrap(torch.nonzero((factored_bin == 0).long())).flatten()
        print(facfea)
        non_control_selector = None
        if len(facfea) > 0:
            name = entity_selector.names[0]
            factored_features = {name: facfea}
            feature_match = dict()
            flat_features = list()
            names = [name]
            for f in facfea:
                flat_eq = self.match_factored_flat(f, name)
                feature_match[name] = np.array([f, np.array(flat_eq)])
                flat_features += flat_eq
            flat_features.sort()
            non_control_selector = FeatureSelector(np.array(flat_features), factored_features, feature_match, names)
        return control_selector, non_control_selector

    def get_subset(self, entity_selector, flat_bin):
        # flat_bin is a binary vector that is 1 when the feature is chosen and 0 otherwise
        # flat_bin[2] = 1
        a = entity_selector.flat_features[torch.nonzero((flat_bin == 1).long())].flatten()
        fs = dict()
        ad = dict()
        names = list()
        for i, (n, idx) in zip(a, [self.flat_to_factored(i) for i in a]):
            if n in ad:
                fs[n].append([idx, [i]])
                ad[n].append(idx)
            else:
                fs[n] = [[idx, [i]]]
                ad[n] = [idx]
            names.append(n)
        lastn = names[0]
        clean_names = [lastn]
        for n in names:
            if n != lastn:
                clean_names.append(n)
        fs = {n: np.array(fs[n]) for n in fs.keys()}
        control_selector = FeatureSelector(a, ad, fs, clean_names)

        a = entity_selector.flat_features[torch.nonzero((flat_bin == 0).long())].flatten()
        fs = dict()
        ad = dict()
        names = list()
        for i, (n, idx) in zip(a, [self.flat_to_factored(i) for i in a]):
            if n in ad:
                fs[n].append([idx,[i]])
                ad[n].append(idx)
            else:
                fs[n] = [[idx,[i]]]
                ad[n] = [idx]
            names.append(n)
        if len(names) > 0:
            lastn = names[0]
            clean_names = [lastn]
            for n in names:
                if n != lastn:
                    clean_names.append(n)
            fs = {n: np.array(fs[n]) for n in fs.keys()}
            non_control_selector = FeatureSelector(a, ad, fs, clean_names)
        else:
            non_control_selector = None
        return control_selector, non_control_selector


    def construct_action_selector(self):
        if self.environment.discrete_actions:
            return FeatureSelector([self.indexes['Action'][1] - 1], {'Action': [self.object_sizes['Action'] - 1]}, {'Action': np.array([self.indexes['Action'][1] - 1, [self.object_sizes['Action'] - 1]])}, ['Action'])
        else:
            return [FeatureSelector([self.indexes['Action'][0] + i], {'Action': np.array([i])}, {'Action': np.array([self.indexes['Action'][0] + i, [i]])}, ['Action']) for i in range( self.object_sizes['Action'])]

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
    vals = np.array(ranges).T.reshape(-1,len(controllable_features))
    MAX_NUM = 200
    if len(vals) > 0:
        vals = vals[np.random.choice(np.arange(len(vals)), size=(min(MAX_NUM, len(vals)),), replace=False)]
    all_states = list()
    for control_values in vals:
        assigned_states = states.copy()
        for cf, f in zip(controllable_features, control_values):
            cf.assign_feature(assigned_states, f)
        all_states.append(assigned_states)
    if len(states.shape) == 1: # a single flattened state
        return np.stack(all_states, axis=0) # if there are no batches, then this is the 0th dim
    return np.stack(all_states, axis=1) # if we have a batch of states, then this is the 1st dim


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

    def relative_range(self, states, factored=False): 
    # assignment is a tuple assignment keys (tuples or indexes)
    # edit means that the assignment is added
        upper_diff = 0
        lower_diff = 0
        assignment = self.feature_selector.flat_features[0]
        if factored: # factored features, assumes that shapes for assignment[0][1] and assignment[1] match
            assignment = list(self.feature_selector.factored_features.items())[0]
            lower_diff = states[assignment[0]][assignment[1]] - self.feature_range[0]
            upper_diff = self.feature_range[1] - states[assignment[0]][assignment[1]]
        else: # flattened state
            lower_diff = states[...,assignment] - self.feature_range[0]
            upper_diff = self.feature_range[1] - states[...,assignment]
        if type(lower_diff) == np.ndarray:
            lower_diff[lower_diff < 0] = 0
            upper_diff[upper_diff < 0] = 0
        else:
            lower_diff = max(lower_diff, 0) 
            upper_diff = max(upper_diff, 0) 
        return lower_diff, upper_diff 


class FeatureSelector():
    def __init__(self, flat_features, factored_features, feature_match, names, add_relative=False):
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
        self.add_relative = add_relative

        def find_idx(i, l2):
            for idx, k in enumerate(l2):
                if k == i:
                    return idx
            return -1

        def hash_names(n, on): # relative state defined by object order
            if find_idx(n, self.names) < find_idx(on, self.names):
                return (n, on)
            return (on, n)

        self.relative_indexes = dict()
        self.relative_flat_indexes = dict()
        self.name_order = list()
        self.len_relative = 0
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
                                print(hsh, t, flt2[idx])
                                self.relative_indexes[hsh].append(i)
                                for flt2t in flt2[idx]:
                                    self.relative_flat_indexes[hsh].append([t.squeeze(),flt2t])
                                self.len_relative += 1
                            if hsh not in self.name_order:
                                self.name_order.append(hsh)
        print(self.relative_indexes, self.relative_flat_indexes)
        print([(hsh, np.array(self.relative_flat_indexes[hsh]).shape) for hsh in self.name_order])
        if len(list(self.relative_flat_indexes.keys())) > 0:
            self.relative_flat_indexes = np.concatenate([np.array(self.relative_flat_indexes[hsh]) for hsh in self.name_order], axis=0)
        print(list(self.feature_match.keys()), self.names, self.relative_indexes, self.relative_flat_indexes)

        self.clean_factored_features()

    def __hash__(self):
        return tuple(self.flat_features)

    def __eq__(self, other):
        return tuple(self.flat_features) == tuple(other.flat_features)

    def get_entity(self):
        return list(self.factored_features.keys())

    def output_size(self):
        return len(self.flat_features) + int(self.add_relative) * self.len_relative

    def clean_factored_features(self): # only cleaned once at the start
        for name in self.names:
            self.factored_features[name] = np.array(self.factored_features[name]).astype("int")
        for tup in self.relative_indexes.keys():
            self.relative_indexes[tup] = np.array(self.relative_indexes[tup]).astype("int")


    def get_relative(self, states):
        # print("states", states, self.relative_indexes)

        if type(states) is dict or type(states) is Batch or type(states) is OrderedDict: # factored features
            if type(states[self.names[0]]) == np.ndarray:
                cat = lambda x, a: np.concatenate(x, axis=a)
            elif type(states[self.names[0]]) == torch.Tensor:
                cat = lambda x, a: torch.cat(x, dim=a)
            # if len(states[self.names[0]].shape) == 1: # only support internal dimension up to 2
            #     return cat([states[names[0]][i] - states[names[1]][i] for names, i in self.relative_indexes.items()], 0)
            # if len(states[self.names[0]].shape) == 2: # only support internal dimension up to 2
            relative_expanded = dict()
            expanded_order = list()
            for names, i in self.relative_indexes.items():
                if names[1] in states:
                    relative_expanded[names] = i
                else:
                    j=0
                    add_name = names[1] + str(j)
                    while add_name in states:
                        relative_expanded[(names[0], add_name)] = i
                        j += 1
                        add_name = names[1] + str(j)
            return cat([states[names[0]][...,i] - states[names[1]][...,i] for names, i in relative_expanded.items()], -1)
            # return [states[names[0]][i] - states[names[1]][i] for names, i in self.relative_indexes.items()]
        return states[..., self.relative_flat_indexes[:,0]] - states[...,self.relative_flat_indexes[:,1]]


    def __call__(self, states):
        # REMOVE LINE BELOW
        if not hasattr(self, "add_relative"):
            self.add_relative = False

        if type(states) is dict or type(states) is Batch or type(states) is OrderedDict: # factored features, returns a flattened state to match the other calls
            # ks = self.factored_features.keys()
            # return {name: states[name][idxes] for name, idxes in self.factored_features}
            # TODO: Above lines are old code, if new code breaks revert back, otherwise remove
            # if not hasattr(self, "names"): 
            #     pnames = list(self.factored_features.keys())
            #     self.names = [n for n in ["Action", "Paddle", "Ball", "Block", 'Done', "Reward"] if n in pnames] # TODO: above lines are hack, remove
            state_check = states[self.names[0]] if self.names[0] in states else states[self.names[0] + str(0)]
            # TODO: I think I can use ... slice notation here to make this much simpler
            if type(state_check) == np.ndarray:
                cat = lambda x, a: np.concatenate(x, axis=a)
            elif type(state_check) == torch.Tensor:
                cat = lambda x, a: torch.cat(x, dim=a)
            if len(self.names) == 1 and len(self.factored_features[self.names[0]].shape) == 0: # TODO: should initialize self.factored_features as arrays, mking this code wrong
                if type(states[self.names[0]]) == np.ndarray:
                    cat = np.array
                elif type(states[self.names[0]]) == torch.Tensor:
                    cat = torch.tensor
                cut_state = cat([states[name][...,self.factored_features[name]] for name in self.names])
            else:
                state_cat = list()
                for name in self.names:
                    if name in states:
                        state_cat.append(states[name][..., self.factored_features[name]])
                    else:
                        i = 0 # TODO: looks for multiple instances by searching through name+index, not sure if this is great
                        while name + str(i) in states.keys():
                            state_cat.append(states[name + str(i)][..., self.factored_features[name]])
                            i += 1
                cut_state = cat(state_cat, -1)
            if self.add_relative:
                rel_state = self.get_relative(states)
                cut_state = cat([state_cat, rel_state], -1)
            return cut_state
        if self.add_relative:
            if type(states) == np.ndarray:
                cat = lambda x, a: np.concatenate(x, axis=a)
            elif type(states) == torch.Tensor:
                cat = lambda x, a: torch.cat(x, dim=a)
                rel_state = self.get_relative(states)
            return cat([states[..., self.flat_features], rel_state], -1)
        return states[..., self.flat_features]

    def reverse(self, delta_state, insert_state):
        '''
        assigns the relavant values of insert_state to delta_state
        '''
        if type(insert_state) == dict: # only works for dict to one object
            for name, idxes in self.factored_features.keys():
                insert_state[name][idxes] = delta_state
        insert_state[..., self.flat_features] = delta_state

def cpu_state(factored_state):
    fs = dict()
    for k in factored_state.keys():
        fs[k] = pytorch_model.unwrap(factored_state[k])
    return fs


def assign_feature(states, assignment, edit=False, clipped=None): 
# assignment is a tuple assignment keys (tuples or indexes), and assignment values
# edit means that the assignment is added 
    if type(states) is dict or type(states) == OrderedDict or type(states) == Batch: # factored features, assumes that shapes for assignment[0][1] and assignment[1] match
        if clipped is not None: # clip before so values are relative to CLIPPED edges
            states[assignment[0][0]][assignment[0][1]] = states[assignment[0][0]][assignment[0][1]].clip(clipped[0], clipped[1])
        states[assignment[0][0]][assignment[0][1]] = assignment[1] if not edit else states[assignment[0][0]][assignment[0][1]] + assignment[1]
        if clipped is not None:
            states[assignment[0][0]][assignment[0][1]] = states[assignment[0][0]][assignment[0][1]].clip(clipped[0], clipped[1])
    else:
        if clipped is not None:
            cstates = states[...,assignment[0]].clip(clipped[0], clipped[1])
            states[...,assignment[0]] = cstates
        states[...,assignment[0]] = assignment[1] if not edit else states[...,assignment[0]] + assignment[1]
        if clipped is not None:
            cstates = states[...,assignment[0]].clip(clipped[0], clipped[1])
            states[...,assignment[0]] = cstates



def discretize_space(action_shape): # converts a continuous action space into a discrete one
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
        self.names = ["state", "state_diff", "next_state", "action", "done", "info"]
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