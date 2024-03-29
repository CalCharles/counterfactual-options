import torch
import numpy as np
import time, copy, cv2
from tianshou.data import ReplayBuffer, Batch

class ObjDict(dict):
    def __init__(self, ins_dict=None):
        super().__init__()
        if ins_dict is not None:
            for n in ins_dict.keys(): 
                self[n] = ins_dict[n]

    def insert_dict(self, ins_dict):
        for n in ins_dict.keys(): 
            self[n] = ins_dict[n]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

# class Rollouts():
#     _key_conversion = {"state": "obs", "action": "act", "reward": "rew", "next_state": "obs_next"}
#     def __init__(self, length, stack_num, shapes_dict):
#         self.shapes_dict = shapes_dict
#         if length > 0:
#             self.buffer = ReplayBuffer(size=length, stack_num=stack_num, ignore_obs_next=False, save_only_last_obs=False, sample_avail=False)
#             self.values = ObjDict() # initialize values pointers with the keys in shapes_dict
#             init_batch = dict()
#             kck = set(_key_conversion.keys())
#             for k in shapes_dict.keys():
#                 if k in kck:
#                     k = _key_conversion[k]
#                 init_batch[k] = {}

#             self.data = Batch(**init_batch)
#             self.buffer.add(self.data)
#         self.iscuda = False # dynamically convert to cuda if cuda


#     def cuda(self):
#         self.iscuda = True

#     def cpu(self):
#         self.iscuda = False

#     def save(self, pth):
#         self.buffer.save_hdf5(pth)

#     def load(self, pth):
#         self.buffer = load_hdf5(ReplayBuffer, pth)

#     def append(self, **kwargs):
#         updated_args = dict()
#         kck = set(_key_conversion.keys())
#         for k in kck:
#             updated_args[_key_conversion[k]] = kwargs[k] 
#         self.batch.update(**updated_args) # I want to keep replacing the recurring batch, but I'm not sure that is what is happening here
#         ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data) # buffer_ids=ready_env_ids


#     def get_value(self, name):
#         return self.values[name]

#     def insert_rollout(self, other, i=-1, j=-1, a=-1, add=False, name=""): # I hope I don't use this
#         raise NotImplementedError

#     def append_rollout(self, other):
#         return self.buffer.update(other.buffer)

#     def split_indexes(self, idxes,existing=None):
#         if existing is None: # existing saves compute by using a pre-generated rollout object
#             rollout = type(self)(len(idxes), self.stack_num, self.shapes)
#         else:
#             rollout = existing
#         if self.iscuda:
#             rollout.cuda()
#         batch = self.buffer[idxes]
#         rollout.buffer.update(batch)
#         return rollout

#     def split_trajectories(self):
#         indexes = self.values.done.nonzero()[:,0].flatten()
#         # print(indexes)
#         last_idx = 0
#         splitted = []
#         for idx in indexes: # inefficient, should have a one-liner for this
#             if idx != self.filled-1:
#                 splitted.append(self.split_indexes(np.arange(last_idx, idx+1)))
#                 last_idx = idx+1
#             else:
#                 break
#         return splitted

#     def split_train_test(self, ratio):
#         idxes = list(range(len(self.buffer)))
#         train_num = int(self.filled * ratio)
#         train = np.random.choice(idxes, size=train_num, replace=False)
#         vals = train.tolist()
#         vals.sort()
#         test = copy.copy(idxes)
#         a = 0
#         popped = 0
#         while len(vals) > 0:
#             if a == vals[0]:
#                 test.pop(a - popped)
#                 popped += 1
#                 vals.pop(0)
#             a += 1
#         return self.split_indexes(np.array(train)), self.split_indexes(np.array(test))


#     def get_batch(self, n, existing=None):
#         '''
#         just calls sample from data/buffer
#         Removed the following input parameters: weights=None, ordered=False, idxes=[], existing=None
#         Things like prioritized replay will have to be supported at initialization
#         '''
#         if not existing:
#             existing = type(self)(n, self.stack_num, self.shapes_dict)
#         ts_batch, idxes = self.buffer.sample(n)
#         existing.update(tsbatch)
#         return existing

#     def get_ts_batch(self, n): # added function to make learning algorithm life easier
#         return self.buffer.sample(n)

# def merge_rollouts(rols, set_dones=False):
#     total_len = sum([len(r) for r in rols])
#     rollout = type(rols[0])(total_len, rols[0].shapes)
#     at = 0
#     for r in rols:
#         rollout.buffer.update(r.buffer)
#         at += r.filled
#     rollout.at = at 
#     return rollout

# Planning to remove below, but keeping for now
class Rollouts():
    '''
    a unified format for keeping data
    '''
    def __init__(self, length, shapes_dict):
        self.length = length
        self.filled = 0
        self.names = []
        self.values = ObjDict(dict())
        self.at = 0
        self.shapes = ObjDict(shapes_dict)
        self.shapes["done"] = (1,) # dones are universal
        self.iscuda = False

    def cuda(self):
        self.iscuda = True
        for n in self.names:
            self.values[n] = self.values[n].detach().cuda()
        return self

    def cpu(self):
        self.iscuda = False
        for n in self.names:
            self.values[n] = self.values[n].detach().cpu()
        return self

    def initialize_shape(self, shape_dict, create=False):
        if create:
            self.values = ObjDict({n: self.init_or_none(shape_dict[n]) for n in self.names})
        else:
            for n, tensor_shape in shape_dict.items():
                self.values[n] = self.init_or_none(tensor_shape)

    def init_or_none(self, tensor_shape):
        if tensor_shape is None:
            return None
        else:
            return torch.zeros(self.length, *tensor_shape).detach()

    def insert_or_none(self, i, name, v):
        if self.values[name] is not None and v is not None:
            if type(v) != torch.Tensor:
                v = torch.tensor(v) # TODO: replace with torch wrapper/cuda handling etc.
            # print(name, self.values[name].shape, v.shape, v)
            # self.values[name][i] = v
            self.values[name][i].copy_(v.detach())


    def insert(self, i, insert_dict):
        for n in self.names:
            if n in insert_dict:
                self.insert_or_none(i, n, insert_dict[n])

    def cut_range(self, start, end):
        for n in self.names:
            self.values[n] = self.values[start:end]

    def cut_filled(self):
        self.cut_range(0, self.filled)

    def append(self, **kwargs):
        # if self.filled == self.length:
        #     for n in self.names:
        #         self.values[n] = self.values[n].roll(-1, 0).detach()
        self.filled += int(self.filled < self.length)
        self.insert(self.at, kwargs)
        self.at = (self.at + 1) % self.length

    def insert_value(self, i, j, a, name, other):
        # insert to handle wraparound
        space = min(self.length-i, a)
        if len(other.shape) == 1: # put in a dimension to a length n vector to be n x 1
            other = other.unsqueeze(1)
        if space > 0:
            self.values[name][i:i+space] = other[j:j+space]
        wrap = a - (self.length - i)
        if self.wrap and wrap > 0 and self.filled == self.length: # only allows single wrap
            self.values[name][:wrap] = other[j+space:j+space+wrap]
        # print(i, i + space, wrap, j, j + space, j + space+ wrap)
        # print("other", other[j:j+space])
        # print("values",self.values[name][i:i+space] )

    def insert_rollout(self, other, i=-1, j=-1, a=-1, add=False, name=""):
        # insert the other rollout at location i starting from location j in the other rollout with amount a
        # if a is negative use the filled amount of other
        if a < 1:
            a = other.filled
        if i < 0:
            i = self.at
        if j < 0: # used the last one, invalid if at is 0
            j = other.at - 1 if other.at != 0 else other.filled - 1
        if len(name) != 0:
            self.insert_value(i, j, a, name, other.get_values(name))
        else:
            for n in self.names:
                # append as much as fits
                space = min(self.length-i, a)
                self.values[n][i:i+space] = other.get_values(n)[j:j+space]
        if add:
            self.at = (self.at + a) % self.length
            self.filled = min(self.filled + a, self.length)

    def append_rollout(self, other):
        self.insert_rollout(other, j=0)
        self.at = (self.at + other.filled) % self.length

    def split_range(self, i, j):
        rollout = type(self)(self.length, self.shapes)
        if self.iscuda:
            rollout.cuda()
        for n in self.names:
            if self.values[n] is not None:
                rollout.values[n] = self.values[n][i:j]
        rollout.filled = j - i
        return rollout

    def split_train_test(self, ratio):
        idxes = list(range(self.filled))
        train_num = int(self.filled * ratio)
        train = np.random.choice(idxes, size=train_num, replace=False)
        vals = train.tolist()
        vals.sort()
        test = copy.copy(idxes)
        a = 0
        popped = 0
        while len(vals) > 0:
            if a == vals[0]:
                test.pop(a - popped)
                popped += 1
                vals.pop(0)
            a += 1
        return self.split_indexes(np.array(train)), self.split_indexes(np.array(test))

    def split_indexes(self, idxes,existing=None):
        if existing is None: # existing saves compute by using a pre-generated rollout object
            rollout = type(self)(len(idxes), self.shapes)
        else:
            rollout = existing
        if self.iscuda:
            rollout.cuda()
        for n in self.names:
            if self.values[n] is not None:
                rollout.values[n] = self.values[n][idxes.tolist()].clone().detach()
        rollout.filled = len(idxes)
        return rollout

    def split_trajectories(self):
        '''
        generates rollout objects for each trajectory, ignoring wraparound properties
        '''
        indexes = self.values["done"].nonzero()[:,0].flatten()
        # print(indexes)
        last_idx = 0
        splitted = []
        for idx in indexes: # inefficient, should have a one-liner for this
            if idx != self.filled-1:
                splitted.append(self.split_range(last_idx, idx+1))
                last_idx = idx+1
            else:
                break
        return splitted

    def copy_values(self, i, n, other, oi):
        for k in self.values.keys():
            # vlen = min(i+n, len(val)) - max(i, 0)
            # olen = min(oi+on, len(oval)) - max(oi, 0)
            self.values[k][i:i+n].copy_(other.values[k][oi:oi+n].detach())

    def values_at(self, i):
        return {n: v[i] for n,v in self.values.items()}

    def get_values(self, name):
        if self.filled == self.length:
            return torch.cat([self.values[name][self.at:], self.values[name][:self.at]], dim=0)
        return self.values[name][:self.filled]

    def get_batch(self, n, weights=None, ordered=False, idxes=[], existing=None):
        if len(idxes) > 0:
            pass
        elif ordered:
            # idxes = np.arange(self.filled)[self.filled-n:]
            # if self.filled==self.length:
            idxes = np.flip((self.at - np.arange(n) - 1) % self.filled) 
        else:
            idxes = np.random.choice(np.arange(self.filled), size=n, p=weights)
        # print(idxes, self.values.action[:100])
        return idxes, self.split_indexes(idxes, existing=existing)


def merge_rollouts(rols, set_dones=False):
    total_len = sum([r.filled for r in rols])
    rollout = type(rols[0])(total_len, rols[0].shapes)
    at = 0
    for r in rols:
        rollout.copy_values(at, r.filled, r, 0)
        if set_dones:
            rollout.values["done"][at:at+r.filled] = 0
            rollout.values["done"][at+r.filled-1] = 1
        at += r.filled
    rollout.at = at 
    return rollout
