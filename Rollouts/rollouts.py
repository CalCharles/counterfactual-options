import torch

class Rollouts():
    '''
    a unified format for keeping data
    '''
    def __init__(self, length, shapes_dict):
        self.length = length
        self.filled = 0
        self.names = []
        self.shapes = shapes_dict
        self.shapes["done"] = (1,) # dones are universal

    def init_or_none(self, tensor_shape):
        if tensor_shape is None:
            return None
        else:
            return torch.zeros(self.length, *tensor_shape)

    def insert_or_none(self, i, name, v):
        if self.values[name] is not None and v is not None:
            if type(v) != torch.Tensor:
                v = torch.tensor(v) # TODO: replace with torch wrapper/cuda handling etc.
            self.values[name][i] = v


    def insert(self, i, insert_dict):
        for n in self.names:
            if n in insert_dict:
                self.insert_or_none(i, n, insert_dict[n])

    def append(self, **kwargs):
        if self.filled == self.length:
            for n in self.names:
                self.values[n] = self.values[n].roll(-1, 0).detach()
        self.filled += int(self.filled < self.length)
        self.insert(self.filled-1, kwargs)

    def split_range(self, i, j):
        rollout = type(self)(self.length, self.shapes)
        for n in self.names:
            if self.values[n] is not None:
                rollout.values[n] = self.values[n][i:j]
                rollout.filled = j - i
        return rollout

    def split_trajectories(self):
        '''
        generates rollout objects for each trajectory, ignoring wraparound properties
        '''
        indexes = self.values["done"].nonzero()[:,0].flatten()
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
            self.values[k][i:i+n] = other.values[k][oi:oi+n]

    def values_at(self, i):
        return {n: v[i] for n,v in self.values.items()}

    def get_values(self, name):
        return self.values[name][:self.filled]

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
    return rollout
