import numpy as np
from Networks.network import pytorch_model

class StateSet():
    def __init__(self, init_vals=None, epsilon_close = .1):
        self.vals = list()
        self.close = epsilon_close
        if init_vals is not None:
            for v in init_vals: 
                self.vals.append(v)
        self.iscuda = False

    # def cuda(self):
    #     self.iscuda = True
    #     for i in range(len(self.vals)):
    #         self.vals[i] = self.vals[i].cuda()

    def __getitem__(self, val):
        for iv in self.vals:
            if np.linalg.norm(iv - val, ord=1) < self.close:
                return iv.copy()
        raise AttributeError("No such attribute: " + name)

    def add(self, val):
        val = pytorch_model.unwrap(val)
        i = self.inside(val)
        if i == -1:
            self.vals.append(val)
        return i

    def inside(self, val):
        for i, iv in enumerate(self.vals):
            if np.linalg.norm(iv - val, ord=1) < self.close:
                return i
        return -1

    def pop(self, idx):
        self.vals.pop(idx)
