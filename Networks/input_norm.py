import numpy as np
import torch
from Networks.network import pytorch_model

class InputNorm():

    def compute_input_norm(self, buffer):
        avail = buffer.sample(0)[0]
        print("trying compute", len(buffer), avail.obs.shape)
        if len (avail) >= 500: # need at least 500 values before applying input variance, typically this is the number of random actions
            print("computing input norm")
            self.input_var = np.sqrt(np.var(avail.obs, axis=0))
            self.input_var[self.input_var < .0001] = .0001 # to prevent divide by zero errors
            self.input_mean = np.mean(avail.obs, axis=0)

    def apply_input_norm(self, batch):
        if self.use_input_norm:
            batch.update(obs=(batch.obs - self.input_mean) / self.input_var)

class InterInputNorm(): # input norm for rollout type data
    def __init__(self):
        self.iscuda=False
        self.device=None

    def compute_input_norm(self, vals, device=None):
        self.device=device
        if type(vals) != torch.tensor:
            vals = pytorch_model.unwrap(vals)
        self.input_var = np.ones((vals.shape[-1], ))
        self.input_mean = np.zeros((vals.shape[-1], ))
        if len (vals) >= 500: # need at least 500 values before applying input variance, typically this is the number of random actions
            self.input_var = np.sqrt(np.var(vals, axis=0))
            self.input_var[self.input_var < .0001] = .0001 # to prevent divide by zero errors
            self.input_mean = np.mean(vals, axis=0)
        self.input_mean, self.input_var = pytorch_model.wrap(self.input_mean, device=self.device), pytorch_model.wrap(self.input_var, device=self.device)

    def assign_mean_var(self, mean, var):
        self.input_mean, self.input_var = pytorch_model.wrap(mean, cuda=self.iscuda, device=self.device), pytorch_model.wrap(var, cuda=self.iscuda, device=self.device)

    def __call__(self, val):
        # print(val-self.input_mean)
        return (val - self.input_mean) / (self.input_var * 2)

    def reverse(self, val):
        return val * (self.input_var * 2) + self.input_mean

    def cuda(self, device = None):
        self.iscuda = True
        self.device = device 
        self.input_mean, self.input_var = pytorch_model.wrap(self.input_mean, cuda=True, device=self.device), pytorch_model.wrap(self.input_var, cuda=True, device=self.device)

    def cpu(self):
        self.iscuda = False
        self.input_mean, self.input_var = self.input_mean.cpu(), self.input_var.cpu()


