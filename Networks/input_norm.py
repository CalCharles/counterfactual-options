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
        input_var, input_mean = self.input_var, self.input_mean
        if val.shape[-1] != self.input_var.shape[-1]: # broadcast mean and variance to fit shape
            val = val.reshape(-1, self.input_var.shape[-1])
        return val * (input_var * 2) + input_mean

    def cuda(self, device = None):
        self.iscuda = True
        self.device = device 
        self.input_mean, self.input_var = pytorch_model.wrap(self.input_mean, cuda=True, device=self.device), pytorch_model.wrap(self.input_var, cuda=True, device=self.device)

    def cpu(self):
        self.iscuda = False
        self.input_mean, self.input_var = self.input_mean.cpu(), self.input_var.cpu()



class PointwiseNorm(InterInputNorm):
    def __init__(self, **kwargs):
        # norms, but with broadcasting for multiple instances flattened
        super().__init__(**kwargs)

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        count = val.shape[-1] // self.mean.shape[0]
        broadcast_mean = torch.cat([self.mean.clone() for _ in range(count)]) if count > 1 else self.mean
        broadcast_inv_std = torch.cat([self.inv_std.clone() for _ in range(count)]) if count > 1 else self.inv_std
        return (val - broadcast_mean) * broadcast_inv_std

    def reverse(self, val):
        count = val.shape[-1] // self.mean.shape[0]
        broadcast_mean = np.concatenate([self.mean.copy() for _ in range(count)]) if count > 1 else self.mean
        broadcast_std = np.concatenate([self.std.copy() for _ in range(count)]) if count > 1 else self.std
        return val * broadcast_std + broadcast_mean

class PointwiseConcatNorm(InterInputNorm):
    def __init__(self, **kwargs):
        # norms, but with broadcasting for multiple instances flattened
        super().__init__(**kwargs)
        self.first_dim = kwargs['first_obj_dim']
        self.first_mean = kwargs['first_mean']
        self.first_std = kwargs['first_variance']
        self.first_inv_std = kwargs['first_invvariance']

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        count = (val.shape[-1] - self.first_dim) // self.mean.shape[0]
        broadcast_mean = np.concatenate([self.mean.copy() for _ in range(count)]) if count > 1 else self.mean
        broadcast_inv_std = np.concatenate([self.inv_std.copy() for _ in range(count)]) if count > 1 else self.inv_std

        first_val = (val[...,:self.first_dim] - self.first_mean) * self.first_inv_std
        rem = (val[...,self.first_dim:] - broadcast_mean) * broadcast_inv_std
        return np.concatenate([first_val, rem], axis=len(val.shape) - 1)

    def reverse(self, val):
        count = (val.shape[-1] - self.first_dim) // self.mean.shape[0]
        broadcast_mean = np.concatenate([self.mean.copy() for _ in range(count)]) if count > 1 else self.mean
        broadcast_std = np.concatenate([self.std.copy() for _ in range(count)]) if count > 1 else self.std
        
        first_val = (val[...,:self.first_dim] * self.first_std) + self.first_mean
        rem = (val[...,self.first_dim:] * broadcast_std) + broadcast_mean
        return np.concatenate([first_val, rem], axis=len(val.shape) - 1)

