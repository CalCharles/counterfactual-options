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
    def __init__(self, **kwargs):
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

    def assign_mean_var(self, mean, std, inv_std):
        self.input_mean, self.input_var = pytorch_model.wrap(mean, cuda=self.iscuda, device=self.device), pytorch_model.wrap(std, cuda=self.iscuda, device=self.device)

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
        self.object_dim = kwargs["object_dim"]

    def compute_input_norm(self, vals, device=None):
        self.device=device
        if type(vals) != torch.tensor:
            vals = pytorch_model.unwrap(vals)
        split_vals = list()
        for v in vals:
            split_vals += [v[i*self.object_dim: (i+1) * self.object_dim] for i in range(v.shape[0] // self.object_dim)]
        vals = np.stack(split_vals, axis=0)
        self.std = np.ones((vals.shape[-1], ))
        self.inv_std = np.ones((vals.shape[-1], ))
        self.mean = np.zeros((vals.shape[-1], ))
        if len (vals) >= 500: # need at least 500 values before applying input variance, typically this is the number of random actions
            self.std = np.sqrt(np.var(vals, axis=0))
            self.std[self.std < .0001] = .0001 # to prevent divide by zero errors
            self.inv_std = 1/self.std
            self.mean = np.mean(vals, axis=0)
        self.mean, self.std, self.inv_std = pytorch_model.wrap(self.mean, device=self.device), pytorch_model.wrap(self.std, device=self.device), pytorch_model.wrap(self.inv_std, device=self.device)

    def assign_mean_var(self, mean, std, inv_std):
        self.mean, self.std, self.inv_std = pytorch_model.wrap(mean, cuda=self.iscuda, device=self.device), pytorch_model.wrap(std, cuda=self.iscuda, device=self.device), pytorch_model.wrap(inv_std, cuda=self.iscuda, device=self.device)

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        count = val.shape[-1] // self.mean.shape[0]
        broadcast_mean = torch.cat([self.mean.clone() for _ in range(count)], dim=0) if count > 1 else self.mean
        broadcast_inv_std = torch.cat([self.inv_std.clone() for _ in range(count)], dim=0) if count > 1 else self.inv_std
        return (val - broadcast_mean) * broadcast_inv_std

    def reverse(self, val):
        count = val.shape[-1] // self.mean.shape[0]
        broadcast_mean = torch.cat([self.mean.clone() for _ in range(count)], dim=0) if count > 1 else self.mean
        broadcast_std = torch.cat([self.std.clone() for _ in range(count)], dim=0) if count > 1 else self.std
        return val * broadcast_std + broadcast_mean

    def cuda(self, device = None):
        self.iscuda = True
        self.device = device 
        self.mean, self.std, self.inv_std = pytorch_model.wrap(self.mean, cuda=True, device=self.device), pytorch_model.wrap(self.std, cuda=True, device=self.device), pytorch_model.wrap(self.inv_std, cuda=True, device=self.device)

    def cpu(self):
        self.iscuda = False
        self.mean, self.std, self.inv_std = self.mean.cpu(), self.std.cpu(), self.inv_std.cpu()

class PointwiseConcatNorm(PointwiseNorm):
    def __init__(self, **kwargs):
        # norms, but with broadcasting for multiple instances flattened
        super().__init__(**kwargs)
        self.first_dim = kwargs['first_obj_dim']

    def compute_input_norm(self, vals, device=None):
        self.device=device
        if type(vals) != torch.tensor:
            vals = pytorch_model.unwrap(vals)
        split_vals_first = list()
        split_vals_broad = list()
        for v in vals:
            split_vals_first = v[:self.first_dim]
            split_vals += [v[i*self.object_dim + self.first_dim: (i+1) * self.object_dim + self.first_dim] for i in range(v.shape[0] // self.object_dim)]
        vals = np.stack(split_vals, axis=0)
        first_vals = np.stack(split_vals_first, axis=0)
        self.first_std = np.ones((first_vals.shape[-1], ))
        self.first_inv_std = np.ones((first_vals.shape[-1], ))
        self.first_mean = np.zeros((first_vals.shape[-1], ))
        self.std = np.ones((vals.shape[-1], ))
        self.inv_std = np.ones((vals.shape[-1], ))
        self.mean = np.zeros((vals.shape[-1], ))
        if len (vals) >= 5000: # need at least 5000 values before applying input std (assuming num objects around 100), typically this is the number of random actions
            self.std = np.sqrt(np.var(vals, axis=0))
            self.std[self.std < .0001] = .0001 # to prevent divide by zero errors
            self.inv_std = 1/self.std
            self.mean = np.mean(vals, axis=0)
            self.first_var = np.sqrt(np.var(first_vals, axis=0))
            self.first_std[self.first_std < .0001] = .0001 # to prevent divide by zero errors
            self.first_inv_std = 1/self.first_std
            self.first_mean = np.mean(first_vals, axis=0)
        self.mean, self.std, self.inv_std = pytorch_model.wrap(self.mean, device=self.device), pytorch_model.wrap(self.std, device=self.device), pytorch_model.wrap(self.inv_std, device=self.device)
        self.first_mean, self.first_std, self.first_inv_std = pytorch_model.wrap(self.first_mean, device=self.device), pytorch_model.wrap(self.first_std, device=self.device), pytorch_model.wrap(self.first_inv_std, device=self.device)

    def cuda(self, device = None):
        self.iscuda = True
        self.device = device 
        self.mean, self.std, self.inv_std = pytorch_model.wrap(self.mean, cuda=True, device=self.device), pytorch_model.wrap(self.std, cuda=True, device=self.device), pytorch_model.wrap(self.inv_std, cuda=True, device=self.device)
        self.first_mean, self.first_std, self.first_inv_std = pytorch_model.wrap(self.first_mean, cuda=True, device=self.device), pytorch_model.wrap(self.first_std, cuda=True, device=self.device), pytorch_model.wrap(self.first_inv_std, cuda=True, device=self.device)

    def cpu(self):
        self.iscuda = False
        self.mean, self.std, self.inv_std = self.mean.cpu(), self.std.cpu(), self.inv_std.cpu()
        self.first_mean, self.first_std, self.first_inv_std = self.first_mean.cpu(), self.first_std.cpu(), self.first_inv_std.cpu()

    def assign_mean_var(self, first_mean, first_std, first_inv_std, mean, std, inv_std):
        self.mean, self.std, self.inv_std = pytorch_model.wrap(mean, cuda=self.iscuda, device=self.device), pytorch_model.wrap(std, cuda=self.iscuda, device=self.device), pytorch_model.wrap(inv_std, cuda=self.iscuda, device=self.device)
        self.first_mean, self.first_std, self.first_inv_std = pytorch_model.wrap(first_mean, cuda=self.iscuda, device=self.device), pytorch_model.wrap(first_std, cuda=self.iscuda, device=self.device), pytorch_model.wrap(first_inv_std, cuda=self.iscuda, device=self.device)

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        count = (val.shape[-1] - self.first_dim) // self.mean.shape[0]
        broadcast_mean = torch.cat([self.mean.clone() for _ in range(count)], dim=0) if count > 1 else self.mean
        broadcast_inv_std = torch.cat([self.inv_std.clone() for _ in range(count)], dim=0) if count > 1 else self.inv_std

        first_val = (val[...,:self.first_dim] - self.first_mean) * self.first_inv_std
        rem = (val[...,self.first_dim:] - broadcast_mean) * broadcast_inv_std
        return torch.cat([first_val, rem], dim=len(val.shape) - 1)

    def reverse(self, val):
        count = (val.shape[-1] - self.first_dim) // self.mean.shape[0]
        broadcast_mean = torch.cat([self.mean.clone() for _ in range(count)], dim=0) if count > 1 else self.mean
        broadcast_std = torch.cat([self.std.clone() for _ in range(count)], dim=0) if count > 1 else self.std
        
        first_val = (val[...,:self.first_dim] * self.first_std) + self.first_mean
        rem = (val[...,self.first_dim:] * broadcast_std) + broadcast_mean
        return torch.cat([first_val, rem], dim=len(val.shape) - 1 )

