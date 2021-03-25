import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if type(data) == torch.Tensor:
            if cuda: # TODO: dtype not handeled 
                return data.clone().detach().cuda()
            else:
                return data.clone().detach()
        else:
            if cuda:
                return torch.tensor(data, dtype=dtype).cuda()
            else:
                return torch.tensor(data, dtype=dtype)

    @staticmethod
    def unwrap(data):
        if type(data) == torch.Tensor:
            return data.clone().detach().cpu().numpy()
        else:
            return data

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

# # normalization functions
class NormalizationFunctions():
    def __init__(self, **kwargs):
        pass

    def __call__(self, val):
        return

    def reverse(self, val):
        return

class ConstantNorm():
    def __init__(self, **kwargs):
        self.mean = kwargs['mean']
        self.std = kwargs['variance']
        self.inv_std = kwargs['invvariance']

    def __call__(self, val):
        return (val - self.mean) * self.inv_std

    def reverse(self, val):
        return val * self.std + self.mean

    def cuda(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            self.inv_std = self.inv_std.cuda()

    def cpu(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
            self.inv_std = self.inv_std.cpu()

## end of normalization functions

class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_inputs, self.num_outputs = kwargs["num_inputs"], kwargs["num_outputs"]
        self.init_form = kwargs["init_form"]
        self.layers = []
        self.acti = self.get_acti(kwargs["activation"])
        self.iscuda = False

    def cuda(self):
        super().cuda()
        self.iscuda = True

    def cpu(self):
        super().cpu()
        self.iscuda = False


    def run_acti(self, acti, x):
        if acti is not None:
            return acti(x)
        return x

    def get_acti(self, acti):
        if acti == "relu":
            return F.relu
        elif acti == "sin":
            return torch.sin
        elif acti == "sigmoid":
            return torch.sigmoid
        elif acti == "tanh":
            return torch.tanh
        elif acti == "none":
            return None

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth":
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Network):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            else:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                    elif self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                    elif self.init_form == "smalluni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "knorm":
                        torch.nn.init.kaiming_normal_(layer.weight.data)
                    elif self.init_form == "kuni":
                        torch.nn.init.kaiming_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if layer.bias is not None:                
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
                    print("layer", self.init_form)

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def set_parameters(self, param_val): # sets the parameters of a model to the parameter values given as a single long vector
        if len(param_val) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_val = param_val[pval_idx : pval_idx+param_size]
            if type(cur_param_val) == torch.Tensor:
                param.data = cur_param_val.reshape(param.size()).float().clone()
            else:
                param.data = torch.from_numpy(cur_param_val) \
                              .reshape(param.size()).float()
            pval_idx += param_size
        if self.iscuda:
            self.cuda()

    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            # print(param.size(), np.prod(param.size()), self.insize, self.hidden_size)
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    def forward(self, x):
        '''
        all should have a forward function, but not all forward functions have the same signature
        '''
        return

class BasicMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor = kwargs['factor']
        self.num_layers = kwargs['num_layers']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.hidden_size = self.factor*self.factor*self.factor // min(4,self.factor)
        
        print("Network Sizes: ", self.num_inputs, self.num_outputs, self.hidden_size)
        if self.num_layers == 1:
            self.l1 = nn.Linear(self.num_inputs,self.num_outputs)
        elif self.num_layers == 2:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.num_outputs)
            if self.use_layer_norm:
                self.ln1 = nn.LayerNorm(self.hidden_size)
        elif self.num_layers == 3:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l3 = nn.Linear(self.hidden_size, self.num_outputs)
            if self.use_layer_norm:
                self.ln1 = nn.LayerNorm(self.hidden_size)
                self.ln2 = nn.LayerNorm(self.hidden_size)
        if self.num_layers > 0:
            self.layers.append(self.l1)
            if self.use_layer_norm:
                self.layers.append(self.ln1)
        if self.num_layers > 1:
            self.layers.append(self.l2)
            if self.use_layer_norm:
                self.layers.append(self.ln2)
        if self.num_layers > 2:
            self.layers.append(self.l3)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        if self.num_layers > 0:
            x = self.l1(x)
            if self.use_layer_norm and self.num_layers > 1:
                x = self.ln1(x)
            # print(x)
        if self.num_layers > 1:
            x = self.acti(x)
            x = self.l2(x)
            if self.use_layer_norm:
                x = self.ln2(x)
            # print(x)
        if self.num_layers > 2:
            x = self.acti(x)
            x = self.l3(x)
            if self.use_layer_norm:
                x = self.ln3(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x

class FactoredMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor = kwargs['factor']
        self.num_layers = kwargs['num_layers']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.MLP = BasicMLPNetwork(**kwargs)
        self.train()
        self.reset_parameters()

    def basic_operations(self, x):
        # add, subtract, outer product
        return

    def forward(self, x):
        x = self.basic_operations(x)
        x = self.MLP(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x