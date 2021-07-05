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
    def wrap(data, dtype=torch.float, cuda=True, device = None):
        # print(Variable(torch.Tensor(data).cuda()))
        if type(data) == torch.Tensor:
            if cuda: # TODO: dtype not handeled 
                return data.clone().detach().cuda(device=device)
            else:
                return data.clone().detach()
        else:
            if cuda:
                return torch.tensor(data, dtype=dtype).cuda(device=device)
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
        # print(val, self.mean, self.inv_std)
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
        for layer in self.model:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth":
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Network):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            elif type(layer) == nn.Linear:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    print("init form", self.init_form)
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
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
                    # print("layer", self.init_form)

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
        self.hs = kwargs['hidden_sizes']
        self.use_layer_norm = kwargs['use_layer_norm']

        if len(self.hs) == 0:
            if self.use_layer_norm: 
                self.model = nn.Sequential(nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.num_outputs))
            else:
                self.model = nn.Sequential(nn.Linear(self.num_inputs, self.num_outputs))
        elif self.use_layer_norm:
            self.model = nn.Sequential(
                *([nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
            )
        else:
            self.model = nn.Sequential(
                *([nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True)] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True)] for i in range(len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
            )
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x

class BasicConvNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = kwargs['hidden_sizes']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        include_last = kwargs['include_last']

        if len(self.hs) == 0:
            layers = [nn.Conv1d(self.object_dim, self.output_dim, 1)]
        else:
            if len(self.hs) == 1:
                layers = [nn.Conv1d(self.object_dim, self.hs[0], 1)]
            elif self.use_layer_norm:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(len(self.hs) - 1)], list())
                    + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            else:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True)] + 
                      sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True)] for i in range(len(self.hs) - 1)], list())
                      + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv1d(self.hs[-1], self.output_dim, 1)]
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x


class PointNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        self.object_dim = kwargs['object_dim']
        self.hs = kwargs["hidden_sizes"]
        self.output_dim = self.hs[-1]
        kwargs["include_last"] = False
        self.conv = BasicConvNetwork(**kwargs)
        kwargs["include_last"] = True
        kwargs["num_inputs"] = self.output_dim
        kwargs["hidden_sizes"] = [512] # TODO: hardcoded final hidden sizes for now
        self.MLP = BasicMLPNetwork(**kwargs)
        self.model = nn.Sequential(self.conv, self.MLP)

    def forward(self, x):
        if len(x.shape) == 1:
            nobj = x.shape[0] // self.object_dim
            x.view(nobj, self.object_dim)
        elif len(x.shape) == 2:
            nobj = x.shape[1] // self.object_dim
            x.view(-1, nobj, self.object_dim)

        x = self.conv(x)
        # TODO: could use additive instead of max
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_dim)
        x = self.MLP(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x

class PairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        self.object_dim = kwargs['object_dim']
        self.hs = kwargs["hidden_sizes"]
        self.first_obj_dim = kwargs["first_obj_dim"]
        if kwargs["first_obj_dim"] > 0: # only supports one to many concatenation, not many to many
            kwargs["object_dim"] += self.first_obj_dim
        if kwargs["aggregate_final"]: 
            self.output_dim = self.hs[-1]
            kwargs["include_last"] = False
        else:
            kwargs["include_last"] = True
            self.output_dim = kwargs['output_dim']
        self.conv = BasicConvNetwork(**kwargs)
        self.aggregate_final = kwargs["aggregate_final"]
        if kwargs["aggregate_final"]:
            kwargs["include_last"] = True
            kwargs["num_inputs"] = self.output_dim
            kwargs["hidden_sizes"] = [] # TODO: hardcoded final hidden sizes for now
            self.MLP = BasicMLPNetwork(**kwargs)
            self.model = nn.Sequential(self.conv, self.MLP)
        else:
            self.model = [self.conv]

    def forward(self, x):
        if len(x.shape) == 1:
            batch_size = 1
            output_shape = x.shape[0] - self.first_obj_dim
            if self.first_obj_dim > 0:
                fx = x[:self.first_obj_dim] # TODO: always assumes first object dim is the first dimensions
                x = x[self.first_obj_dim:]
            nobj = x.shape[0] // self.object_dim
            x = x.view(nobj, self.object_dim)
            if self.first_obj_dim > 0:
                broadcast_fx = torch.stack([fx.clone() for i in range(nobj)], dim=0)
                x = torch.cat((broadcast_fx, x), dim=1)
            x = x.transpose(1,0)
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            batch_size = x.shape[0]
            output_shape = x.shape[1] - self.first_obj_dim
            if self.first_obj_dim > 0:
                fx = x[:, :self.first_obj_dim] # TODO: always assumes first object dim is the first dimensions
                x = x[:, self.first_obj_dim:]
            nobj = x.shape[1] // self.object_dim
            x = x.view(-1, nobj, self.object_dim)
            if self.first_obj_dim > 0:
                broadcast_fx = torch.stack([fx.clone() for i in range(nobj)], dim=1)
                # print(broadcast_fx.shape, x.shape)
                x = torch.cat((broadcast_fx, x), dim=2)
            # torch.set_printoptions(threshold=1000000)
            # print(x)
            x = x.transpose(2,1)


        x = self.conv(x)
        if self.aggregate_final:
            # TODO: could use additive instead of max
            x = torch.max(x, 2, keepdim=True)[0]
            # print(x.shape)
            x = x.view(-1, self.output_dim)
            x = self.MLP(x)
        else:
            x = x.transpose(2,1)
            # print(output_shape, x.shape)
            x = x.reshape(batch_size, -1)
        # print(x.shape)
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