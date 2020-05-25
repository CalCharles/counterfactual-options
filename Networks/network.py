import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True):
        # print(Variable(torch.Tensor(data).cuda()))
        if type(data) == torch.Tensor:
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

class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_inputs, self.num_outputs = kwargs["num_inputs"], kwargs["num_outputs"]
        self.init_form = kwargs["init_form"]
        self.layers = []

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth":
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Model):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            else:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    # print("layer", self, layer)
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                    if self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                    if self.init_form == "smalluni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if layer.bias is not None:                
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
        print("parameter number", self.count_parameters(reuse=False))

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

    def forward(self, x):
        '''
        all should have a forward function, but not all forward functions have the same signature
        '''
        return

    def train(self, dataset, labels):
        '''
        multiple objectives might require a different signature, but all networks should have some train function
        '''
        pass

class BasicMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor = kwargs['factor']
        self.hidden_size = factor*factor*factor // min(4,factor)
        print("Network Sizes: ", self.num_inputs, self.num_outputs, self.hidden_size)
        if args.num_layers == 1:
            self.l1 = nn.Linear(self.num_inputs,self.num_outputs)
        elif args.num_layers == 2:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.num_outputs)
        elif args.num_layers == 3:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l3 = nn.Linear(self.hidden_size, self.num_outputs)
        if args.num_layers > 0:
            self.layers.append(self.l1)
        if args.num_layers > 1:
            self.layers.append(self.l2)
        if args.num_layers > 2:
            self.layers.append(self.l3)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        if self.num_layers > 0:
            x = self.l1(x)
            x = self.acti(x)
        if self.num_layers > 1:
            x = self.l2(x)
            x = self.acti(x)
        if self.num_layers > 2:
            x = self.l3(x)
            x = self.acti(x)
        return x