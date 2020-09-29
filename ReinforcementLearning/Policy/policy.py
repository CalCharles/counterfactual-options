import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import copy, os
from file_management import default_value_arg

class RLoutput():
    def __init__(self, values, dist_entropy, probs, log_probs, action_values, std, Q_vals, dist):
        self.values = values 
        self.dist_entropy = dist_entropy 
        self.probs = probs 
        self.log_probs = log_probs 
        self.action_values = action_values 
        self.std = std 
        self.Q_vals = Q_vals 
        self.dist = dist

    def values(self):
        return self.values, self.dist_entropy, self.probs, self.log_probs, self.action_values, self.std, self.Q_vals, self.dist


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
                return Variable(torch.tensor(data, dtype=dtype).cuda())
            else:
                return Variable(torch.tensor(data, dtype=dtype))

    @staticmethod
    def unwrap(data):
        return data.clone().detach().cpu().numpy()

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class Policy(nn.Module):
    def __init__(self, **kwargs):
        super(Policy, self).__init__()
        self.num_inputs = default_value_arg(kwargs, 'num_inputs', 10)
        print("num_inputs", self.num_inputs)
        self.param_size = default_value_arg(kwargs, 'param_size', 10)
        self.num_outputs = default_value_arg(kwargs, 'num_outputs', 1)
        self.factor = default_value_arg(kwargs, 'factor', None)
        # self.minmax = default_value_arg(kwargs, 'minmax', None)
        # self.use_normalize = self.minmax is not None
        self.name = default_value_arg(kwargs, 'name', 'option')
        self.iscuda = default_value_arg(kwargs, 'cuda', True) # TODO: don't just set this to true
        self.init_form = default_value_arg(kwargs, 'init_form', 'xnorm') 
        self.scale = default_value_arg(kwargs, 'scale', 1) 
        self.activation = default_value_arg(kwargs, 'activation', 'relu') 
        self.test = not default_value_arg(kwargs, 'train', True) # test is also true if the policy is not trainable
        self.Q_critic = default_value_arg(kwargs, 'Q_critic', False) 
        self.continuous = default_value_arg(kwargs, 'continuous', False)
        model_form = default_value_arg(kwargs, 'train', 'basic') 
        self.has_final = default_value_arg(kwargs, 'needs_final', True)
        self.last_param = default_value_arg(kwargs, 'last_param', False)
        self.normalize = default_value_arg(kwargs, 'normalize', False)
        self.option_values = torch.zeros(1, self.param_size) # changed externally to the parameters
        self.num_layers = default_value_arg(kwargs, 'num_layers', 1)
        if self.num_layers == 0:
            self.insize = self.num_inputs + self.param_size
        else:
            self.insize = self.factor * self.factor * self.factor // min(self.factor, 8)
        if self.last_param:
            self.insize += self.param_size
        self.layers = []
        self.init_last(self.num_outputs)
        if self.activation == "relu":
            self.acti = F.relu
        elif self.activation == "sin":
            self.acti = torch.sin
        elif self.activation == "sigmoid":
            self.acti = torch.sigmoid
        elif self.activation == "tanh":
            self.acti = torch.tanh
        print("activation", self.acti)
        print("current insize", self.insize)
            
    def init_last(self, num_outputs):
        self.critic_linear = nn.Linear(self.insize, 1)
        self.sigma = nn.Linear(self.insize, 1)
        if self.continuous:
            self.QFunction = nn.Linear(self.insize, 1)
        else:
            self.QFunction = nn.Linear(self.insize, num_outputs)
        self.action_eval = nn.Linear(self.insize, num_outputs)
        if len(self.layers) > 0:
            self.layers = self.layers[5:]
        self.layers = [self.critic_linear, self.sigma, self.QFunction, self.action_eval] + self.layers

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                print("layer", layer, self.init_form)
                if self.init_form == "orth":
                    # print(layer.weight.shape, layer.weight)
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Policy):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            else:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    print("layer", layer, self.init_form)
                    # print("layer", self, layer)
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                        # print(layer.weight[10:,10:])
                    if self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 3 / layer.weight.data.shape[0])
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
        # if self.has_final:
        #     nn.init.orthogonal_(self.action_probs.weight.data, gain=0.01)
        print("parameter number", self.count_parameters(reuse=False))

    def last_layer(self, x, param):
        '''
        input: [batch size, insize]
        output [batch size, 1], [batch size, 1], [batch_size, num_actions], [batch_size, num_actions], [batch_size, num_actions]
        '''
        if self.last_param:
            x = torch.cat((x,param), dim=1)
        std = self.sigma(x)
        Q_vals = self.QFunction(x)
        if self.Q_critic:
            values = Q_vals.max(dim=1)[0]
        else:
            values = self.critic_linear(x)
        # print("Qvals", Q_vals, values)
        dist = None
        action_values = self.action_eval(x)
        if self.continuous:
            dist = FixedNormal(action_values, std)
            log_probs = dist.log_probs(actions)
            dist_entropy = dist.entropy().mean()
            probs = torch.exp(log_probs)
        else:
            probs = F.softmax(action_values, dim=1) 
            log_probs = F.log_softmax(action_values, dim=1)
            dist_entropy = action_values - action_values.logsumexp(dim=-1, keepdim=True)
        return values, dist_entropy, probs, log_probs, action_values, std, Q_vals, dist

    def preamble(self, x, p):
        if self.normalize:
            return (x / 84) * self.scale, p
        return x,p


    def forward(self, x, p):
        x, p = self.preamble(x, p) # TODO: if necessary, a preamble can be added back in
        x = self.hidden(x, p)
        values, dist_entropy, probs, log_probs, action_values, std, Q_vals, dist = self.last_layer(x, p)
        return RLoutput(values, dist_entropy, probs, log_probs, action_values, std, Q_vals, dist)

    def save(self, pth, name):
        torch.save(self, os.path.join(pth, name + ".pt"))

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

    # count number of parameters
    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    # TODO: write code to remove last layer if unnecessary
    def remove_last(self):
        self.critic_linear = None
        self.QFunction = None
        self.action_probs = None
        self.layers = self.layers[3:]


class BasicPolicy(Policy):    
    def __init__(self, **kwargs):
        super(BasicPolicy, self).__init__(**kwargs)
        self.hidden_size = self.factor*self.factor*self.factor // min(4,self.factor)
        print("Network Sizes: ", self.num_inputs, self.insize, self.hidden_size)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        if not self.last_param:
            self.num_inputs += self.param_size
        # self.num_inputs = self.param_size # Turn this on to only input the parameter (also line in hidden)
        print(self.last_param, self.num_inputs)
        if self.num_layers == 1:
            self.l1 = nn.Linear(self.num_inputs,self.insize)
        elif self.num_layers == 2:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.insize)
        elif self.num_layers == 3:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l3 = nn.Linear(self.hidden_size, self.insize)
        if self.num_layers > 0:
            self.layers.append(self.l1)
        if self.num_layers > 1:
            self.layers.append(self.l2)
        if self.num_layers > 2:
            self.layers.append(self.l3)
        self.train()
        self.reset_parameters()

    def hidden(self, x, p):
        # print(x.shape, p.shape, x, p)
        if not self.last_param:
            x = torch.cat((x,p), dim=1)
        # x = p # turn this on to only input the parameter
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

    def compute_layers(self, x):
        layer_outputs = []
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())
        if self.num_layers > 1:
            x = self.l2(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())

        return layer_outputs

class ImagePolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: assumes images of size 84x84, make general
        self.no_preamble = True
        self.num_stack = 4
        factor = self.factor
        self.conv1 = nn.Conv2d(self.num_stack, 2 * factor, 8, stride=4)
        self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        self.conv3 = nn.Conv2d(4 * factor, 2 * factor, 3, stride=1)
        self.viewsize = 7
        # if self.args.post_transform_form == 'none':
        #     self.linear1 = None
        #     self.insize = 2 * self.factor * self.viewsize * self.viewsize
        #     self.init_last(self.num_outputs)
        # else:
        self.linear1 = nn.Linear(2 * factor * self.viewsize * self.viewsize, self.insize)
        self.layers.append(self.linear1)
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        self.layers.append(self.conv3)
        self.reset_parameters()

    def hidden(self, inputs, p):
        norm_term = 1.0
        if self.normalize:
            norm_term =  255.0
        x = self.conv1(inputs / norm_term)
        x = self.acti(x)

        x = self.conv2(x)
        x = self.acti(x)

        x = self.conv3(x)
        x = self.acti(x)
        x = x.view(-1, 2 * self.factor * self.viewsize * self.viewsize)
        x = self.acti(x)
        if self.linear1 is not None:
            x = self.linear1(x)
            x = self.acti(x)
        return x


policy_forms = {"basic": BasicPolicy, "image": ImagePolicy}
