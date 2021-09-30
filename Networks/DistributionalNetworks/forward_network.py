import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork, PointNetwork, PairNetwork, BasicResNetwork, pytorch_model
from Networks.basis_expansion import MultiBasisExpansion

class DiagGaussianForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = BasicMLPNetwork(**kwargs)
        self.std = BasicMLPNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.model = [self.mean, self.std]
        self.base_variance = kwargs['base_variance']

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()


    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        # print(x)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + self.base_variance

class DiagGaussianForwardPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["aggregate_final"] = False # don't aggregate the last layer, just convert the dim
        kwargs["output_dim"] = kwargs["object_dim"] # output the state of the object
        self.mean = PairNetwork(**kwargs) # literally only these two lines are different so there should be a way to compress this...
        self.std = PairNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.model = [self.mean, self.std]
        self.base_variance = kwargs['base_variance']

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()


    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + self.base_variance

class BinaryPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["aggregate_final"] = False # don't aggregate the last layer, just convert the dim
        kwargs["output_dim"] = 1 # output the state of the object
        self.binaries = PairNetwork(**kwargs) # literally only these two lines are different so there should be a way to compress this...
        self.normalization = kwargs['normalization_function']
        self.model = [self.binaries]
        self.base_variance = kwargs['base_variance']

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()


    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        return torch.sigmoid(self.binaries(x))

class OutputPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs["aggregate_final"] = False # don't aggregate the last layer, just convert the dim
        kwargs["output_dim"] = kwargs["num_outputs"] # output the state of the object
        self.binaries = PairNetwork(**kwargs) # literally only these two lines are different so there should be a way to compress this...
        self.normalization = kwargs['normalization_function']
        self.model = [self.binaries]
        self.base_variance = kwargs['base_variance']
        self.value = kwargs['value']

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()


    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        return torch.tanh(self.binaries(x))

class FlatForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.mean = BasicMLPNetwork(**kwargs)
        self.mean = BasicResNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.model = [self.mean]
        self.value=kwargs["value"]

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        return torch.tanh(self.mean(x)) * self.value

class FlatBasisForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.mean = BasicMLPNetwork(**kwargs)
        self.basis_expansion = MultiBasisExpansion(**kwargs)
        kwargs["num_inputs"] = self.basis_expansion.expanded_dim
        self.mean = BasicResNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.model = [self.mean]
        self.value=kwargs["value"]

        self.train()
        self.reset_parameters()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()
        self.basis_expansion.cpu()
    
    def cuda(self):
        super().cuda()
        self.normalization.cuda()
        self.basis_expansion.cuda()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        x = self.basis_expansion(x)
        return torch.tanh(self.mean(x)) * self.value

class FlatDisjointForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subnets = list()
        kwargs["num_outputs"] = 1
        for i in range(self.num_outputs):
            self.subnets.append(forward_nets[kwargs["subnet_type"](**kwargs)])
        self.subnets = nn.ModuleList(self.subnets)

    def forward(self, x):
        outputs = list()
        for net in self.subnets:
            outputs.append(net(x))
        return torch.concatenate(outputs, dim=-1)


forward_nets = {"basic": DiagGaussianForwardNetwork, "pair": DiagGaussianForwardPairNetwork, 'flat': FlatForwardNetwork,
                 'bin': BinaryPairNetwork, 'out': OutputPairNetwork, 'basis': FlatBasisForwardNetwork}