import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork, PointNetwork, PairNetwork, pytorch_model

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
        self.base_variance = .0001

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
        # x = self.normalization(x)
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + 1e-2


forward_nets = {"basic": DiagGaussianForwardNetwork, "pair": DiagGaussianForwardPairNetwork}