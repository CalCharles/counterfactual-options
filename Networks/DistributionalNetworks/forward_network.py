import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork, pytorch_model

class DiagGaussianForwardNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = BasicMLPNetwork(**kwargs)
        self.std = BasicMLPNetwork(**kwargs)
        self.normalization = kwargs['normalization_function']
        self.model = [self.mean, self.std]

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
        return torch.tanh(self.mean(x)), torch.sigmoid(self.std(x)) + 1e-2

forward_nets = {"base": DiagGaussianForwardNetwork}