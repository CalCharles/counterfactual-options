import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks.network import Network, BasicMLPNetwork, PointNetwork, PairNetwork, pytorch_model

class InteractionNetwork(BasicMLPNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_linear = nn.Linear(self.num_outputs, self.num_outputs)        
        self.normalization = kwargs['normalization_function']
        #TODO: model does not have final linear
        self.train()
        self.reset_parameters()
        
    def cuda(self):
        super().cuda()
        self.normalization.cuda()

    def cpu(self):
        super().cpu()
        self.normalization.cpu()

    def forward(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        v = super().forward(x)
        v = self.final_linear(v)
        v = torch.sigmoid(v)
        return v

class InteractionPairNetwork(PairNetwork):
    def __init__(self, **kwargs):
        kwargs["aggregate_final"] = False
        super().__init__(**kwargs)
        # self.final_linear = nn.Linear(self.num_outputs, self.num_outputs)        
        self.normalization = kwargs['normalization_function']
        #TODO: model does not have final linear
        self.train()
        self.reset_parameters()

    def instance_labels(self, x):
        x = pytorch_model.wrap(x, cuda=self.iscuda)
        x = self.normalization(x)
        v = super().forward(x)
        v = torch.sigmoid(v)
        return v
        
    def forward(self, x):
        v = self.instance_labels(x)
        v, a = torch.max(v, dim=1, keepdim=True)
        # v = self.final_linear(v)
        # v = torch.sigmoid(v)
        return v


interaction_nets = {"basic": InteractionNetwork, 'pair': InteractionPairNetwork}