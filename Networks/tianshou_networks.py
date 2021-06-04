import torch
import torch.nn as nn
import numpy as np
from Networks.network import pytorch_model, BasicMLPNetwork, PointNetwork
import torch.nn.functional as F

class TSNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.iscuda = kwargs["cuda"]
        self.output_dim = int(np.prod(kwargs["num_outputs"]))

    def cuda(self):
        super().cuda()
        self.iscuda = True

    def cpu(self):
        super().cpu()
        self.iscuda = False

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = pytorch_model.wrap(obs, dtype=torch.float, cuda=self.iscuda)
        batch = obs.shape[0]
        obs = obs.reshape(batch, -1)
        logits = self.model(obs)
        return logits, state


class BasicNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_shape = kwargs["num_inputs"]
        print(self.output_dim)
        kwargs["num_outputs"] = self.output_dim
        self.model = BasicMLPNetwork(**kwargs)
        # nn.Sequential(
        #     *([nn.Linear(np.prod(state_shape), kwargs["hidden_sizes"][0]), nn.ReLU(inplace=True)] + 
        #       sum([[nn.Linear(kwargs["hidden_sizes"][i-1], kwargs["hidden_sizes"][i]), nn.ReLU(inplace=True)] for i in range(len(kwargs["hidden_sizes"]))], list()) + 
        #     [nn.Linear(kwargs["hidden_sizes"][-1], self.output_dim)])
        # )
        if self.iscuda:
            self.cuda()

class PointPolicyNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_shape = kwargs["num_inputs"]
        print(self.output_dim)
        kwargs["aggregate_final"] = True
        kwargs["num_outputs"] = self.output_dim
        self.model = PairNetwork(**kwargs)
        # nn.Sequential(
        #     *([nn.Linear(np.prod(state_shape), kwargs["hidden_sizes"][0]), nn.ReLU(inplace=True)] + 
        #       sum([[nn.Linear(kwargs["hidden_sizes"][i-1], kwargs["hidden_sizes"][i]), nn.ReLU(inplace=True)] for i in range(len(kwargs["hidden_sizes"]))], list()) + 
        #     [nn.Linear(kwargs["hidden_sizes"][-1], self.output_dim)])
        # )
        if self.iscuda:
            self.cuda()



class PixelNetwork(TSNet): # no relation to pixelnet, just a network that operates on pixels
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: assumes images of size 84x84, make general
        self.num_stack = 4
        factor = self.factor
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_stack, kwargs["hidden_sizes"][0], 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(kwargs["hidden_sizes"][0], kwargs["hidden_sizes"][1], 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(kwargs["hidden_sizes"][1], kwargs["hidden_sizes"][2], 3, stride=1), nn.ReLU(inplace=True),
        )
        self.model = nn.Sequential(
            nn.Linear(2 * kwargs["hidden_sizes"][2] * self.viewsize * self.viewsize, kwargs["hidden_sizes"][3]), nn.ReLU(inplace=True),
            nn.Linear(kwargs["hidden_sizes"][3], kwargs["num_outputs"])
        )

        if self.iscuda:
            self.cuda()

        # self.conv1 = nn.Conv2d(self.num_stack, 2 * factor, 8, stride=4)
        # self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        # self.conv3 = nn.Conv2d(4 * factor, 2 * factor, 3, stride=1)
        # self.viewsize = 7
        # self.reshape = kwargs["reshape"]
        # # if self.args.post_transform_form == 'none':
        # #     self.linear1 = None
        # #     self.insize = 2 * self.factor * self.viewsize * self.viewsize
        # #     self.init_last(self.num_outputs)
        # # else:
        # self.linear1 = nn.Linear(2 * factor * self.viewsize * self.viewsize, self.insize)
        # self.layers.append(self.linear1)
        # self.layers.append(self.conv1)
        # self.layers.append(self.conv2)
        # self.layers.append(self.conv3)
        self.reset_parameters()

    def forward(self, obs, state=None, info={}):
        batch_size= obs.shape[0]
        instate = self.conv(obs / 255.0)
        instate.view(batch_size, -1)
        return super().forward(instate, state=state, info={})

class GridWorldNetwork(TSNet):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        N = 20 # hardcoded at the moment
        H, W = N, N
        self.H, self.W = H, W
        self.C = 3
        self.num_stack = 3
        self.hs = kwargs["hidden_sizes"]
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.num_stack, self.hs[0], 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        #     nn.Conv2d(self.hs[0], self.hs[1], 3, stride=1, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(self.hs[1], self.hs[2], 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        # )
        self.conv1 = torch.nn.Conv2d(in_channels=self.num_stack,out_channels=self.hs[0],kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.hs[0],out_channels=self.hs[1],kernel_size=3,stride=1,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.hs[1],out_channels=self.hs[2],kernel_size=3,stride=1,padding=1)
        self.fc1 = torch.nn.Linear(int(self.hs[2]*H*W/16),self.hs[3])
        self.fc2 = torch.nn.Linear(self.hs[3],kwargs["num_outputs"])


        self.model = nn.Sequential(
            nn.Linear(int(kwargs["hidden_sizes"][2] * H * W / 16), kwargs["hidden_sizes"][3]), nn.ReLU(inplace=True),
            nn.Linear(kwargs["hidden_sizes"][3], kwargs["num_outputs"])
        )
        self.mid_size = int(kwargs["hidden_sizes"][2] * H * W / 16)
        self.preprocess = kwargs["preprocess"]
        # self.fc2 = torch.nn.Linear(564,self.insize)
        if self.iscuda:
            self.cuda()

    def forward(self, obs, state=None, info={}):
        # needs to ensure parameter is used correctly:
        # x[:,:,:,2] = p.reshape(-1,*self.reshape[:-1])
        # x = x.transpose(3,2).transpose(2,1)
        if self.preprocess: obs = self.preprocess(obs)
        if not isinstance(obs, torch.Tensor):
            obs = pytorch_model.wrap(obs, dtype=torch.float, cuda=self.iscuda)
        batch_size = obs.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(obs)),2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)),2)

        x = x.reshape(batch_size,int(self.hs[2]*self.H*self.W/16))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # instate = self.conv(obs)



        # print(instate.shape)
        # instate = instate.reshape(batch_size, self.mid_size)
        # return super().forward(instate, state=state, info={})
        return x, state

networks = {'basic': BasicNetwork, 'pixel': PixelNetwork, 'grid': GridWorldNetwork, 'point': PointPolicyNetwork}