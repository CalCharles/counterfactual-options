import torch
import torch.nn as nn
import numpy as np
from Networks.network import pytorch_model

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
        obs.view(batch, -1)
        logits = self.model(obs)
        return logits, state


class BasicNetwork(TSNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_shape = kwargs["num_inputs"]
        print(self.output_dim)
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), kwargs["hidden_sizes"][0]), nn.ReLU(inplace=True),
            nn.Linear(kwargs["hidden_sizes"][0], kwargs["hidden_sizes"][1]), nn.ReLU(inplace=True),
            nn.Linear(kwargs["hidden_sizes"][1], kwargs["hidden_sizes"][2]), nn.ReLU(inplace=True),
            nn.Linear(kwargs["hidden_sizes"][2], self.output_dim)
        )
        if self.iscuda:
            self.cuda()



class PixelNetwork(TSNet): # no relation to pixelnet, just a network that operates on pixels
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: assumes images of size 84x84, make general
        self.num_stack = 4
        factor = self.factor
        nn.Sequential(
            nn.Conv2d(self.num_stack, kwargs["hidden_sizes"][0], 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(kwargs["hidden_sizes"][0], kwargs["hidden_sizes"][1], 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(kwargs["hidden_sizes"][1], kwargs["hidden_sizes"][2], 3, stride=1), nn.ReLU(inplace=True),
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
        if self.reshape[0] != -1:
            inputs = inputs.reshape(-1, *self.reshape)
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