from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.discrete import NoisyLinear

from Networks.network import Basic2DConvNetwork, PairNetwork
from Networks.tianshou_networks import PixelNetwork

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        obs_dim,
        action_shape,
        device,
        observation_info,
    ) -> None:
        super().__init__()
        self.device = device

        self.obs_dim = obs_dim
        self.output_dim = action_shape

        observation_type = observation_info['observation-type']

        if observation_type in ["delta", "full-encoding"]:
            self.obs_dim = self.obs_dim[0]
            self.net = nn.Sequential(
                nn.Linear(self.obs_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.output_dim)
            )
        elif observation_type == "image":
            # kwargs = { 'hidden_sizes' : [32, 64, 128],
            #            'kernel' : 5,
            #            'use_layer_norm' : True,
            #            'input_dims' : self.obs_dim,
            #            'stride' : 1,
            #            'padding' : 0,
            #            'output_dim' : self.output_dim,
            #            'reduce': True,
            #            'include_last' : False,
            #            'num_inputs' : 0,
            #            'num_outputs' : self.output_dim,
            #            'init_form': "xnorm",
            #            'activation' : 'relu'
            #            }

            # self.net = Basic2DConvNetwork(**kwargs)

            kwargs = {
                "hidden_sizes" : [32, 64, 64, 128],
                "cuda" : True,
                "bound_output" : 0,
                "num_outputs" : self.output_dim
            }

            self.net = PixelNetwork(**kwargs)
        elif observation_type == "multi-block-encoding":
            first_obj_dim = observation_info['first-obj-dim']
            object_dim = observation_info['obj-dim']

            kwargs = { 'hidden_sizes' : [128, 128, 128],
                       'first_obj_dim' : first_obj_dim,
                       'object_dim' : object_dim,
                       'aggregate_final' : True,
                       'use_layer_norm' : False,
                       'post_dim' : 0,
                       'output_dim' : self.output_dim,
                       'num_inputs' : 0,
                       'num_outputs' : self.output_dim,
                       'init_form': "xnorm",
                       'activation' : 'relu',
            }

            self.net = PairNetwork(**kwargs)


    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state

class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        obs_dim,
        action_shape,
        num_atoms = 51,
        noisy_std = 0.5,
        device = "cpu",
        is_dueling = True,
        is_noisy = True,
        observation_info = {"observation-type" : "delta"},
    ) -> None:
        super().__init__(obs_dim, action_shape, device, observation_info)
        self.action_num = action_shape
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        x, state = super().forward(x)
        q = self.Q(x)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(x)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        y = logits.softmax(dim=2)
        return y, state

class SACNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        output_dim,
        device,
        observation_info,
    ) -> None:
        super().__init__()
        self.device = device

        self.obs_dim = obs_dim
        self.output_dim = output_dim

        observation_type = observation_info['observation-type']

        if observation_type in ["delta", "full-encoding"]:
            self.obs_dim = self.obs_dim[0]
            self.net = nn.Sequential(
                nn.Linear(self.obs_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.output_dim)
            )
        elif observation_type == "image":
            kwargs = {
                "hidden_sizes" : [32, 64, 64, 128],
                "cuda" : True,
                "bound_output" : 0,
                "num_outputs" : self.output_dim
            }

            self.net = PixelNetwork(**kwargs)
        elif observation_type == "multi-block-encoding":
            first_obj_dim = observation_info['first-obj-dim']
            object_dim = observation_info['obj-dim']

            kwargs = { 'hidden_sizes' : [128, 128, 128],
                       'first_obj_dim' : first_obj_dim,
                       'object_dim' : object_dim,
                       'aggregate_final' : True,
                       'use_layer_norm' : False,
                       'post_dim' : 0,
                       'output_dim' : self.output_dim,
                       'num_inputs' : 0,
                       'num_outputs' : self.output_dim,
                       'init_form': "xnorm",
                       'activation' : 'relu',
            }

            self.net = PairNetwork(**kwargs)


    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""

        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state
