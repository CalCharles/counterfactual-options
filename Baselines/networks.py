from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.discrete import NoisyLinear

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_shape: int,
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        # self.net = nn.Sequential(
        #     nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        #     nn.Flatten()
        # )

        self.obs_dim = obs_dim
        self.output_dim = action_shape
        # with torch.no_grad():
        #    self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.output_dim)
        )

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x), state
