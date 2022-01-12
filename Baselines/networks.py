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

class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_shape: int,
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: Union[str, int, torch.device] = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(obs_dim, action_shape, device, features_only=True)
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
