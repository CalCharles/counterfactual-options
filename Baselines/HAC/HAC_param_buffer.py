from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer



class ParamPriorityReplayBuffer(PrioritizedReplayBuffer):
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "param", "next_target", "mapped_act", "gamma")

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indice = self.sample_index(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indice = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indice, "obs")
        if self._save_obs_next:
            obs_next = self.get(indice, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indice), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indice],
            rew=self.rew[indice],
            done=self.done[indice],
            obs_next=obs_next,
            info=self.get(indice, "info", Batch()),
            param = self.param[indice], # all below lines differ from prioritized replay buffer to handle option values
            next_target=self.next_target[indice],
            mapped_act = self.mapped_act[indice],
            gamma=self.gamma[indice],
        )
