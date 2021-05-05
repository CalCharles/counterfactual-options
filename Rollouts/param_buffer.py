from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np

from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer

class ParamReplayBuffer(ReplayBuffer):
    # obs, obs_next  is the observation form as used by the highest level option (including param, and relative state, if used)
    # act is the action of the highest level option
    # rew is the reward of the highest level option
    # done is the termination of the highest level option
    # param is the param used at the time of input
    # target, next target is the state of the target object, used for reward and termination
    # true_reward, true_done are the actual dones and rewards
    # option_terminate is for temporal extension, stating if the last object terminated
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy", "param", "target", "next_target", "true_reward", "true_done", "option_resample", "mapped_act")

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
            policy=self.get(indice, "policy", Batch()),
            param = self.param[indice], # only these three lines differ from getitem in ReplayBuffer
            target = self.target[indice], # only these three lines differ from getitem in ReplayBuffer
            next_target=self.next_target[indice], # only these three lines differ from getitem in ReplayBuffer
            true_reward=self.true_reward[indice],
            true_done = self.true_done[indice],
            option_resample = self.option_resample[indice],
            mapped_act = self.mapped_act[indice]
        )

class ParamPriorityReplayBuffer(PrioritizedReplayBuffer): # not using double inheritance so exactly the same as above.
    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy", "param", "target", "next_target", "true_reward", "true_done", "option_resample", "mapped_act")

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
            policy=self.get(indice, "policy", Batch()),
            param = self.param[indice], # only these three lines differ from getitem in ReplayBuffer
            target = self.target[indice], # only these three lines differ from getitem in ReplayBuffer
            next_target=self.next_target[indice], # only these three lines differ from getitem in ReplayBuffer
            true_reward=self.true_reward[indice],
            true_done = self.true_done[indice],
            option_resample = self.option_resample[indice],
            mapped_act = self.mapped_act[indice]
        )