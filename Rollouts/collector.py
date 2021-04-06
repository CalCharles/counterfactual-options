import numpy as np
import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer
from Options.option import Option
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class OptionCollector(Collector): # change to line  (update batch) and line 12 (param parameter), the rest of parameter handling must be in policy
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        option: Option = None,
    ) -> None:
        self.option = option
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)


    def reset_env(self, keep_statistics: bool = False):
        obs = self.env.reset()
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get("obs", obs)
        if self.option: 
            self.data.full_state = obs
            obs = self.option.get_state(obs)
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)


    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        param: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param np.ndarray param: the parameter which is the goal for the option
        :param Option option: the option that is being trained (which as attribute
            "policy")

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            # I believe both can be active at once.... we will see though....
            # assert n_episode is None, ( 
            #     f"Only one of n_step or n_episode is allowed in Collector."
            #     f"collect, got n_step={n_step}, n_episode={n_episode}."
            # )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        if n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        if not (n_step or n_episode):
            raise TypeError("Please specify at least one (either n_step or n_episode) "
                            "in AsyncCollector.collect().")

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        if param is not None: self.data.param = [param]
        if 'state_chain' not in self.data: state_chain = None # support for RNNs while handling feedforward
        else: state_chain = self.data.state_chain
        while True:
            # print(step_count, self.data, len(self.data), len(ready_env_ids))
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                if self.option: 
                    action_chain, result, state_chain = self.option.sample_action_chain(self.data, state_chain, random=True)
                else:
                    action_chain=[self._action_space[i].sample() for i in ready_env_ids] #line makes no sense
                self.data.update(act=[[action_chain[-1]]], true_action=[action_chain[0]], action_chain=action_chain)
                if state_chain is not None: self.data.update(state_chain = state_chain)
                act = self.data.true_action
            else:
            # Lines altered from collector
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        if self.option: action_chain, result, state_chain = self.option.sample_action_chain(self.data, state_chain)
                        else: result = self.policy(self.data, last_state)
                else:
                    if self.option: action_chain, result, state_chain = self.option.sample_action_chain(self.data, state_chain)
                    else: result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    # print(len(state.shape) == 1, len(result.act.shape) == 2)
                    if len(state.shape) == 1 and len(result.act.shape) == 2: state = [state] 
                    policy.hidden_state = state  # save state into buffer
                    self.data.update(state_chain=state_chain)
                if self.option:
                    act = to_numpy(action_chain[-1])
                    self.data.update(true_action=[action_chain[0]], action_chain=action_chain)
                else: act = to_numpy(result.act)
                # if self.exploration_noise:
                #     act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=[act])
            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.true_action) # the problem is here
            # step in env
            # print(action_remap, self.data.act, action_chain)
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)
            next_full_state = obs_next[0] # only handling one environment


            true_reward = rew
            dones, rewards, true_done, true_reward = [], [], done, rew
            if self.option:
                self.option.step(next_full_state, action_chain)
                obs_next = self.option.get_state(next_full_state)
                self.option.step_timer(true_done)
                dones, rewards = self.option.terminate_reward(obs_next, param, action_chain)
                done, rew = dones[-1], rewards[-1]
            self.data.update(obs_next=[obs_next], next_full_state=[next_full_state], rew=[rew], done=[done], info=info, param=[param], dones=dones, rewards=rewards) # edited
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                ))

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            if self.policy.collect: self.policy.collect(self.data)
            # print(ready_env_ids)
            # print(self.data)
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids)
            # line altered from collector ABOVE

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                obs_reset = self.env.reset(env_ind_global)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(obs=obs_reset).get("obs", obs_reset)
                self.data.obs_next[env_ind_local] = obs_reset
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next
            self.data.full_state = self.data.next_full_state

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(obs={}, act={}, rew={}, done={},
                              obs_next={}, info={}, policy={}, full_state={}, param={}, next_full_state={})
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(map(
                np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "done": int(n_episode is not None and episode_count >= n_episode),
        }
