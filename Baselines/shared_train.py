import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from Environments.SelfBreakout.breakout_screen import Screen
from Environments.SelfBreakout.gym_wrapper import BreakoutGymWrapper

import gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
    Collector
)

from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy


def make_breakout_env(args):
    return BreakoutGymWrapper(Screen(breakout_variant=args.variant, angle_mode=args.use_angle_mode), args)

def make_breakout_env_fn(args):
    def create():
        return make_breakout_env(args)

    return create

""" Overloaded version of tianshou collector to take recorded video images
"""
class VideoCollector(Collector):
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
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
            .. note::
                One and only one collection number specification is permitted, either
                ``n_step`` or ``n_episode``.
            :return: A dict including the following keys
                * ``n/ep`` collected number of episodes.
                * ``n/st`` collected number of steps.
                * ``rews`` array of episode reward over collected episodes.
                * ``lens`` array of episode length over collected episodes.
                * ``idxs`` array of episode start index in buffer over collected episodes.
                * ``rew`` mean of episodic rewards.
                * ``len`` mean of episodic lengths.
                * ``rew_std`` standard error of episodic rewards.
                * ``len_std`` standard error of episodic lengths.
            """
            assert not self.env.is_async, "Please use AsyncCollector if using async venv."
            if n_step is not None:
                assert n_episode is None, (
                    f"Only one of n_step or n_episode is allowed in Collector."
                    f"collect, got n_step={n_step}, n_episode={n_episode}."
                )
                assert n_step > 0
                if not n_step % self.env_num == 0:
                    warnings.warn(
                        f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                        "which may cause extra transitions collected into the buffer."
                    )
                ready_env_ids = np.arange(self.env_num)
            elif n_episode is not None:
                assert n_episode > 0
                ready_env_ids = np.arange(min(self.env_num, n_episode))
                self.data = self.data[:min(self.env_num, n_episode)]
            else:
                raise TypeError(
                    "Please specify at least one (either n_step or n_episode) "
                    "in AsyncCollector.collect()."
                )

            start_time = time.time()

            step_count = 0
            episode_count = 0
            episode_rews = []
            episode_lens = []
            episode_start_indices = []
            saved_images = []
            drops = []
            assessment = []

            while True:
                assert len(self.data) == len(ready_env_ids)
                # restore the state: if the last state is None, it won't store
                last_state = self.data.policy.pop("hidden_state", None)

                # get the next action
                if random:
                    self.data.update(
                        act=[self._action_space[i].sample() for i in ready_env_ids]
                    )
                else:
                    if no_grad:
                        with torch.no_grad():  # faster than retain_grad version
                            # self.data.obs will be used by agent to get result
                            result = self.policy(self.data, last_state)
                    else:
                        result = self.policy(self.data, last_state)
                    # update state / act / policy into self.data
                    policy = result.get("policy", Batch())
                    assert isinstance(policy, Batch)
                    state = result.get("state", None)
                    if state is not None:
                        policy.hidden_state = state  # save state into buffer
                    act = to_numpy(result.act)
                    if self.exploration_noise:
                        act = self.policy.exploration_noise(act, self.data)
                    self.data.update(policy=policy, act=act)

                # get bounded and remapped actions first (not saved into buffer)
                action_remap = self.policy.map_action(self.data.act)
                # step in env
                result = self.env.step(action_remap, ready_env_ids)  # type: ignore
                obs_next, rew, done, info = result

                assert(len(info) == 1)

                if info[0]['assessment'] <= -1000:
                    info[0]['assessment'] = info[0]['assessment'] + 1000
                    drops.append(1)
                    assessment.append(info[0]["assessment"])
                elif info[0]["assessment"] > -900:
                    assessment.append(info[0]["assessment"])
                    drops.append(0)
                else:
                    drops.append(1)

                self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
                if self.preprocess_fn:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            done=self.data.done,
                            info=self.data.info,
                            policy=self.data.policy,
                            env_id=ready_env_ids,
                        )
                    )

                if render:
                    saved_images.append(self.env.render()[0])
                    if render > 0 and not np.isclose(render, 0):
                        time.sleep(render)

                # add data into the buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.data, buffer_ids=ready_env_ids
                )

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
                        obs_reset = self.preprocess_fn(
                            obs=obs_reset, env_id=env_ind_global
                        ).get("obs", obs_reset)
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

                if (n_step and step_count >= n_step) or \
                        (n_episode and episode_count >= n_episode):
                    break

            # generate statistics
            self.collect_step += step_count
            self.collect_episode += episode_count
            self.collect_time += max(time.time() - start_time, 1e-9)

            if n_episode:
                self.data = Batch(
                    obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
                )
                self.reset_env()

            if episode_count > 0:
                rews, lens, idxs = list(
                    map(
                        np.concatenate,
                        [episode_rews, episode_lens, episode_start_indices]
                    )
                )
                rew_mean, rew_std = rews.mean(), rews.std()
                len_mean, len_std = lens.mean(), lens.std()
            else:
                rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
                rew_mean = rew_std = len_mean = len_std = 0

            return {
                "n/ep": episode_count,
                "n/st": step_count,
                "rews": rews,
                "lens": lens,
                "idxs": idxs,
                "rew": rew_mean,
                "len": len_mean,
                "rew_std": rew_std,
                "len_std": len_std,
                "saved_images" : saved_images,
                "drops" : np.array(drops),
                "assessment" : np.array(assessment),
            }
