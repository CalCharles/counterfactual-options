import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from Environments.SelfBreakout.breakout_screen import Screen
from Environments.SelfBreakout.gym_wrapper import BreakoutGymWrapper
from DistributionalModels.InteractionModels.samplers import BreakoutRandomSampler
from DistributionalModels.DatasetModels.dataset_model import DatasetModel
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel

import gym
import numpy as np
import torch
import tqdm
from collections import defaultdict

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
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, TensorboardLogger, MovAvg, tqdm_config
from torch.utils.tensorboard import SummaryWriter

class DummyOptionNode(object):
    def __init__(self):
        self.num_params = 0
        self.name = "dummy"

def make_breakout_env(args):
    if args.variant == 'proximity':
        screen = Screen(breakout_variant=args.variant, angle_mode=args.use_angle_mode, drop_stopping=True)
        env_model = BreakoutEnvironmentModel(screen)

        dummy_dataset_model = DatasetModel(**{ 'option_node' : DummyOptionNode() })
        sampler = BreakoutRandomSampler(**{ 'environment_model' : env_model, 'init_state' : screen.get_state(), 'dataset_model' : dummy_dataset_model, 'no_combine_param_mask' : True })
        screen.sampler = sampler

        return BreakoutGymWrapper(screen, args)
    else:
        return BreakoutGymWrapper(Screen(breakout_variant=args.variant, angle_mode=args.use_angle_mode, drop_stopping=False), args)

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
        episode_limit: int = -1,
        timeout_penalty: int = -1
    ):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.episode_limit = episode_limit
        self.timeout_penalty = timeout_penalty

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
            curr_ep_step_count = 0
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

                step_count += len(ready_env_ids)
                curr_ep_step_count += len(ready_env_ids)

                if np.any(done) or (self.episode_limit >= 0 and curr_ep_step_count >= self.episode_limit):
                    if self.episode_limit >= 0 and curr_ep_step_count >= self.episode_limit:
                        assessment.append(self.timeout_penalty)
                        done[0] = True
                    elif info[0]['assessment'] <= -1000:
                        info[0]['assessment'] = info[0]['assessment'] + 1000
                        drops.append(1)
                        while info[0]['assessment'] <= -1000:
                            drops.append(1)
                            info[0]['assessment'] = info[0]['assessment'] + 1000
                        assessment.append(info[0]["assessment"])
                    elif info[0]["assessment"] > -900:
                        assessment.append(info[0]["assessment"])
                        drops.append(0)
                    else:
                        info[0]['assessment'] = info[0]['assessment'] + 1000
                        drops.append(1)
                        assessment.append(info[0]["assessment"])

                    curr_ep_step_count = 0

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
                if np.any(done):
                    curr_ep_step_count = 0
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

# Modified offpolicy_trainer to track # training episodes rather than only training steps
def offpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Optional[Collector],
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.
    The "step" in trainer means an environment step (a.k.a. transition).
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatedly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.
    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    start_epoch, env_step, env_ep, gradient_step = 0, 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, env_ep, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_in_train = test_in_train and (
        train_collector.policy == policy and test_collector is not None
    )

    if test_collector is not None:
        test_c: Collector = test_collector  # for mypy
        test_collector.reset_stat()
        test_result = test_episode(
            policy, test_c, test_fn, start_epoch, episode_per_test, logger, env_step,
            reward_metric
        )
        best_epoch = start_epoch
        best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    if save_fn:
        save_fn(policy)

    last_print = 0
    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)

                if env_step - last_print >= 1000:
                    last_print = env_step
                    print(f'At time step {env_step}')

                result = train_collector.collect(n_step=step_per_collect)
                if result["n/ep"] > 0 and reward_metric:
                    rew = reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())

                if result["rew"] != 0.0:
                    print(f'Nonzero reward {result["rew"]}')

                env_step += int(result["n/st"])
                env_ep += int(result["n/ep"])

                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result['rew'] if result["n/ep"] > 0 else last_rew
                last_len = result['len'] if result["n/ep"] > 0 else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_c, test_fn, epoch, episode_per_test, logger,
                            env_step
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, env_ep, save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"]
                            )
                        else:
                            policy.train()
                for _ in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(batch_size, train_collector.buffer)
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        logger.save_data(epoch, env_step, gradient_step, env_ep, save_checkpoint_fn)
        # test
        if test_collector is not None:
            test_result = test_episode(
                policy, test_c, test_fn, epoch, episode_per_test, logger, env_step,
                reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if best_epoch < 0 or best_reward < rew:
                best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
                if save_fn:
                    save_fn(policy)
            if verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                    f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
                )
            if stop_fn and stop_fn(best_reward):
                break

    if test_collector is None and save_fn:
        save_fn(policy)

    if test_collector is None:
        return gather_info(start_time, train_collector, None, 0.0, 0.0)
    else:
        return gather_info(
            start_time, train_collector, test_collector, best_reward, best_reward_std
        )

class ModifiedLogger(TensorboardLogger):
    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ) -> None:
        super().__init__(writer, train_interval, test_interval, update_interval, save_interval)

    def save_data(
            self,
            epoch : int,
            env_step : int,
            gradient_step : int,
            env_ep : int,
            save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
            ):
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step, env_ep)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step", gradient_step,
                {"save/gradient_step": gradient_step}
            )
