import numpy as np
import gym
import time
import torch
import warnings
import cv2
import copy
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from argparse import Namespace
from collections import deque
from file_management import printframe, saveframe
from Networks.network import pytorch_model

from Rollouts.param_buffer import ParamReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer
from Options.option import Option
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from visualizer import visualize

def print_shape(batch, prefix=""):
    print(prefix, {n: batch[n].shape for n in batch.keys() if type(batch[n]) == np.ndarray})

class TemporalAggregator():
    def __init__(self):
        self.current_data = Batch()
        self.next_action = False
        self.next_param = False
        self.keep_next = True
        self.temporal_skip = False
        self.ptr = 0

    def reset(self, data):
        self.current_data = copy.deepcopy(data)

    def update(self, data):
        self.current_data = copy.deepcopy(data)

    def aggregate(self, data, buffer, ptr, ready_env_ids):
        # updates "next" values to the current value, and combines dones, rewards
        skipped = False
        if self.keep_next: 
            self.current_data = copy.deepcopy(data)
            self.keep_next = self.temporal_skip
            self.temporal_skip = False
        else: # the reward is already recorded if we copy the input data
            self.current_data.rew += data.rew
            self.current_data.true_reward += data.true_reward
        self.current_data.update(next_full_state = data.next_full_state, next_target=data.next_target, obs_next=data.obs_next, inter_state=data.inter_state)
        self.current_data.done = [np.any(self.current_data.done) + np.any(data.done)] # basically an OR
        self.current_data.terminate = [np.any(self.current_data.terminate) + np.any(data.terminate)] # basically an OR
        self.current_data.true_done = [np.any(self.current_data.true_done) + np.any(data.true_done)] # basically an OR
        self.current_data.option_resample = data.option_resample
        self.current_data.info["TimeLimit.truncated"] = data.info["TimeLimit.truncated"]
        next_data = copy.deepcopy(self.current_data)
        
        # if we just resampled (meaning temporal extension occurred)
        if (np.any(data.ext_term) # going to resample a new action
            or np.any(data.done)
            or np.any(data.terminate)):
            self.keep_next = True
            if not self.temporal_skip:
                self.ptr, ep_rew, ep_len, ep_idx = buffer.add(
                        next_data, buffer_ids=ready_env_ids)
            else:
                skipped = True
            self.temporal_skip = "TimeLimit.truncated" in self.current_data.info[0] and self.current_data.info[0]["TimeLimit.truncated"] and np.any(data.done)
        return next_data, skipped, self.ptr

class OptionCollector(Collector): # change to line  (update batch) and line 12 (param parameter), the rest of parameter handling must be in policy
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        option: Option = None,
        test: bool = False,
        args: Namespace = None,
    ) -> None:
        self.param_recycle = args.param_recycle # repeat a parameter
        self.option = option
        self.at = 0
        self.full_at = 0 # the pointer for the buffer without temporal extension
        self.test = test
        self.full_buffer = copy.deepcopy(buffer) # TODO: make this conditional on usage?
        self.hit_miss_queue = deque(maxlen=2000) # not sure if this is the best way to record hits, but this records when a target position is reached

        
        # shortcut calling option attributes through option
        self.state_extractor = self.option.state_extractor
        self.get_param = self.option.sampler.get_param # sampler manages either recalling the param, or getting a new one
        self.exploration_noise = self.option.policy.exploration_noise
        self.temporal_aggregator = TemporalAggregator()
        self.ext_reset = self.option.temporal_extension_manager.reset
        self._aggregate = self.temporal_aggregator.aggregate
        self.policy_collect = self.option.policy.collect
        self._done_check = self.option.done_model.done_check
        
        env = DummyVectorEnv([lambda: env])
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)

    def reset(self):
        super().reset()

    def reset_env(self, keep_statistics: bool = False):
        full_state = self.env.reset()
        if self.preprocess_fn:
            full_state = self.preprocess_fn(obs=full_state).get("obs", full_state)
        self._reset_state(0)
        self.option.reset(full_state[0])
        param, mask = self._reset_data(full_state[0])
        self.data.update(obs=[self.state_extractor.get_obs(full_state[0], param)]) # self.option.get_state(obs, setting=self.option.input_setting, param=self.param if self.param is not None else None)
        self.temporal_aggregator.reset(self.data)

    def next_index(self, val):
        return (val + 1) % self.buffer.maxsize

    def _reset_data(self, full_state):
        # ensure that data has the correct: param, obs, full_state, option_resample
        # will always sample a new param
        self.data.update(full_state=[full_state])
        param, mask, _ = self.get_param(full_state, True)
        self.data.update(target=self.state_extractor.get_target(self.data.full_state),
            obs=[self.state_extractor.get_obs(self.data.full_state, param)])
        self.data.update(param=[param], mask = [mask], option_resample=[[True]])
        term_chain = self.option.reset(full_state)
        act, chain, policy_batch, state, masks, resampled = self.option.extended_action_sample(self.data, None, random=False)
        self._policy_state_update(policy_batch)
        self.data.update(terminate=[term_chain[-1]], terminations=term_chain, ext_term=[term_chain[-2]])
        self.option.update(True, self.data["full_state"], act, chain, term_chain, param, masks)
        self.last_resampled_idx = self.full_at
        self.last_resampled_full_state = self.data.full_state
        self.temporal_aggregator.update(self.last_resampled_full_state)
        return param, mask

    def _policy_state_update(self, result):
        # update state / act / policy into self.data
        policy = result.get("policy", Batch())
        assert isinstance(policy, Batch)
        state = result.get("state", None)
        if state is not None:
            policy.hidden_state = state  # save state into buffer
            self.data.update(state_chain=state_chain, policy=policy)

    def perform_reset(self):
        # artificially create term to sample a new param
        # reset the temporal extension manager
        self.data.update(terminate=[True])
        self.ext_reset()

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        n_term: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        visualize_param: str = "",
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
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_step > 0
            ready_env_ids = np.arange(self.env_num)
        if n_episode is not None:
            assert n_episode > 0
        if n_term is not None:
            assert n_term > 0

        start_time = time.time()

        step_count = 0
        episode_count = 0
        true_episode_count = 0
        term_count = 0
        rews = 0
        term_done = False # signifies if the termination was the result of a option terminal state

        while True:
            param, mask, new_param = self.get_param(self.data.full_state[0], self.data.terminate[0])
            if self.test and new_param: print("new param", param)
            state_chain = self.data.state_chain if hasattr(self.data, "state_chain") else None
            if no_grad:
                with torch.no_grad():  # faster than retain_grad version
                    act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, random=random)
            else:
                act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, random=random)
            self._policy_state_update(result)
            self.data.update(true_action=[action_chain[0]], act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled])

            # step in env
            action_remap = self.data.true_action
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)

            # cv2.imshow('state', obs_next[0]['raw_state'])
            # cv2.waitKey(1)

            next_full_state = obs_next[0] # only handling one environment

            true_done, true_reward = done, rew
            # if self.option: # seems pointless to support no-option code
            obs = self.state_extractor.get_obs(self.data.full_state[0], param) # one environment reliance
            obs_next = self.state_extractor.get_obs(next_full_state, param) # one environment reliance
            target = self.state_extractor.get_target(self.data.full_state[0])
            next_target = self.state_extractor.get_target(next_full_state)
            inter_state = self.state_extractor.get_inter(self.data.full_state[0])

            done, rewards, terminations, ext_terms = self.option.terminate_reward_chain(self.data.full_state[0], next_full_state, param, action_chain, mask, masks)
            done, rew, term, ext_term = done, rewards[-1], terminations[-1], ext_terms[-1]
            self.option.update(done, self.data.full_state[0], act, action_chain, terminations, param, masks)

            # update hit-miss values

            # update the current values
            self.data.update(next_target=[next_target], target=[target], inter_state=[inter_state], obs_next=[obs_next], obs = [obs],
                next_full_state=[next_full_state], true_done=true_done, true_reward=true_reward, 
                param=[param], mask = [mask], info = info,
                rew=[rew], done=[done], terminate=[term], ext_term = [ext_term], # all prior are stored, after are not 
                terminations= terminations, rewards=rewards, masks=masks, ext_terms=ext_terms)
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                ))

            # render calls not usually used in our case
            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # we keep a buffer with all of the values
            full_ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(
                self.data, buffer_ids=ready_env_ids)

            next_data, skipped, self.at = self._aggregate(self.data, self.buffer, full_ptr, ready_env_ids)
            if not self.test: self.policy_collect(next_data, self.data, skipped)

            # debugging and visualization
            if self.test: print(self.data.obs.squeeze(), self.data.act.squeeze())
            if len(visualize_param) != 0:
                frame = np.array(self.env.render()).squeeze()
                new_frame = visualize(frame, self.data.target[0], self.param, self.mask)
                if visualize_param != "nosave":
                    saveframe(new_frame, pth=visualize_param, count=self.counter, name="param_frame")
                    self.counter += 1
                printframe(new_frame, waittime=1)

            # collect statistics
            step_count += len(ready_env_ids)

            # update counters
            if np.any(done) or np.any(term):
                episode_count += int(np.any(done))
                term_count += int(np.any(term))
                term_done, timer, true = self._done_check(term, true_done)
            if np.any(true_done):
                true_episode_count += 1
                # if we have a true done, reset the environments and self.data
                self.reset_env()
                full_state_reset = self.data.full_state[0] # set by reset_env
                if self.preprocess_fn:
                    full_state_reset = self.preprocess_fn(obs=full_state_reset).get("obs", full_state_reset)
                self.data.update(next_full_state = [full_state_reset]) # obs_reset
                self._reset_state(0)
                self._reset_data(full_state_reset)

            self.data.full_state = self.data.next_full_state

            # controls termination
            if (n_step and step_count >= n_step):
                break
            if (n_episode and episode_count >= n_episode):
                break
            if (n_term and term_count >= n_term):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        rews = np.array(rews)

        return { # TODO: some of these don't return valid values
            "n/ep": episode_count,
            "n/tr": term_count,
            "n/st": step_count,
            "rews": rews,
            "terminate": term_done
        }
