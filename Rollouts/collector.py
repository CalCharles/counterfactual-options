import numpy as np
import gym
import time
import os
import torch
import warnings
import cv2
import copy
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from argparse import Namespace
from collections import deque
from file_management import printframe, saveframe, action_toString
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
    def __init__(self, sum_reward=False, only_termination=False):
        self.current_data = Batch()
        self.next_action = False
        self.next_param = False
        self.keep_next = True
        self.temporal_skip = False
        self.last_true_done = False
        self.ptr = 0
        self.sum_reward = sum_reward # sums reward for the length of the trajectory
        self.time_counter = 0 # counts the number of time steps in the temporal extension
        self.only_termination = only_termination # only samples when there is a termination of the current option

    def reset(self, data):
        self.current_data = copy.deepcopy(data)
        self.keep_next = True
        self.time_counter = 0

    def update(self, data):
        self.current_data = copy.deepcopy(data)

    def aggregate(self, data, buffer, ptr, ready_env_ids):
        # updates "next" values to the current value, and combines dones, rewards
        added = False
        skipped = False
        if self.keep_next: 
            self.current_data = copy.deepcopy(data)
            # print("keeping:", self.current_data.true_done, self.current_data.obs, self.current_data.full_state["factored_state"], self.current_data.next_full_state["factored_state"])
            self.current_data.rew = [self.current_data.rew.squeeze().astype(float)] # TODO: hack to make sure that reward is a regular shape
            self.current_data.true_reward= self.current_data.true_reward.astype(float)
            self.keep_next = self.temporal_skip or self.last_true_done # if the environment just reset, we will have an obs from the old environment, so keep the next
        else: # the reward is already recorded if we copy the input data
            if self.sum_reward:
                self.current_data.rew += data.rew.squeeze()
                self.current_data.true_reward += data.true_reward.squeeze()
            else:
                self.current_data.rew = [data.rew.squeeze().astype(float)]
                self.current_data.true_reward = [data.true_reward.squeeze().astype(float)]
        self.current_data.update(next_full_state = data.next_full_state, next_target=data.next_target, obs_next=data.obs_next, inter_state=data.inter_state)
        self.current_data.done = [np.any(self.current_data.done) + np.any(data.done)] # basically an OR
        self.current_data.terminate = [np.any(self.current_data.terminate) + np.any(data.terminate)] # basically an OR
        self.current_data.true_done = [np.any(self.current_data.true_done) + np.any(data.true_done)] # basically an OR
        self.current_data.option_resample = data.option_resample
        self.current_data.info["TimeLimit.truncated"] = data.info["TimeLimit.truncated"] if "TimeLimit.truncated" in data.info else False
        self.current_data.update(time=[self.time_counter])
        self.current_data.inter = [max(data.inter[0], self.current_data.inter[0])]
        next_data = copy.deepcopy(self.current_data)
        self.last_true_done = next_data.true_done
        # if we just resampled (meaning temporal extension occurred)
        added = False
        # print(data.act, data.rew, data.ext_term, self.temporal_skip, data.mapped_act, data.param, data.target)
        # if (np.any(data.ext_term) and not np.any(data.terminate)):
        #     print("bouncing", next_data.act, next_data.mapped_act, next_data.full_state["factored_state"]["Ball"], next_data.next_full_state["factored_state"]["Ball"], next_data.obs)
        if (
            (np.any(data.ext_term) and not self.only_termination) or # going to resample a new action
            np.any(data.done)
            or np.any(data.terminate)):
            self.keep_next = True
            # print("keeping", self.temporal_skip, np.any(data.ext_term),np.any(data.terminate), self.current_data.obs.squeeze()[:5], self.current_data.obs_next.squeeze()[:5])
            if not self.temporal_skip:
                added = True
                # print(next_data.act)
                # print("adding", 
                #     next_data.param, next_data.obs[:10], next_data.time, 
                #     next_data.act, next_data.mapped_act, 
                #     next_data.rew)
                    # next_data.done, next_data.true_done, next_data.rew, self.current_data.info[0]["TimeLimit.truncated"],
                    # next_data.full_state["factored_state"], next_data.next_full_state["factored_state"])
                # print(len(buffer))
                self.ptr, ep_rew, ep_len, ep_idx = buffer.add(
                        next_data, buffer_ids=ready_env_ids)
            else:
                skipped = True
            # print(self.temporal_skip, self.keep_next)
            self.time_counter = 0
        self.temporal_skip = "TimeLimit.truncated" in self.current_data.info[0] and self.current_data.info[0]["TimeLimit.truncated"] and np.any(data.done)
        self.time_counter += 1
        return next_data, skipped, added, self.ptr

class BufferWrapper():
    # wraps around the buffer components for pickleing
    def __init__(self, at, buffer, full_at, full_buffer, her_at=0, hindsight_buffer=None):
        self.at = at
        self.buffer = buffer
        self.full_at = full_at
        self.full_buffer = full_buffer
        if hindsight_buffer is not None:
            self.hindsight_buffer = hindsight_buffer
            self.hindsight_at = her_at

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
        environment_model = None,
        args: Namespace = None,
    ) -> None:
        self.param_recycle = args.param_recycle # repeat a parameter
        self.option = option
        self.at = 0
        self.env_reset = args.env_reset # if true, then the environment handles resetting
        self.full_at = 0 # the pointer for the buffer without temporal extension
        self.test = test
        self.full_buffer = copy.deepcopy(buffer) # TODO: make this conditional on usage?
        self.hit_miss_queue = deque(maxlen=2000) # not sure if this is the best way to record hits, but this records when a target position is reached
        self.true_interaction = args.true_interaction
        self.source, self.target = self.option.next_option.name, self.option.name
        self.environment_model = environment_model
        self.save_action = args.save_action # saves the option level action at each time step in option_action.txt in environment.save_dir
        self.save_path = self.environment_model.environment.save_path
        self._keep_proximity = args.keep_proximity
        self.terminate_reset = args.terminate_reset
        self.env = args.env
        option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'w')
        param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'w')
        option_dumps.close()
        param_dumps.close()
        
        # shortcut calling option attributes through option
        self.state_extractor = self.option.state_extractor
        self.get_param = self.option.sampler.get_param # sampler manages either recalling the param, or getting a new one
        self.exploration_noise = self.option.policy.exploration_noise
        self.temporal_aggregator = TemporalAggregator(sum_reward=args.sum_rewards, only_termination=args.only_termination)
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
        self._reset_components(full_state[0])

    def _reset_components(self, full_state):
        # if "param" in self.data: # reset components
        #     print("resetting components", self.data.param, self.data.obs, self.data.next_obs, self.data.terminations, self.data.done)
        if self.preprocess_fn:
            full_state = self.preprocess_fn(obs=full_state).get("obs", full_state)
        self._reset_state(0)
        self.option.reset(full_state)
        param, mask = self._reset_data(full_state)
        self.data.update(obs=[self.state_extractor.get_obs(full_state, param, mask)]) # self.option.get_state(obs, setting=self.option.input_setting, param=self.param if self.param is not None else None)
        self.temporal_aggregator.reset(self.data)
        # print("called reset", param, self.data.obs, self.data.terminations)

    def next_index(self, val):
        return (val + 1) % self.buffer.maxsize

    def _reset_data(self, full_state):
        # ensure that data has the correct: param, obs, obs_next, full_state, option_resample
        # will always sample a new param
        self.data.update(full_state=[full_state])
        param, mask, _ = self.get_param(full_state, True)
        self.data.update(target=self.state_extractor.get_target(self.data.full_state),
            obs=[self.state_extractor.get_obs(self.data.full_state, param, mask)])
        if "next_full_state" in self.data:
            self.data.update(obs_next=[self.state_extractor.get_obs(self.data.next_full_state, param, mask)])
        else:
            self.data.update(obs_next=[self.state_extractor.get_obs(self.data.full_state, param, mask)])
        self.data.update(param=[param], mask = [mask], option_resample=[[True]])
        term_chain = self.option.reset(full_state)
        act, chain, policy_batch, state, masks, resampled = self.option.extended_action_sample(self.data, None, term_chain, term_chain[:-1], random=False)
        self._policy_state_update(policy_batch)
        self.data.update(terminate=[term_chain[-1]], terminations=term_chain, ext_term=[term_chain[-2]], ext_terms=term_chain[:-1])
        self.data.update(done=[False], true_done=[False])
        self.option.update(self.buffer, True, self.data["full_state"], act, chain, term_chain, param, masks, not self.test)
        self.last_resampled_idx = self.full_at
        self.last_resampled_full_state = self.data.full_state
        self.temporal_aggregator.update(self.data)
        return param, mask

    def _policy_state_update(self, result):
        # update state / act / policy into self.data
        policy = result.get("policy", Batch())
        assert isinstance(policy, Batch)
        state = result.get("state", None)
        if state is not None:
            policy.hidden_state = state  # save state into buffer
            self.data.update(state_chain=state_chain, policy=policy)

    def _save_mapped_action(self, mapped_act, param, resampled, term):
        option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'a')
        option_dumps.write(str(self.environment_model.environment.get_itr() - 1) + ":" + action_toString(mapped_act) + "\t")
        option_dumps.close()
        param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'a')
        # print("param itr", self.environment_model.environment.get_itr(), action_toString(param), term)
        param_dumps.write(str(self.environment_model.environment.get_itr() - 1) + ":" + action_toString(param) + "|" + str(int(resampled)) + "," + str(int(term)) + "\t") # action_toString handles numpy vectors
        param_dumps.close()

    def perform_reset(self):
        # artificially create term to sample a new param
        # reset the temporal extension manager
        self.data.update(terminate=[True])
        self.ext_reset()

    def adjust_param(self):
        param, mask, new_param = self.get_param(self.data.full_state[0], self.data.terminate[0])
        if new_param and np.random.rand() > self.param_recycle:
            self.data.update(param=[param], mask=[mask])
        else:
            param, mask = self.data.param.squeeze(), self.data.mask.squeeze()
        self.data.obs = self.state_extractor.assign_param(self.data.full_state, self.data.obs, param, mask)
        self.data.obs_next = self.state_extractor.assign_param(self.data.full_state, self.data.obs_next, param, mask)
        return param, mask, new_param

    def get_buffer_idx(self):
        return self.at

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        n_term: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        visualize_param: str = "",
        no_grad: bool = True,
        force: [np.array, int] = None
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
        saved_fulls = list()
        hit_count, miss_count = 0,0
        term = False
        term_done = False # signifies if the termination was the result of a option terminal state
        term_end = False # if termination ends collection, should be True
        since_last = 0

        while True:
            param, mask, new_param = self.adjust_param()
            if self.test and new_param: print("new param", param)
            state_chain = self.data.state_chain if hasattr(self.data, "state_chain") else None
            if no_grad:
                with torch.no_grad():  # faster than retain_grad version
                    act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, self.data.terminations, self.data.ext_terms, random=random, force=force)
            else:
                act, action_chain, result, state_chain, masks, resampled = self.option.extended_action_sample(self.data, state_chain, self.data.terminations, self.data.ext_terms, random=random, force=force)
            self._policy_state_update(result)
            self.data.update(true_action=[action_chain[0]], act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled])
            # if resampled: print("resampling", act, since_last, action_chain[-1], self.data[0].target[:10], param, self.data[0].obs[:10], pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            since_last = (since_last) * int(not resampled) + 1
            # step in env
            action_remap = self.data.true_action
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)
            # print("obs_next_ball", self.data.full_state[0]['factored_state']['Ball'], obs_next[0]['factored_state']['Ball'])
            next_full_state = obs_next[0] # only handling one environment

            true_done, true_reward = done, rew
            # if self.option: # seems pointless to support no-option code
            obs = self.state_extractor.get_obs(self.data.full_state[0], param, mask) # one environment reliance
            obs_next = self.state_extractor.get_obs(next_full_state, param, mask) # one environment reliance
            target = self.state_extractor.get_target(self.data.full_state[0])
            next_target = self.state_extractor.get_target(next_full_state)
            inter_state = self.state_extractor.get_inter(self.data.full_state[0])

            # update the target, next target, obs, next_obs pair
            self.data.update(next_target=[next_target], target=[target], obs_next=[obs_next], obs = [obs])

            # get the dones, rewards, terminations and temporal extension terminations
            done, rewards, terminations, ext_terms, inter, time_cutoff = self.option.terminate_reward_chain(self.data.full_state[0], next_full_state, param, action_chain, mask, masks, environment_model=self.environment_model)
            done, rew, term, ext_term = done, rewards[-1], terminations[-1], ext_terms[-1]
            if self.save_action: self._save_mapped_action(action_chain[-1], param, resampled, term)

            cutoff = (np.array([time_cutoff]) or (true_done)) #and not term))  # treat true dones like time limits (TODO: this is not valid when learning to control true dones)
            if type(cutoff) != bool: cutoff = cutoff.squeeze()
            info[0]["TimeLimit.truncated"] = bool(cutoff + info[0]["TimeLimit.truncated"]) # environment might send truncated itself
            self.option.update(self.buffer, done, self.data.full_state[0], act, action_chain, terminations, param, masks, not self.test)

            # update hit-miss values
            rews += rew
            # print("rew sum", rews, rew) # debugging
            if term: 
                reward_check =  (done and rew > 0)
                if self.option.dataset_model.multi_instanced:
                    nt = self.option.dataset_model.split_instances(next_target)
                    ct = self.option.dataset_model.split_instances(target)
                    hit_idx = np.nonzero((nt[...,-1] - ct[...,-1]).flatten())
                    inst_hit = nt[hit_idx]
                    close = self.option.terminate_reward.epsilon_close
                    if self._keep_proximity: close = self.option.policy.learning_algorithm.dist
                    hit = ((np.linalg.norm((param-inst_hit), ord=1) <= close and not true_done)
                                or reward_check)
                    # print(reward_check, param, next_target, inst_hit, mask)
                else:
                    hit = ((np.linalg.norm((param-next_target) * mask) <= self.option.terminate_reward.epsilon_close and not true_done)
                                or reward_check)
                # print(self.option.dataset_model.multi_instanced,nt, ct, hit, hit_idx, param, inst_hit, true_done, reward_check)
                hit_count += int(hit)
                miss_count += int(not hit)
            # print(rew, target, self.data.mapped_act.squeeze(), act, param)

            # update the current values
            self.data.update(inter_state=[inter_state], next_full_state=[next_full_state], true_done=true_done, true_reward=true_reward, 
                param=[param], mask = [mask], info = info, inter = [inter], time=[1],
                rew=[rew], done=[done], terminate=[term], ext_term = [ext_term], # all prior are stored, after are not 
                terminations= terminations, rewards=rewards, masks=masks, ext_terms=ext_terms)
            # print (self.data) # FDO
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
            saved_fulls.append(copy.deepcopy(self.data))
            full_ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(
                self.data, buffer_ids=ready_env_ids)

            next_data, skipped, added, self.at = self._aggregate(self.data, self.buffer, full_ptr, ready_env_ids)
            if not self.test and self.policy_collect is not None: self.policy_collect(next_data, self.data, skipped, added)

            if term: 
                # print("term", rew)
                # print("term", self.data.full_state['factored_state']['Ball'], param, obs[:10], pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()),
                #     act, self.data.mapped_act)
                print("term", hit, term, true_done, done, inter, not time_cutoff, hit_count, miss_count, self.temporal_aggregator.time_counter, rew, 
                    pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()), 
                    param, self.data.mapped_act.squeeze(), act, next_target, 
                    inter_state, obs)

            # debugging and visualization
            # print(obs[5:10], self.data.full_state['factored_state']['Ball'])
            # if self.test: print(self.data.target.squeeze(), self.data.next_target.squeeze(), self.data.param.squeeze(), self.data.mapped_act.squeeze(), np.round_(self.data.act.squeeze(), 2), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            # print(self.data.mapped_act, self.data.inter_state, self.data.param, self.data.obs)
            # if self.test: print(step_count, self.data.param.squeeze(), self.data.target.squeeze(), self.data.mapped_act.squeeze(), np.round_(self.data.act.squeeze(), 2), self.data.rew.squeeze(), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            # if self.test and resampled: print(self.data.param.squeeze(), self.data.target.squeeze(), self.data.mapped_act.squeeze(), np.round_(self.data.act.squeeze(), 2), self.data.rew.squeeze(), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            # if self.test: print(self.data.obs.squeeze(), self.data.target.squeeze(), self.data.param.squeeze(), np.round_(self.data.act.squeeze(), 2), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            if self.test: print(self.data.param.squeeze(), self.data.target.squeeze(), self.data.mapped_act.squeeze(), np.round_(self.data.act.squeeze(), 2), self.data.rew.squeeze(), pytorch_model.unwrap(self.option.policy.compute_Q(self.data, nxt=False).squeeze()))
            
            if len(visualize_param) != 0:
                frame = np.array(self.env.render()).squeeze()
                if self.env == "SelfBreakout": # TODO: the visualize code only work for breakout at the moment
                    frame = visualize(frame, self.data.target[0], param, mask)
                if visualize_param != "nosave":
                    saveframe(frame, pth=visualize_param, count=self.counter, name="param_frame")
                    self.counter += 1
                printframe(frame, waittime=10)

            # collect statistics
            step_count += len(ready_env_ids)

            # update counters
            if np.any(done) or np.any(term):
                term_count += int(np.any(term))
                term_done, timer, true = self._done_check(term, true_done)
                term_end = term and not time_cutoff
                if np.any(done):
                    episode_count += int(np.any(done))
            if np.any(true_done) or (np.any(term) and self.terminate_reset):
                true_episode_count += 1
                # if we have a true done, reset the environments and self.data
                if self.env_reset: # the same as reset_env except it does not reset the env
                    full_state = self.environment_model.get_state()
                    self._reset_components(full_state)
                else:
                    # print("reset from collector", true_done)
                    self.reset_env()

                # full_state_reset = self.data.full_state[0] # set by reset_env
                # if self.preprocess_fn:
                #     full_state_reset = self.preprocess_fn(obs=full_state_reset).get("obs", full_state_reset)
                # self.data.update(next_full_state = [full_state_reset]) # obs_reset
                # self._reset_state(0)
                # self._reset_data(full_state_reset)
                # print(self.data)

            self.data.full_state = self.data.next_full_state
            self.data.target = self.data.next_target
            self.data.obs = self.data.obs_next

            # controls termination
            if (n_step and step_count >= n_step):
                break
            if (n_episode and episode_count >= n_episode):
                break
            if (n_term and term_count >= n_term):
                break
        # print(rews)
        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)
        return { # TODO: some of these don't return valid values
            "n/ep": episode_count,
            "n/tr": term_count,
            "n/st": step_count,
            "n/h": hit_count,
            "n/m": miss_count,
            "rews": rews,
            "terminate": term_end,
            "saved_fulls": saved_fulls
        }
