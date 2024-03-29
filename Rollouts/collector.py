import numpy as np
import gym
import time
import torch
import warnings
import cv2
import copy
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
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


class OptionCollector(Collector): # change to line  (update batch) and line 12 (param parameter), the rest of parameter handling must be in policy
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        option: Option = None,
        use_forced_actions = False, # TODO: debugging parameter, remove
        use_param = False,
        test = False,
        true_env=False,
        use_rel = False,
        print_test=False,
        param_recycle = False,
        grayscale = False
    ) -> None:
        self.param_recycle = param_recycle
        self.param = None
        self.mask = None
        self.last_done = True
        self.last_term = True
        self.use_param=use_param
        self.use_rel = use_rel
        self.option = option
        self.counter = 0
        self.at = 0
        self.full_at = 0 # the pointer for the buffer without temporal extension
        self.use_forced_actions = use_forced_actions
        self.test = test
        self.print_test = print_test
        self.true_env=true_env
        self.full_buffer = copy.deepcopy(buffer) # TODO: make this conditional on usage?
        self.last_resampled_full_state = None
        self.last_resampled_idx = 0
        self.force_action = True
        self.new_param = True
        self.grayscale = grayscale
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.hit_miss_queue = deque(maxlen=2000)
        self.last_truncated_done=False
        self.last_truncated = False
        self.last_ptr = 0
        self.skip_next = False
        # self.other_buffer = ReplayBuffer.load_hdf5("data/working_rollouts.hdf5")

    def reset(self):
        super().reset()
        # if self.use_param: self.param, mask, new_param = self.option.get_param(self.data.full_state, self.last_done, force= True)

    def reset_env(self, keep_statistics: bool = False):
        obs = self.env.reset()
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get("obs", obs)
        # if self.option: 
        self.data.full_state = obs
        self.reset_temporal_extension()
        obs = self.option.get_state(obs, setting=self.option.input_setting, param=self.param if self.param is not None else None)
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

    def next_index(self, val):
        return (val + 1) % self.buffer.maxsize

    def get_param(self, last_term, force): # a new param is needed if: terminated, resetting
        self.mask, self.new_param = None, False
        if self.use_param: self.param, self.mask, self.new_param = self.option.get_param(self.data.full_state, last_term, force= force)
        else: self.param, self.mask = np.ones(1), np.ones(1)
        # require a new action from temporal extension after new_param
        self.force_action = self.new_param
        return self.param, self.mask, self.new_param

    def reset_temporal_extension(self):
        # print("reset called")
        # ensure that data has the correct: param, obs, full_state, option_resample
        self.data.update(full_state=[self.option.environment_model.get_state()])
        self.param, self.mask, self.new_param = self.get_param(0, True)
        self.force_action = False # don't force an action because we will manually run a new action
        self.data.update(param=[self.param], mask = [self.mask], obs=[self.option.get_state(setting=self.option.input_setting, param=self.param)], option_resample=[[True]])
        act, chain, policy_batch, state, resampled = self.option.sample_action_chain(self.data, None, random=False, force=True)
        self.option.step(self.data["full_state"], chain)
        self.data.update(target=[self.option.get_state(setting=self.option.output_setting)])
        self.last_resampled_idx = self.full_at
        self.last_resampled_full_state = self.data.full_state

    def aggregate(self, ptr, ready_env_ids):
        # get the indices between the last resampled and ptr
        ep_rew, ep_len, ep_idx = None, None, None
        next_data = copy.deepcopy(self.data)
        indices = [self.last_resampled_idx]
        while indices[-1] != ptr[0]:
            indices.append(self.next_index(indices[-1]))
        indices.pop(-1) # we add up to but not including the last state
        if len(indices) == 0:
            indices.append(self.last_resampled_idx) # skips the next one when there is an empty value
        skipped=False
        if len(indices) > 0 and not self.skip_next:
            self.skip_next = np.any(self.data.invalid)
            indices = np.stack(indices)

            # the last state took the last action (not the new, resampled action). The next state is the next full state
            last_param, mask = self.full_buffer.param[indices[-1]], self.full_buffer.mask[indices[-1]]
            # act might have problems if it is more than one dimensional...
            # print(self.last_resampled_full_state, self.data.full_state, self.option.get_state(self.last_resampled_full_state, form=1, inp=2 if self.use_param else 0, rel=1 if self.use_rel else 0).shape)
            next_data.update(full_state=self.last_resampled_full_state, next_full_state =self.data.full_state, act = [self.full_buffer.act[indices[-1]]], mapped_act = [self.full_buffer.mapped_act[indices[-1]]])
            next_data.update(target=self.option.get_state(self.last_resampled_full_state, setting=self.option.output_setting), next_target=self.option.get_state(self.data.full_state, setting=self.option.output_setting))
            next_data.update(obs=self.option.get_state(self.last_resampled_full_state, setting=self.option.input_setting, param=last_param), obs_next=self.option.get_state(self.data.full_state, setting=self.option.input_setting, param=last_param))
            next_data.update(param=[last_param], mask = [mask])
            # reward is the sum of rewards, done is if a done occurred anywhere in the temporal extension (but done should force a resampling)
            next_data.update(rew=[np.sum(self.full_buffer.rew[indices], axis=0)], done = [bool(min(1, np.sum(self.full_buffer.done[indices])))], terminate = [min(1, np.sum(self.full_buffer.terminate[indices]))]) 
            next_data.update(true_reward=[np.sum(self.full_buffer.true_reward[indices], axis=0)], true_done = [bool(min(1, np.sum(self.full_buffer.true_done[indices])))])
            next_data.info["TimeLimit.truncated"] = [self.last_truncated]
            # print("adding", next_data.terminate, next_data.target, next_data.next_target, next_data.true_done, self.data.true_done, self.data.done)
            # print("adding", next_data.act, next_data.mapped_act, next_data.obs, next_data.obs_next)
            # print("from", self.data.act, self.data.mapped_act, self.data.obs, self.data.obs_next)
            # print(next_data.rew, next_data.done, next_data.terminate, self.data.rew, self.data.done)
            # print_shape(next_data, "next_data")
            # print_shape(self.data, "data")
            self.last_resampled_idx = ptr[0]
            # print ("adding", ptr, next_data.target[0], next_data.next_target[0], next_data.done[0], next_data.true_done[0])

            # print(next_data.obs_next[0,4:7], next_data.true_done, next_data.done)
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    next_data, buffer_ids=ready_env_ids)
            self.at = ptr # might have some issues with this assignment
            self.last_truncated = self.data.info["TimeLimit.truncated"]
            # print(self.full_buffer.act.shape, self.buffer.act.shape)
        else:
            # print("skipping", np.any(self.data.invalid), self.skip_next, self.option.get_state(self.last_resampled_full_state, setting=self.option.output_setting), self.option.get_state(self.data.full_state, setting=self.option.output_setting))
            self.last_resampled_idx = ptr[0]
            self.at = [self.at]
            self.skip_next = False
            skipped = True
        # print(self.last_resampled_idx, ptr[0])
        self.last_resampled_full_state = self.data.full_state # TODO: figure out whether to use full_state or next_full_state
        return next_data, skipped, self.at, ep_rew, ep_len, ep_idx


    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        n_term: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        visualize_param: str = "",
        no_grad: bool = True,
        needs_new_param = False,
        action_record: Optional[list] = None
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
        if n_term is not None:
            assert n_term > 0
            self.data = self.data[:min(self.env_num, n_term)]
        if not (n_step or n_episode):
            raise TypeError("Please specify at least one (either n_step or n_episode) "
                            "in AsyncCollector.collect().")

        start_time = time.time()

        step_count = 0
        episode_count = 0
        term_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        rews = 0
        true_done_trigger = 0
        closest = list()
        idxs = list()

        self.data.update(full_state=[self.option.environment_model.get_state()])
        targets, actions, obs = list(), list(), list()
        if self.param is not None and not needs_new_param: 
            self.data.param = [self.param]
            self.data.mask = [self.mask]
            self.new_param = False
        else:
            self.param, self.mask, self.new_param = self.get_param(False, True)
            # if self.use_param: self.param, mask = self.option.get_param(self.data.full_state, self.last_done, force= True)
            # else: self.param = np.ones(1)
            if self.print_test: print("param", self.use_param, self.param)
            # print("param got", self.param)
        if self.param is not None:
            self.data.update(param=[self.param], mask=[self.mask])
        self.data.update(obs=[self.option.get_state(setting=self.option.input_setting, param=self.param)])
        if 'state_chain' not in self.data: state_chain = None # support for RNNs while handling feedforward
        else: state_chain = self.data.state_chain
        while True:
            # print(step_count, self.data, len(self.data), len(ready_env_ids))
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                # if self.option: 
                act, action_chain, result, state_chain, resampled = self.option.sample_action_chain(self.data, state_chain, random=True, force=self.force_action)
                # print("rand:", resampled)
                # else:
                #     action_chain=[self._action_space[i].sample() for i in ready_env_ids] #line makes no sense
                if type(act) == np.ndarray and not self.option.expand_policy_space: act = act.squeeze()
                if self.option.discrete_actions: act = act.squeeze()
                self.data.update(act=[act], mapped_act=[action_chain[-1]], true_action=[action_chain[0]], option_resample=[[resampled]])
                if state_chain is not None: self.data.update(state_chain = state_chain)
                # act = [action_chain[-1]]
            else:
            # Lines altered from collector
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        # print(self.data.obs_next)
                        act, action_chain, result, state_chain, resampled = self.option.sample_action_chain(self.data, state_chain, force=self.force_action)
                    # print("pol:", resampled)
                else:
                    act, action_chain, result, state_chain, resampled = self.option.sample_action_chain(self.data, state_chain, force=self.force_action)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    # print(len(state.shape) == 1, len(result.act.shape) == 2)
                    if len(state.shape) == 1 and len(result.act.shape) == 2: state = [state] 
                    policy.hidden_state = state  # save state into buffer
                    self.data.update(state_chain=state_chain)
                self.data.update(true_action=[action_chain[0]])
                # else: act = to_numpy(result.act)
                # if self.exploration_noise:
                #     act = self.policy.exploration_noise(act, self.data)
                if type(act) == np.ndarray and not self.option.expand_policy_space: act = act.squeeze()
                if self.option.discrete_actions: act = act.squeeze()
                self.data.update(policy=policy, act=[act], mapped_act=[action_chain[-1]], option_resample=[resampled])
            if action_record is not None:
                action_record.append(action_chain[-1])
            self.force_action=False
            # print(act)
            actions.append(act)
            # get bounded and remapped actions first (not saved into buffer)
            # if self.use_forced_actions:
            #     action_remap = self.policy.map_action([self.other_buffer[self.at].act])
            #     # print(self.at, self.data.true_action, [self.other_buffer[self.at].act])
            # else:
            action_remap = self.data.true_action
            # print(self.data.true_action, action_remap)
            # step in env
            # print(action_remap, self.data.act, action_chain)
            # print("to env", action_remap, "in data", self.data.act[0], self.data.mapped_act[0])
            obs_next, rew, done, info = self.env.step(action_remap, id=ready_env_ids)
            # print(info)
            # print("obs", obs_next[0]["factored_state"]["Gripper"])

            # cv2.imshow('state', obs_next[0]['raw_state'])
            # cv2.waitKey(1)

            # print("postprocessing", obs_next, rew, done, info)
            next_full_state = obs_next[0] # only handling one environment
            timed_out = False

            dones, rewards, terminations, true_done, true_reward = [], [], [], done, rew
            # if self.option: # seems pointless to support no-option code
            self.option.step(next_full_state, action_chain)
            obs_next = self.option.get_state(next_full_state, setting=self.option.input_setting, param=self.param) # one environment reliance
            next_target = [self.option.get_state(next_full_state, setting=self.option.output_setting)]
            target = [self.option.get_state(self.data.full_state[0], setting=self.option.output_setting)]
            targets.append(obs_next)
            # print(target, self.option.next_option.get_state(self.data.full_state, form=1,inp=1))
            # print(self.param, self.mask)
            # print(self.option.object_name, self.option.reward)
            dones, rewards, terminations = self.option.terminate_reward(self.data.full_state[0], next_full_state, self.param, action_chain, mask=self.mask)
            done, rew, term = dones[-1], rewards[-1], terminations[-1]
            timed_out = self.option.step_timer(done)
            # if term or done or self.option.termination.inter:
            if term or done:
                # if self.option.termination.param_interaction:
                p = self.param * self.mask
                if self.option.dataset_model.multi_instanced:
                    pos = self.option.get_state(self.data.full_state[0], setting=self.option.output_setting)
                    os = self.option.get_state(next_full_state, setting=self.option.output_setting)
                    instanced = self.option.dataset_model.split_instances(os)
                    pinst = self.option.dataset_model.split_instances(pos)
                    hit = pytorch_model.unwrap(self.option.dataset_model.interaction_model.instance_labels(self.option.get_state(self.data.full_state[0], setting=self.option.inter_setting)))
                    hit[hit < self.option.dataset_model.interaction_minimum] = 0
                    hit = hit.nonzero()[1]
                    inverse_mask = self.mask.copy()
                    inverse_mask[self.mask == 0] = -1
                    inverse_mask[self.mask == 1] = 0
                    inverse_mask *= -1
                    # print(inverse_mask, self.param, np.abs((instanced - self.param) * inverse_mask).shape, np.abs((instanced - self.param) * inverse_mask)[0], instanced[0])
                    remasked = np.abs((instanced - self.param) * inverse_mask)
                    diff = np.sum(np.abs(remasked), axis=1)
                    idxes = np.argmin(diff, axis=0)
                    pos, os = pinst[idxes], instanced[idxes]
                    phit, hit = pinst[hit], instanced[hit]
                    hitv = hit[-1] if len(hit) > 0 else hit
                    # print(pos, os)
                    ep_close = self.option.termination.terminator.epsilon
                    inter = self.option.termination.inter
                    # print("terminate or done", term, done)

                    if term:
                        print("hit target", phit, hitv, pos[-1], os[-1], self.param, inter, term, done, rew)
                        self.hit_miss_queue.append(1)
                    else:
                        print("missed target", phit, hitv, pos[-1], os[-1], self.param, inter, term, done, rew)
                        self.hit_miss_queue.append(0)
                else:
                    os = self.option.get_state(next_full_state, setting=self.option.output_setting) * self.mask
                    ep_close = self.option.termination.epsilon_close
                    inter = self.option.termination.inter
                    phit = self.option.termination.p_hit
                    # print("terminate or done", term, done)
                    if np.linalg.norm(os - p, ord  = 1) <= ep_close:
                        print("hit target", self.option.get_state(self.data.full_state[0], setting=self.option.inter_setting), self.option.get_state(next_full_state, setting=self.option.inter_setting), self.param, inter, term, done, phit, rew)
                        self.hit_miss_queue.append(1)
                    else:
                        print("missed target", self.option.get_state(self.data.full_state[0], setting=self.option.input_setting, param=self.param), self.option.get_state(next_full_state, setting=self.option.inter_setting), self.param, inter, term, done, phit, rew)
                        self.hit_miss_queue.append(0)
            # print("rew", rew)
            rews += rew
            # if self.print_test: print(rews, rew)
            self.data.update(next_target=next_target, target=target)
            # print("obs", [obs_next], "next full", [next_full_state], "done", [true_done], "reward", [true_reward], "rew", [rew], "done", [done], info, self.param, "lists", dones, rewards)
            # print("next term", term)
            self.data.update(obs_next=[obs_next], next_full_state=[next_full_state], true_done=true_done, true_reward=true_reward, 
            rew=[rew], done=[done], terminate=[term], info=info, param=[self.param], mask=[self.mask], dones=dones, rewards=rewards, terminations=terminations) # edited
            if self.preprocess_fn:
                self.data.update(self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                ))
            # print("prerender", self.option.get_state(self.data.full_state[0], form = 1, inp=0))
            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # if self.use_forced_actions:
            #     self.data.update(act=[self.other_buffer[self.at].act])

            # add data into the buffer
            # print(obs_next[4:7], true_done, done, term, rew, (resampled and not self.last_done and not self.last_term) or done or term)
            # print(ready_env_ids)
            # print(self.data)
            # print(self.data)
            # print(self.data.obs.shape, self.data.obs_next.shape, self.data.param.shape, self.data.target.shape, self.data.next_target.shape)
            # print([(k, self.data[k].shape) for k in self.data.keys()])
            # print("preadd", self.option.get_state(self.data.full_state[0], form = 1, inp=0))
            truncated_done = "TimeLimit.truncated" in self.data.info and self.data.info["TimeLimit.truncated"] and np.any(self.data.true_done)
            if self.last_truncated_done:
                self.data["invalid"] = [True]
                # print("invalid above", done, self.data.true_done)
                ptr = self.last_ptr
                full_ptr = self.last_ptr
                # print ("skipping", ptr, self.data.target[0], self.data.next_target[0], self.data.done[0], self.data.true_done[0])
            else:
                self.data["invalid"] = [False]
                ptr, ep_rew, ep_len, ep_idx = self.full_buffer.add(
                    self.data, buffer_ids=ready_env_ids)
                # print ("adding", ptr, self.data.target[0], self.data.next_target[0], self.data.done[0], self.data.true_done[0])
                # if not self.print_test: print( "local", self.full_buffer[ptr-1].done[0], self.full_buffer[ptr].done[0], self.full_buffer[ptr+1].done[0])
                full_ptr = ptr
                self.last_ptr = ptr
            # print("resampled", resampled)
            # print("normal step")
            if (resampled #and not self.last_done and not self.last_term) 
                or done 
                or term 
                or (self.option.dataset_model.multi_instanced and self.option.termination.inter)): #pytorch_model.unwrap(self.option.dataset_model.interaction_model(pytorch_model.wrap(self.option.get_state(self.data.full_state[0], self.option.inter_setting), cuda=self.option.iscuda)))):
                next_data, skipped, ptr, ep_rew_te, ep_len_te, ep_idx_te = self.aggregate(ptr, ready_env_ids)
                if not skipped: idxs.append(ptr)
                self.at = self.next_index(ptr[0])
                if self.policy.collect and not self.test: self.policy.collect(next_data, self.data, skipped)
            # if self.skip_next
            self.full_at = self.next_index(full_ptr[0])
            self.last_done = np.any(done)
            self.last_term = np.any(term)
            self.last_truncated_done = truncated_done

            if self.print_test: print(self.data.obs.squeeze(), self.data.act.squeeze())
            if len(visualize_param) != 0:
                frame = np.array(self.env.render()).squeeze()
                new_frame = visualize(frame, self.data.target[0], self.param, self.mask)
                if visualize_param != "nosave":
                    saveframe(new_frame, pth=visualize_param, count=self.counter, name="param_frame")
                    self.counter += 1
                printframe(new_frame, waittime=1)
            # if self.visualize:
            #     frame = np.array(self.env.render()).squeeze()
            #     # print(np.array(frame).shape)
            #     printframe(frame, waittime=10)

            # print(self.data.param.shape, self.buffer.param.shape, ready_env_ids)
            # print([(k, self.buffer._meta[k].shape) for k in self.buffer._meta.keys()])
            # error
            # if self.policy.collect and not self.test and not random: self.policy.collect(self.data)
            # if self.policy.collect and not self.test and not random: 
            #     print(self.buffer.act.shape, self.data.act.shape, self.policy.learning_algorithm.replay_buffer.act.shape)
            # if self.use_forced_actions:
            #     print(ptr, self.data.act, self.data.obs, self.data.done, np.any(done), self.data.rew)
            # print(self.buffer)
            # self.full_at = (full_ptr[0] + 1) % self.buffer.maxsize
            # self.at = (ptr[0] + 1) % self.buffer.maxsize
            # print(self.full_at, self.at, ptr[0])
            # line altered from collector ABOVE

            # collect statistics
            step_count += len(ready_env_ids)

            # print(self.at)
            if np.any(done):
                # print(dones, done, true_done)
                # print("done triggered", dones, true_done, target[0], next_target[0])
                # if self.test: error
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                # if n_episode:
                #     surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                #     if surplus_env_num > 0:
                #         mask = np.ones_like(ready_env_ids, dtype=bool)
                #         mask[env_ind_local[:surplus_env_num]] = False
                #         ready_env_ids = ready_env_ids[mask]
                #         self.data = self.data[mask]
                # print(self.data)
                # print("param", self.param)
            if np.any(done) or np.any(term):
                # print("getting new param", done, term, self.option.timer)
                self.force_action = True
                if np.any(term):
                    term_count += 1
                if np.random.rand() > self.param_recycle:
                    self.param, self.mask, self.new_param = self.get_param(True, False) # technically, self.last_done is forcing
                else:
                    self.new_param = True # treat this as a new param even though it is the same
                if self.print_test: print("new param", self.use_param, self.param)

                # if self.use_param: self.param, mask, new_param = self.option.get_param(self.data.full_state, self.last_done) # set the param
                # if not self.test: print("done called: param", self.param, obs_next, rewards, dones, self.option.get_state(next_full_state, form=1, inp=0), self.option.timer, true_done)
                # print("episode occurred", episode_count, step_count)
            if np.any(true_done):
                # print("reached done", done, true_done, dones, term)
                obs_reset = self.env.reset(env_ind_global)
                # print("param at done", self.param)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(obs=obs_reset).get("obs", obs_reset)
                self.data.obs_next[env_ind_local] = self.option.get_state(obs_reset, setting=self.option.input_setting, param=self.param) # obs_reset
                for i in env_ind_local:
                    self._reset_state(i)
                true_done_trigger += 1


            self.data.obs = self.data.obs_next
            self.data.full_state = self.data.next_full_state
            self.new_param = False

            if (n_step and step_count >= n_step):
                if self.test: self.hit_miss_queue.append(0)
                # print("stbreak", step_count, episode_count)
                break
            if (n_episode and episode_count >= n_episode):
                # print("epbreak", step_count, episode_count)
                break
            if (n_term and term_count >= n_term):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            # print("calling reset env", n_episode)
            self.data = Batch(**{k: dict() for k in ParamReplayBuffer._reserved_keys})
            # self.reset_env()

        # if episode_count > 0:
        #     rews, lens, idxs = list(map(
        #         np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
        # else:
        rews, lens, idxs = np.array(rews), np.array([], int), np.array(idxs, int)
        # print(actions)
        act_stack = np.stack(actions, axis=0)
        if len(act_stack.shape) < 2:
            act_stack = np.expand_dims(actions, axis=1)
        # if self.print_test: print("actions, ", act_stack.squeeze())
        # print(self.data)
        # print(targets[0].shape, self.data.target.shape, act_stack.shape)
        # print(np.stack(targets, axis=0), act_stack.shape)
        # if self.print_test: print("targets, actions", np.concatenate((np.stack(targets, axis=0), act_stack), axis=1))
        return { # TODO: some of these don't return valid values
            "n/ep": episode_count,
            "n/tr": term_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "done": int(n_episode is not None and episode_count >= n_episode),
            "true_done": true_done_trigger
        }
