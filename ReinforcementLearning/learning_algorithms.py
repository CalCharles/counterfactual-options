import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from Networks.network import pytorch_model
import cv2
import time
import tianshou as ts
from Rollouts.param_buffer import ParamReplayBuffer, ParamPriorityReplayBuffer
from collections import deque



from file_management import load_from_pickle

class LearningOptimizer():
    def __init__(self, args, option):
        self.option = option
        self.i = 0
        self.updated = 0


    def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        '''
        records internal states for HER (basically), NOTE: no changes should be necessary for this
        '''
        self.i = i
        pass

class HER(LearningOptimizer):
    # TODO: adapt to work with TianShou
    def __init__(self, args, option):
        super().__init__(args, option)
        # only sample one other goal (the successful one)
        # self.rollouts = RLRollouts(option.rollouts.length, option.rollouts.shapes)
        if len(args.prioritized_replay) > 0:
            self.replay_buffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
        else:
            self.replay_buffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

        self._hypothesize = option.dataset_model.hypothesize
        self._check_interaction = option.terminate_reward.check_interaction
        self._get_mask_param = option.sampler.get_mask_param
        self.state_extractor = args.state_extractor
        self.done_model = option.done_model
        self.terminate_reward = option.terminate_reward

        self.last_done = 0
        self.last_res = 0
        self.select_positive = args.select_positive
        self.resample_timer = args.resample_timer
        self.use_interact = not args.true_environment and args.use_interact
        self.resample_interact = args.resample_interact # resample whenever an interaction occurs
        self.max_hindsight = args.max_hindsight
        self.replay_queue = deque(maxlen=args.max_hindsight)
        self.early_stopping = args.early_stopping # stops after seeing early stopping number of termiantions

    def record_state(self, full_batch, single_batch, skipped):
        if skipped: # see collector aggregate class method, but skipped follows a "true done" to avoid adding the transition 
            return 
        self.replay_queue.append(copy.deepcopy(full_batch))
        if np.any(full_batch.done) or np.any(full_batch.terminate): # either end of episode or end of timer
            interacting = True
            if self.use_interact:
                interact, pred, var = self._hypothesize(single_batch.full_state) # edit interaction model to take numpy arrays
                interacting = self._check_interaction(interact)
            if interacting:
                mask = full_batch.mask[0]
                if self.option.dataset_model.multi_instanced: # if multiinstanced, param should not be masked, and needs to be defined by the instance, not just the object state
                    dataset_model = self.option.dataset_model
                    instances = dataset_model.split_instances(self.option.get_state(single_batch["next_full_state"][0], setting=self.option.inter_setting))
                    interact_bin = dataset_model.interaction_model.instance_labels(self.option.get_state(full_batch.full_state[0], setting=self.option.inter_setting))
                    idx = pytorch_model.unwrap(dataset_model.check_interaction(interact_bin).nonzero())
                    idx = idx[0][1]
                    param = self._get_mask_param(instances[idx], mask)
                else:
                    param = self._get_mask_param(full_batch.next_target[0], mask)# self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
                rv_search = list()
                for i in range(1, len(self.replay_queue)): # go back in the replay queue, but stop if last_done is hit
                    batch = self.replay_queue[-i]
                    full_state = copy.deepcopy(batch)
                    inter_state = batch.inter_state[0]
                    batch.update(param=[param.copy()], obs = self.state_extractor.assign_param(batch.obs, param),
                        obs_next = self.state_extractor.assign_param(batch.obs_next, param), mask=[mask])
                    true_done = batch.true_done
                    true_reward = batch.true_reward
                    term, rew = self.option.terminate_reward.check(batch.full_state[0], batch.next_full_state[0], param, mask, use_timer=False)
                    timer, self.done_model.timer = self.done_model.timer, 0
                    done = self.done_model.check(term, true_done)
                    self.done_model.timer = timer
                    full_state.update(done=done, terminate=term, rew=[rew])
                    rv_search.append(full_state)

                early_stopping_counter = self.early_stopping
                for i in range(1, len(rv_search)+1):
                    full_state = rv_search[-i]
                    if self.early_stopping > 0 and np.any(full_state.terminate):
                        early_stopping_counter -= 1
                        if early_stopping_counter == 0:
                            full_state.update(done=[i < len(rv_search)]) # force it to be done to prevent fringing effects
                    ptr, ep_rew, ep_len, ep_idx = self.replay_buffer.add(full_state, buffer_ids=[0])
                    if early_stopping_counter == 0 and self.early_stopping > 0:
                        break
            del self.replay_queue
            self.replay_queue = list()

    def sample_buffer(self, buffer):
        if np.random.random() > self.select_positive or len(self.replay_buffer) == 0:
            return buffer
        else:
            return self.replay_buffer
