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
        self.args = args
        self.i = 0
        self.updated = 0

    def initialize_optimizer(self, args, policy):
        '''
        Initializes the choice of optimizer from the choices in TianShao, and applies the necessary startup to run update
        '''
        return

    def step(self, args, rollouts):
        '''
        call policy.update in tianshao (wrapped)
        '''
        pass

    def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        '''
        records internal states for HER (basically), NOTE: no changes should be necessary for this
        '''
        self.i = i
        pass

class TSOptimizer(LearningOptimizer):
    def __init__(self, args, option):
        self.option = option
        self.policy = self.option.policy # optimizer is built into policy, initialized outside
        self.args = args
        self.i = 0
        self.updated = 0

    def initialize_optimizer(self, args, policy):
        '''
        No effect for calling this because everything should be done when the policy is initialized
        '''
        pass

    def single_step(self, args, rollouts):
        batch, idxes = rollouts.get_ts_batch(args.batch_size)
        loss_dict = self.policy.learn(batch)
        return loss_dict

    def step(self, args, rollouts):
        '''
        call policy.update in tianshao (wrapped)
        interface changed for step to return loss dict
        '''
        for i in range(args.grad_epoch):
            loss_dict = self.single_step(args, rollouts) # better logging using logger later
        return loss_dict

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
        # self.replay_buffer = ParamReplayBuffer(args.buffer_len, 1)

        self.last_done = 0
        self.last_res = 0
        self.select_positive = args.select_positive
        self.resample_timer = args.resample_timer
        self.use_interact = not args.true_environment and args.use_interact
        self.resample_interact = args.resample_interact # resample whenever an interaction occurs
        self.max_hindsight = args.max_hindsight
        self.replay_queue = deque(maxlen=args.max_hindsight)
        self.early_stopping = args.early_stopping # stops after seeing early stopping number of termiantions




    # def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
    def record_state(self, full_batch, single_batch, skipped):
        # super().record_state(i, state, next_state, action_chain, rl_outputs, param, rewards, dones)
                
        # ptr, ep_rew, ep_len, ep_idx = self.replay_buffer.add(full_batch, buffer_ids=[0]) # TODO: hacked buffer IDs because parallelism not supported
        if skipped: # see collector aggregate class method, but skipped follows a "true done" to avoid adding the transition 
            return 
        self.replay_queue.append(copy.deepcopy(full_batch))
        # self.last_true_done += 1 
        # print("shapes", full_batch.param.shape, self.replay_buffer.param.shape)
        # self.rollouts.append(**self.option.get_state_dict(state, next_state, action_chain, rl_outputs, param, rewards, dones))
        self.last_done += 1
        self.last_res += 1
        # print(self.last_res, self.last_done, full_batch.terminate)
        # if (self.last_res == self.resample_timer and self.resample_timer > 0) or np.any(full_batch.terminate):
        #     print("ENTERING HER INTERNAL LOOP")

        # print(self.last_done)
        # if dones[-1] == 1: # option terminated, resample
        # print("adding", self.last_res, self.resample_timer, (self.last_res == self.resample_timer and self.resample_timer > 0), np.any(full_batch.terminate))
        if (self.last_res == self.resample_timer and self.resample_timer > 0) or np.any(full_batch.terminate): # either end of episode or end of timer
            # if ((self.last_done == self.resample_timer and self.resample_timer > 0)):
            #     print("resetting")
            interacting = True
            # interact = self.option.dataset_model.interaction_model(self.option.get_state(full_batch.full_state)) # edit interaction model to take numpy arrays
            if self.use_interact:
                interact = self.option.dataset_model.interaction_model(self.option.get_state(full_batch.full_state, setting=self.option.inter_setting)) # edit interaction model to take numpy arrays
                # interact = self.option.dataset_model.interaction_model(self.option.get_state(next_state))
                interacting = self.option.dataset_model.check_interaction(interact)
                # print(interacting, interact)
            # print("DONE REACHED!!", self.use_interact, interacting, self.last_done)
            
            # print(interacting, self.last_res, self.resample_timer)
            if interacting:
                # indexes = (self.rollouts.at - self.last_done) % self.rollouts.length
                # indexes = np.array([(ptr[0] + 1 - self.last_done + i) % self.replay_buffer.maxsize for i in range(self.last_done)])
                mask = full_batch.mask[0]
                if self.option.dataset_model.multi_instanced: # if multiinstanced, param should not be masked, and needs to be defined by the instance, not just the object state
                    dataset_model = self.option.dataset_model
                    instances = dataset_model.split_instances(self.option.get_state(single_batch["next_full_state"][0], setting=self.option.inter_setting))
                    interact_bin = dataset_model.interaction_model.instance_labels(self.option.get_state(full_batch.full_state[0], setting=self.option.inter_setting))
                    # print(interact_bin, dataset_model.check_interaction(interact_bin))
                    idx = pytorch_model.unwrap(dataset_model.check_interaction(interact_bin).nonzero())
                    # if len(idx) > 1: # only choose one target
                    idx = idx[0][1]
                    param = instances[idx]
                    # print(param, idx, instances.shape)
                else:
                    # param = self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
                    param = full_batch.next_target[0]# self.option.get_state(full_batch["next_full_state"][0], setting=self.option.output_setting) * mask
                # if self.min_hindsight < 0:
                #     num_back = max(self.min_hindsight, self.last_true_done * float(self.min_hindsight > 0))
                # else:
                num_back = min(self.max_hindsight, self.last_done)
                num_vals = self.last_done
                # broadcast_object_state = np.stack([param.copy() for _ in range(num_vals)], axis=0)
                rv_search = list()
                # max_val = max(self.min_hindsight, self.last_true_done * float(self.min_hindsight > 0))
                for i in range(1, num_back+1): # go back in the replay queue, but stop if last_done is hit
                    full_state = self.replay_queue[-i]
                    input_state = self.option.get_state(full_state.next_full_state[0], setting=self.option.inter_setting)
                    object_state = full_state.next_target
                    # print (param, full_state.obs_next, full_state.next_target, self.option.assign_param(full_state.obs_next, param, full_state.next_target))
                    full_state.update(param=[param.copy()], obs = self.option.assign_param(full_state.obs, param, full_state.next_target),
                        obs_next = self.option.assign_param(full_state.obs_next, param, full_state.next_target), mask=[mask])
                    # input_state = self.option.strip_param(full_state.obs) # technically, full state doesn't need to strip since it has the full dict
                    true_done = full_state.true_done
                    true_reward = full_state.true_reward
                    # print(input_state, object_state, param, true_done)
                    term = self.option.termination.check(input_state, object_state, param, mask, true_done)
                    done = self.option.done_model.check(term, 1, true_done)
                    rew = self.option.reward.get_reward(input_state, object_state, param, mask, true_reward)
                    # print(done, term, rew, true_done, full_state.obs, param, input_state, object_state)
                    full_state.update(done=done, terminate=term, rew=rew)
                    rv_search.append(full_state)

                early_stopping_counter = self.early_stopping
                for i in range(1, len(rv_search)+1):
                    full_state = rv_search[-i]
                    if self.early_stopping > 0 and np.any(full_state.terminate):
                        early_stopping_counter -= 1
                        if early_stopping_counter == 0:
                            full_state.update(done=[i < len(rv_search)]) # force it to be done to prevent fringing effects
                    # print("adding", full_state.target, full_state.obs, full_state.terminate, full_state.act, full_state.rew, full_state.done)
                    ptr, ep_rew, ep_len, ep_idx = self.replay_buffer.add(full_state, buffer_ids=[0])
                    # print(ptr, full_state.target, full_state.next_target, full_state.done, full_state.rew)
                    if early_stopping_counter == 0 and self.early_stopping > 0:
                        break

            #     # print(broadcast_object_state.shape)
            #     # print("HER insert", object_state, self.use_interact, self.last_done, self.resample_timer)
            #     self.replay_buffer.param[indexes] = broadcast_object_state
            #     self.replay_buffer.obs[indexes] = self.option.assign_param(self.replay_buffer.obs[indexes], broadcast_object_state)
            #     self.replay_buffer.obs_next[indexes] = self.option.assign_param(self.replay_buffer.obs_next[indexes], broadcast_object_state)
            #     # self.rollouts.insert_value(last_done_at, 0, num_vals, "param", broadcast_object_state)
            #     param = broadcast_object_state
            #     # TODO: sample param mask 
            #     # p, mask = self.sampler.sample(self.get_state(full_state, form=0))
            #     # print(self.replay_buffer.obs_next[indexes.squeeze()].shape)
            #     # param = self.rollouts.get_values("param")[-self.last_done:]
            #     # input_state = self.rollouts.get_values("next_state")[-self.last_done:]
            #     # object_state = self.rollouts.get_values("next_object_state")[-self.last_done:]
            #     # true_done = self.rollouts.get_values("true_done")[-self.last_done:]
            #     # true_reward = self.rollouts.get_values("true_reward")[-self.last_done:]

            #     # print(self.rollouts.at, input_state.shape, object_state.shape, param.shape)
                
            #     # self.rollouts.insert_value(last_done_at, 0, num_vals, "done", self.option.termination.check(input_state, object_state, param, true_done))
            #     # self.rollouts.insert_value(last_done_at, 0, num_vals, "reward", self.option.reward.get_reward(input_state, object_state, param, true_reward))
            #     # print("learning algorithm", input_state, object_state, param)
            #     self.replay_buffer.done[indexes] = self.option.termination.check(input_state, object_state, param, true_done)
            #     self.replay_buffer.rew[indexes] = self.option.reward.get_reward(input_state, object_state, param, true_reward)
            #     # print(self.option.termination.check(input_state, object_state, param, true_done))
            #     # print(pytorch_model.unwrap(self.rollouts.get_values("reward")[-self.last_done:]))
            #     # for i in range(self.last_done):
            #     #     # print(self.rollouts.at - self.last_done + i)
            #     #     state = new_states[self.rollouts.at - self.last_done + i]
            #     #     state[:,:,2] = new_params[self.rollouts.at - self.last_done + i]
            #     #     cv2.imshow('Example - Show image in window',pytorch_model.unwrap(state))
            #     #     cv2.waitKey(100)
            #     # print("inserting", self.option.reward.get_reward(input_state, object_state, param))
            #     # print("added", self.replay_buffer.done)
            #     # print([np.argwhere(p == 10.0).squeeze() for p in self.replay_buffer.param])
            #     # print(self.replay_buffer.done, self.replay_buffer.rew)
            # # else:
            # #     print("no interact", self.last_done, interact, interacting, 
            # #         self.option.dataset_model.interaction_model(self.option.get_state(full_batch.next_full_state)), self.option.get_state(full_batch.full_state),
            # #         self.option.get_state(full_batch.next_full_state),
            # #         self.option.get_state(single_batch.full_state), self.option.get_state(single_batch.next_full_state))
            self.last_res = 0
            if np.any(full_batch.terminate):
                # print("terminated")
                self.last_done = 0 # resamples as long as any done is reached, not just an interacting one
                del self.replay_queue
                self.replay_queue = list()
        if np.any(full_batch.true_done[0]):
            self.last_res = 0
            self.last_done = 0
            self.last_true_done = 0
            del self.replay_queue
            self.replay_queue = list()

    def sample_buffer(self, buffer):
        if np.random.random() > self.select_positive or len(self.replay_buffer) == 0:
        # print(len(buffer), len(self.replay_buffer))
        # if np.random.random() < (len(buffer) / (len(self.replay_buffer) + len(buffer))):
            # print("negative", (len(buffer) / (len(self.replay_buffer) + len(buffer))))
            print("negative")
            return buffer
        else:
            # print("positive", (len(buffer) / (len(self.replay_buffer) + len(buffer))))
            print("positive")
            return self.replay_buffer
            
    def step(self, rollouts, use_range=None):
        total_loss = 0
        for i in range(self.args.grad_epoch):
            # loss = self.step_fn(rollouts, i)
            if np.random.random() > self.select_positive:
                idxes, batch = rollouts.get_batch(self.args.batch_size)
            else:
                idxes, batch = self.rollouts.get_batch(self.args.batch_size)
            # vals = np.array([v[1] for v in b[i]])
            # if b[i][0][0] == 0:
            #     idxes, batch = rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))
            # else:
            #     idxes, batch = self.rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))

            loss = self.loss_calc(batch)
            self.step_optimizer(loss, RL=0)
            total_loss += loss.clone().detach()
            # print(q_values[0], q_values.grad[0])
        return PolicyLoss(total_loss/self.args.grad_epoch, None, loss, None, None, None)

    def step_fn(self, rollouts, i):
            self.optimizer.zero_grad()
            # if self.policy.QFunction.weight is not None:
            #     print("before", pytorch_model.unwrap(self.policy.QFunction.weight[:,:9]))
            # if i == 2:
            #     error
            vals = np.array([v[1] for v in b[i]])
            if b[i][0][0] == 0:
                idxes, batch = rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))
            else:
                idxes, batch = self.rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))
            
            S0 = pytorch_model.wrap(pytorch_model.unwrap(batch.values.state), cuda=self.args.cuda)
            S1 = pytorch_model.wrap(pytorch_model.unwrap(batch.values.next_state), cuda=self.args.cuda)
            P = pytorch_model.wrap(pytorch_model.unwrap(batch.values.param), cuda=self.args.cuda)

            # next_maxq = self.policy(batch.values.next_state, batch.values.param).Q_vals.max(dim=1)[0]

            next_output = self.option.policy.forward(S1, P)
            # if self.args.double_Q > 0: 
            #     double_output = self.policy.forward(batch.values.next_state, batch.values.param)
            #     next_value, _ = self.option.get_action(double_output.Q_best, next_output.Q_vals, next_output.std)
            # else:
            next_value = next_output.Q_vals.max(dim=1)[0]

            value_estimate = (next_value.detach().squeeze() * self.args.gamma) * (1-batch.values.done.squeeze().detach()) + batch.values.reward.squeeze().detach()
            
            output = self.policy.forward(S0, P)
            q_values, _ = self.option.get_action(batch.values.action, output.Q_vals, output.std)
            # q_values.retain_grad()
            
            q_loss = F.smooth_l1_loss(q_values.squeeze(), value_estimate.squeeze())#.norm(p=2) / self.args.batch_size
            (q_loss).backward()
            # print("before")
            self.optimizer.step()
            # print(rollouts.filled, b[i][0], vals, 
            #     np.argwhere(pytorch_model.unwrap(batch.values.state[0].reshape(20,20,3)[:,:,1]) > 0), 
            #     np.argwhere(pytorch_model.unwrap(batch.values.next_state[0].reshape(20,20,3)[:,:,1]) > 0), 
            #     np.argwhere(pytorch_model.unwrap(batch.values.param[0].reshape(20,20,1)[:,:,0]) > 0), 
            #     pytorch_model.unwrap(batch.values.action.squeeze()),
            #     pytorch_model.unwrap(batch.values.reward.squeeze()), pytorch_model.unwrap(batch.values.done.squeeze()),
            #     "ve", pytorch_model.unwrap(value_estimate.squeeze()[0]), pytorch_model.unwrap(q_values.squeeze()[0]), 
            #     "next", pytorch_model.unwrap(next_value.squeeze()[0]),
            #     "grad", pytorch_model.unwrap(q_values.grad[0]), pytorch_model.unwrap(q_loss)) 
            # print("after", pytorch_model.unwrap(self.policy.QFunction.weight[:,:9]))
            # print("policy", pytorch_model.unwrap(self.option.policy.QFunction.weight[:,:9]))
            q_values.detach()
            return q_loss.detach().item()

# learning_algorithms = {'ppo': PPO_optimizer, 'dqn': DQN_optimizer, 'a2c':
#                         A2C_optimizer, 'her': HER_optimizer, 'gsr': GSR_optimizer, 'ddpg': DDPG_optimizer}
