import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from Networks.network import pytorch_model
import cv2
import time



from file_management import load_from_pickle

class LearningOptimizer():
    def __init__(self, args, option):
        self.option = option
        self.policy = self.option.policy # optimizer is built into policy, initialized outside
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
        self.rollouts = RLRollouts(option.rollouts.length, option.rollouts.shapes)
        if args.cuda:
            self.rollouts.cuda()
        self.last_done = 0
        self.select_positive = args.select_positive
        self.resample_timer = args.resample_timer
        self.use_interact = not args.true_environment
        if args.Q_updator == "DQN":
            self.internal_optimizer = DQN_optimizer(args, option)
        elif args.Q_updator == "DDPG":
            self.internal_optimizer = DDPG_optimizer(args, option)

        self.loss_calc = self.internal_optimizer.loss_calc
        self.policy = self.internal_optimizer.policy
        self.double_Q_counter = self.internal_optimizer.double_Q_counter
        self.optimizer = self.internal_optimizer.optimizer



    # def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
    def record_state(self, batch):
        super().record_state(i, state, next_state, action_chain, rl_outputs, param, rewards, dones)
        self.rollouts.append(**self.option.get_state_dict(state, next_state, action_chain, rl_outputs, param, rewards, dones))
        self.last_done += 1
        # if dones[-1] == 1: # option terminated, resample
        if (self.last_done == self.resample_timer and self.resample_timer > 0) or dones[-1] == 1: # either end of episode or end of timer
            interacting = True
            if self.use_interact:
                interact = self.option.dataset_model.interaction_model(self.option.get_state(next_state))
                interacting = self.option.dataset_model.check_interaction(interact)
            if interacting:
                last_done_at = (self.rollouts.at - self.last_done) % self.rollouts.length
                object_state = self.option.get_state(next_state, inp=1)
                num_vals = self.last_done
                broadcast_object_state = torch.stack([object_state.clone() for _ in range(num_vals)], dim=0)
                self.rollouts.insert_value(last_done_at, 0, num_vals, "param", broadcast_object_state)
                param = self.rollouts.get_values("param")[-self.last_done:]
                input_state = self.rollouts.get_values("next_state")[-self.last_done:]
                object_state = self.rollouts.get_values("next_object_state")[-self.last_done:]
                true_done = self.rollouts.get_values("true_done")[-self.last_done:]
                true_reward = self.rollouts.get_values("true_reward")[-self.last_done:]
                # print(self.rollouts.at, input_state.shape, object_state.shape, param.shape)
                self.rollouts.insert_value(last_done_at, 0, num_vals, "done", self.option.termination.check(input_state, object_state, param, true_done))
                self.rollouts.insert_value(last_done_at, 0, num_vals, "reward", self.option.reward.get_reward(input_state, object_state, param, true_reward))
                # print(pytorch_model.unwrap(self.rollouts.get_values("reward")[-self.last_done:]))
                # for i in range(self.last_done):
                #     # print(self.rollouts.at - self.last_done + i)
                #     state = new_states[self.rollouts.at - self.last_done + i]
                #     state[:,:,2] = new_params[self.rollouts.at - self.last_done + i]
                #     cv2.imshow('Example - Show image in window',pytorch_model.unwrap(state))
                #     cv2.waitKey(100)
                # print("inserting", self.option.reward.get_reward(input_state, object_state, param))
                self.last_done = 0 # resamples as long 

    def sample_buffer(self, buffer):
        if np.random.random() > self.select_positive:
            return buffer
        else:
            return self.buffer
            
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
