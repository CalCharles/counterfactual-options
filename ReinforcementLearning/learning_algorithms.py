import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from ReinforcementLearning.train_RL import sample_actions, PolicyLoss
from ReinforcementLearning.Policy.policy import pytorch_model
from ReinforcementLearning.rollouts import RLRollouts
import cv2
import time

b = [[(1, 44), (1, 45), (1, 46), (1, 47)], [(1, 27), (1, 30), (1, 44), (1, 11)], [(0, 22), (0, 90), (0, 52), (0, 62)], [(1, 6), (1, 37), (1, 5), (1, 41)], [(0, 3), (0, 28), (0, 71), (0, 6)], [(1, 9), (1, 16), (1, 28), (1, 21)], [(0, 64), (0, 60), (0, 39), (0, 47)], [(1, 29), (1, 2), (1, 25), (1, 45)], [(0, 68), (0, 4), (0, 82), (0, 23)], [(0, 81), (0, 71), (0, 17), (0, 77)], [(0, 45), (0, 72), (0, 49), (0, 96)], [(1, 36), (1, 30), (1, 19), (1, 1)], [(0, 63), (0, 21), (0, 93), (0, 72)], [(0, 99), (0, 66), (0, 73), (0, 11)], [(0, 25), (0, 18), (0, 26), (0, 25)], [(0, 64), (0, 77), (0, 95), (0, 70)], [(0, 13), (0, 97), (0, 92), (0, 58)], [(0, 49), (0, 37), (0, 42), (0, 19)], [(1, 10), (1, 0), (1, 31), (1, 36)], [(1, 30), (1, 40), (1, 17), (1, 30)], [(0, 0), (0, 57), (0, 62), (0, 70)], [(1, 27), (1, 47), (1, 2), (1, 34)], [(0, 0), (0, 67), (0, 62), (0, 1)], [(1, 18), (1, 24), (1, 37), (1, 11)], [(1, 28), (1, 32), (1, 25), (1, 15)], [(1, 20), (1, 8), (1, 48), (1, 41)], [(0, 6), (0, 31), (0, 89), (0, 52)], [(1, 1), (1, 43), (1, 10), (1, 0)], [(1, 35), (1, 46), (1, 6), (1, 21)], [(1, 4), (1, 16), (1, 22), (1, 26)], [(0, 51), (0, 70), (0, 27), (0, 7)], [(0, 71), (0, 10), (0, 68), (0, 41)], [(1, 28), (1, 40), (1, 15), (1, 10)], [(0, 75), (0, 19), (0, 17), (0, 62)], [(0, 41), (0, 96), (0, 88), (0, 47)], [(1, 12), (1, 29), (1, 29), (1, 49)], [(1, 40), (1, 49), (1, 19), (1, 18)], [(0, 48), (0, 68), (0, 20), (0, 5)], [(0, 43), (0, 39), (0, 61), (0, 38)], [(0, 1), (0, 98), (0, 69), (0, 46)], [(0, 37), (0, 79), (0, 38), (0, 38)], [(1, 2), (1, 5), (1, 7), (1, 23)], [(0, 70), (0, 91), (0, 87), (0, 67)], [(1, 34), (1, 21), (1, 17), (1, 34)], [(0, 94), (0, 24), (0, 90), (0, 35)], [(0, 66), (0, 67), (0, 61), (0, 39)], [(1, 48), (1, 8), (1, 43), (1, 31)], [(1, 31), (1, 32), (1, 31), (1, 34)], [(0, 81), (0, 98), (0, 63), (0, 94)], [(0, 91), (0, 33), (0, 38), (0, 95)]]

class LearningOptimizer():
    def __init__(self, args, option):
        self.option = option
        self.policy = self.option.policy
        self.optimizer = self.initialize_optimizer(args, self.option.policy)
        self.args = args
        self.i = 0
        self.updated = 0

    def initialize_optimizer(self, args, policy):
        # if args.model_form == "population":
        #     return PopOptim(model, args)
        if args.optim == "SGD":
            return torch.optim.SGD(policy.parameters(), lr=args.lr)
        elif args.optim == "RMSprop":
            print("optimparams", args.lr, args.eps, args.alpha)
            return optim.RMSprop(policy.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        elif args.optim == "Adam":
            print("using ADAM")
            # return optim.Adam(policy.parameters(), args.lr)
            return optim.Adam(policy.parameters(), .0001)
            # return optim.Adam(policy.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError("Unimplemented optimization")

    def step_optim(self, loss):
        self.optimizer.zero_grad()
        (loss).backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
        self.optimizer.step()

    def step_optimizer(self, loss, RL=0):
        '''
        steps the optimizer. This is shared between RL algorithms
        '''
        if RL == 0:
            self.step_optim(loss)
        else:
            raise NotImplementedError("Check that Optimization is appropriate")
        if self.args.double_Q > 0: # double Q learning targets
            if self.args.double_Q > 1: # double Q is a schedule
                if self.i % self.args.double_Q == 0 and self.i != 0 and self.updated != self.i:
                    print("update policy params")
                    new_params = self.policy.state_dict().copy()
                    self.option.policy.load_state_dict(new_params)
                    self.updated = self.i
            else:
                old_params = pytorch_model.unwrap(self.option.policy.get_parameters())
                new_params = pytorch_model.unwrap(self.policy.get_parameters())
                params = (1-self.args.double_Q) * old_params + self.args.double_Q * new_params
                self.option.policy.set_parameters(params)

    def step(self, args, rollouts):
        '''
        step the optimizer once. This computes the gradient if necessary from rollouts, and then steps the appropriate values
        in train_models. It can step either the current, or all the models.Usually accompanied with a step_optimizer
        function which steps the inner optimization algorithm
        output entropy is the batch entropy of outputs
        returns Value loss, policy loss, distribution entropy, output entropy, entropy loss, action log probabilities
        '''
        # raise NotImplementedError("step should be overriden")
        pass

    def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
        self.i = i
        pass


class DQN_optimizer(LearningOptimizer):
    def __init__(self, args, option): 
        super().__init__(args, option)
        # initialize double Q network
        self.double_Q_counter = 0
        if args.double_Q > 0:
            self.policy = type(option.policy)(**vars(args)) # initialize a copy of the current policy
            params = pytorch_model.unwrap(self.option.policy.get_parameters())
            self.policy.set_parameters(params) # uses numpy which means cloned
            self.optimizer = self.initialize_optimizer(args, self.policy)
        else:
            self.policy = option.policy

    def DQN_loss(self, batch):
        # start = time.time()
        # print(time.time() - start)
        # start = time.time()
        next_output = self.option.policy.forward(batch.values.next_state, batch.values.param)
        next_value = next_output.values

        output = self.policy.forward(batch.values.state, batch.values.param)
        # next_output = self.policy.forward(batch.values.next_state, batch.values.param)
        # print(time.time() - start)
        # start = time.time()
        q_values, _ = self.option.get_action(batch.values.action, output.Q_vals, output.std)
        # print(time.time() - start)
        # start = time.time()
        # q_values.retain_grad()

        # if self.args.double_Q > 0:
        #     double_output = self.policy.forward(batch.values.next_state, batch.values.param)
        #     next_value, _ = self.option.get_action(double_output.Q_best, next_output.Q_vals, next_output.std)
        # else:
        
        # print(double_output.Q_best[0], next_value[0], next_output.values[0])
        value_estimate = (next_value.detach().squeeze() * self.args.gamma) * (1-batch.values.done.squeeze().detach()) + batch.values.reward.squeeze().detach()
        # np.set_printoptions(threshold=np.inf, precision=3)
        # print(pytorch_model.unwrap(self.policy.mean.reshape(20,20,3).transpose(2,1).transpose(1,0)))
        # error
        # for v in batch.values.state:
        #     cv2.imshow("mystate", pytorch_model.unwrap(v.reshape(20,20,3)))
        #     cv2.waitKey(4000)
        # print(pytorch_model.unwrap(q_values), pytorch_model.unwrap(next_value), pytorch_model.unwrap(value_estimate))
        # print(time.time() - start)
        # start = time.time()
        # print(value_estimate, q_values)
        q_loss = F.smooth_l1_loss(q_values, value_estimate)#.norm(p=2) / self.args.batch_size

        # print(torch.stack((batch.values.object_state.squeeze(), batch.values.next_object_state.squeeze(), batch.values.param.squeeze(), batch.values.action.squeeze(), next_value.squeeze(), value_estimate), dim=1))
        # print("before", torch.cat((output.Q_vals, next_value.unsqueeze(1)), dim=1))
        # print(batch.values.reward.squeeze(), (next_value.detach().squeeze() * self.args.gamma + batch.values.reward.squeeze() - q_values[0].squeeze()).pow(2))
        # print("loss", q_loss, "reward", batch.values.reward.squeeze(), "q_values", q_values, "next_value", next_value.detach() * self.args.gamma)
        # print(time.time() - start)
        # start = time.time()
        return q_loss

    # def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
    #     # use record state to measure number of time steps for double-Q updates
    #     if self.args.double_Q > 0 self.double_Q_counter == self.args.double_Q:
    #         params = pytorch_model.upwrap(self.policy.get_parameters())
    #         self.double_Q_counter = 0
    #     self.double_Q_counter += 1


    def step(self, rollouts, use_range=None):
        # # self.step_counter += 1
        total_loss = 0
        for _ in range(self.args.grad_epoch):
            # start = time.time()
            weights = None
            if len(self.args.prioritized_replay) > 0:
                rew = rollouts.get_values(self.args.prioritized_replay) # TODO: use different values than just max reward
                total = rew.sum() + self.args.weighting_lambda * len(rew)
                weights =  (rew + self.args.weighting_lambda) / total
                weights = weights.squeeze().cpu().numpy()
            idxes, batch = rollouts.get_batch(self.args.batch_size, weights = weights)
            # print("batch", time.time() - start)
            # start = time.time()
            q_loss = self.DQN_loss(batch)
            # print("loss", time.time() - start)
            # start = time.time()
            self.step_optimizer(q_loss, RL=0)
            # print("step", time.time() - start)
            total_loss += q_loss
        return PolicyLoss(total_loss/self.args.grad_epoch, None, q_loss, None, None, None)

class GSR_optimizer(DQN_optimizer): # goal search replay
    def step(self, rollouts, use_range=None):
        total_loss = 0
        for _ in range(self.args.grad_epoch):
            idxes, batch = rollouts.get_batch(self.args.batch_size)
            possible_params = self.option.get_possible_parameters()
            rewards = []
            params = []
            # checks all parameters for rewards, and takes high reward parameters with probability args.search_rate, and a random parameter with probability 1-args.search_rate
            # for object_state, diff in zip(batch.values.next_object_state, batch.values.state_diff):
                # print(last_state, object_state, param, diff, self.option.reward.get_reward(object_state, diff, param))
                # rewarded_params = list()
                # no_reward = list()
                # for param, mask in possible_params:
                #     reward = self.option.reward.get_reward(object_state*mask, diff*mask, param)
                #     if reward == 1:
                #         rewarded_params.append((param, reward))
                #     else:
                #         no_reward.append((param, reward))
                # if np.random.rand() < self.args.search_rate and len(rewarded_params) > 0:
                #     sample_from = rewarded_params
                # else:
                #     sample_from = rewarded_params + no_reward
                # idx = np.random.randint(len(sample_from))
                # param, reward = sample_from[idx]
                # params.append(param)
                # rewards.append(reward)
            # checks the pre-saved rewards
            possible_params = self.option.get_possible_parameters()
            for all_reward, max_reward in zip(batch.values.all_reward, batch.values.max_reward):
                if max_reward <= 0 or np.random.rand() >= self.args.search_rate:
                    idx = np.random.randint(len(possible_params))
                else:
                    possible_idxes = (all_reward == max_reward).nonzero()
                    # print(all_reward, max_reward, possible_idxes)
                    idx = np.random.randint(len(possible_idxes))
                params.append(possible_params[idx][0])
                rewards.append(all_reward[idx])

            params = torch.stack(params, dim=0)
            if self.args.cuda:
                rewards = pytorch_model.wrap(rewards, cuda=self.args.cuda)
            batch.values.param = params
            batch.values.reward = rewards
            q_loss = self.DQN_loss(batch)
            self.step_optimizer(q_loss, RL=0)
            total_loss += q_loss
        return PolicyLoss(total_loss/self.args.grad_epoch, None, q_loss, None, None, None)

class HER_optimizer(DQN_optimizer):
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


    def record_state(self, i, state, next_state, action_chain, rl_outputs, param, rewards, dones):
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

            loss = self.DQN_loss(batch)
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
            q_values.retain_grad()
            
            q_loss = F.smooth_l1_loss(q_values.squeeze(), value_estimate.squeeze())#.norm(p=2) / self.args.batch_size
            (q_loss).backward()
            # print("before")
            self.optimizer.step()
            print(rollouts.filled, b[i][0], vals, 
                np.argwhere(pytorch_model.unwrap(batch.values.state[0].reshape(20,20,3)[:,:,1]) > 0), 
                np.argwhere(pytorch_model.unwrap(batch.values.next_state[0].reshape(20,20,3)[:,:,1]) > 0), 
                np.argwhere(pytorch_model.unwrap(batch.values.param[0].reshape(20,20,1)[:,:,0]) > 0), 
                pytorch_model.unwrap(batch.values.action.squeeze()),
                pytorch_model.unwrap(batch.values.reward.squeeze()), pytorch_model.unwrap(batch.values.done.squeeze()),
                "ve", pytorch_model.unwrap(value_estimate.squeeze()[0]), pytorch_model.unwrap(q_values.squeeze()[0]), 
                "next", pytorch_model.unwrap(next_value.squeeze()[0]),
                "grad", pytorch_model.unwrap(q_values.grad[0]), pytorch_model.unwrap(q_loss)) 
            # print("after", pytorch_model.unwrap(self.policy.QFunction.weight[:,:9]))
            # print("policy", pytorch_model.unwrap(self.option.policy.QFunction.weight[:,:9]))
            q_values.detach()
            return q_loss.detach().item()


    def step_fn_old(self, rollouts, i):
            self.optimizer.zero_grad()
            # if np.random.random() > self.select_positive:
            #     idxes, batch = rollouts.get_batch(self.args.batch_size)
            # else:
            #     idxes, batch = self.rollouts.get_batch(self.args.batch_size)
            vals = np.array([v[1] for v in b[i]])
            if b[i][0][0] == 0:
                idxes, batch = rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))
            else:
                idxes, batch = self.rollouts.get_batch(self.args.batch_size, idxes = np.array(vals))
            # rewards = list()
            # params = list()
            # # params, masks = self.option.dataset_model.sample(batch.values.state, batch.length, both = self.args.use_both == 2, diff=self.args.use_both == 1, name=self.option.object_name)
            # # if self.args.cuda:
            # #     params, masks = params.cuda(), masks.cuda()
            # # when using the actual reward function and not all_rewards
            # # for last_state, object_state, diff, param_idx, mask in zip(batch.values.state, batch.values.next_object_state, batch.values.state_diff, params, masks):
            # #     # rewards.append(self.option.reward.get_reward(object_state*mask, diff*mask, param)) # when all_reward not used
            # #     print(object_state*mask, diff*mask, param)
            # possible_params = self.option.get_possible_parameters()
            # for all_reward, param_idx in zip( batch.values.all_reward, [np.random.randint(len(possible_params)) for i in range(self.args.batch_size)]):
            #     rewards.append(float(all_reward[param_idx].squeeze().clone().cpu().numpy()))
            #     params.append(possible_params[param_idx][0])
            # params = torch.stack(params, dim=0)
            # if self.args.cuda:
            #     params = params.cuda()
            #     # print(rewards)
            #     rewards = pytorch_model.wrap(rewards, cuda=self.args.cuda)
            # batch.values.param = params
            # batch.values.reward = rewards
            # q_loss, q_values = self.DQN_loss(batch)


            output = self.policy.forward(batch.values.state, batch.values.param)
            # print(time.time() - start)
            # start = time.time()
            next_output = self.option.policy.forward(batch.values.next_state, batch.values.param)
            # next_output = self.policy.forward(batch.values.next_state, batch.values.param)
            # print(time.time() - start)
            # start = time.time()
            q_values, _ = self.option.get_action(batch.values.action, output.Q_vals, output.std)
            # print(time.time() - start)
            # start = time.time()
            q_values.retain_grad()
            if self.args.double_Q > 0:
                double_output = self.policy.forward(batch.values.next_state, batch.values.param)
                next_value, _ = self.option.get_action(double_output.Q_best, next_output.Q_vals, next_output.std)
            else:
                next_value = next_output.values
            # print(double_output.Q_best[0], next_value[0], next_output.values[0])
            value_estimate = (next_value.detach().squeeze() * self.args.gamma) * (1-batch.values.done.squeeze().detach()) + batch.values.reward.squeeze().detach()
            # np.set_printoptions(threshold=np.inf, precision=3)
            # print(pytorch_model.unwrap(self.policy.mean.reshape(20,20,3).transpose(2,1).transpose(1,0)))
            # error
            # for v in batch.values.state:
            #     cv2.imshow("mystate", pytorch_model.unwrap(v.reshape(20,20,3)))
            #     cv2.waitKey(4000)
            # print(pytorch_model.unwrap(q_values), pytorch_model.unwrap(next_value), pytorch_model.unwrap(value_estimate))
            # print(time.time() - start)
            # start = time.time()
            # print(value_estimate, q_values)
            q_loss = F.smooth_l1_loss(q_values, value_estimate)#.norm(p=2) / self.args.batch_size

            
            (q_loss).backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            # output = self.policy.forward(batch.values.state, batch.values.param)
            # next_output = self.option.policy.forward(batch.values.next_state, batch.values.param)
            # q_values, _ = self.option.get_action(batch.values.action, output.Q_vals, output.std)
            # print("input", pytorch_model.unwrap(batch.values.state[0].reshape(20,20,3).transpose(2,1).transpose(1,0)))
            # print(pytorch_model.unwrap(rollouts.get_values("action")).tolist() + pytorch_model.unwrap(self.rollouts.get_values("action")).tolist())
            # print("before", pytorch_model.unwrap(batch.values.action), 
            #     pytorch_model.unwrap(output.Q_vals),
            #     pytorch_model.unwrap(q_values),
            #     "next", pytorch_model.unwrap(batch.values.reward), pytorch_model.unwrap(batch.values.done),
            #     pytorch_model.unwrap(next_output.values),
            #     pytorch_model.unwrap(q_values), pytorch_model.unwrap(q_loss))        
            # error
            # self.step_optimizer(q_loss, q_values, RL=0)
            print(rollouts.filled, b[i][0], vals, 
                np.argwhere(pytorch_model.unwrap(batch.values.state[0].reshape(20,20,3)[:,:,1]) > 0), 
                np.argwhere(pytorch_model.unwrap(batch.values.next_state[0].reshape(20,20,3)[:,:,1]) > 0), 
                np.argwhere(pytorch_model.unwrap(batch.values.param[0].reshape(20,20,1)[:,:,0]) > 0), 
                pytorch_model.unwrap(batch.values.reward[0]), pytorch_model.unwrap(batch.values.done[0]),
                pytorch_model.unwrap(q_values[0]), pytorch_model.unwrap(q_values.grad[0]), pytorch_model.unwrap(q_loss))        
            # batch = rollouts.get_values("state").reshape(-1,20,20,3)
            # for i in range(100):
            #     print(i)
            #     cv2.imshow("buffer states", pytorch_model.unwrap(batch[i]))
            #     cv2.waitKey(200)

            # cv2.imshow("learn", pytorch_model.unwrap(batch.values.state[0].reshape(20,20,3)))
            # cv2.waitKey(1000)
            # rloutnext = self.policy.forward(batch.values.state, batch.values.param)
            # print("after", rloutnext.Q_vals)
            return q_loss.detach()


class PPO_optimizer(LearningOptimizer):
    def step(self, rollouts, use_range=None):
        for _ in range(self.args.grad_epoch):
            idxes, batch = rollouts.get_batch(self.args.batch_size)
            output = self.option.forward(batch.values.state, batch.values.param)
            probs, log_probs = self.option.get_action(batch.values.action, output.probs, output.std)
            old_probs, old_log_probs = self.option.get_action(batch.values.action, batch.values.probs, batch.values.std)
            values = output.values
            log_probs, old_log_probs = torch.log(probs + 1e-10), torch.log(old_probs + 1e-10)
            advantages = batch.values.returns.view(-1, 1) - values
            a = (advantages - advantages.mean())
            advantages = a / (advantages.std() + 1e-5)
            ratio = torch.exp(log_probs - old_log_probs.detach()).squeeze()
            surr1 = ratio * advantages.squeeze().detach()
            surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages.squeeze().detach()
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            value_loss = (batch.values.returns - values).pow(2).mean()
            dist_entropy = -(log_probs * probs).sum(-1).mean()
            entropy_loss = -dist_entropy * self.args.entropy_coef
            self.step_optimizer(value_loss * self.args.value_loss_coef + action_loss + entropy_loss, RL=0)
        return PolicyLoss(value_loss, action_loss, dist_entropy, None, entropy_loss, log_probs)

class A2C_optimizer(LearningOptimizer):
    def step(self, rollouts, use_range=None):
        idxes, batch = rollouts.get_batch(self.args.batch_size, ordered=True)
        output = self.option.forward(batch.values.state, batch.values.param)
        log_probs, probs = self.option.get_action(batch.values.action, output.probs, output.std)
        values = output.values
        advantages = batch.values.returns.detach() - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * log_probs.unsqueeze(1)).mean()
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        entropy_loss = -dist_entropy * self.args.entropy_coef
        # print(value_loss, action_loss, "lp", output.log_probs,"p", output.probs)
        # print(output.values, output.probs, batch.values.returns.detach() - values, advantages.detach() * log_probs.unsqueeze(1), batch.values.returns, batch.values.reward, batch.values.action)
        # print(output.values, output.probs, dist_entropy, advantages, batch.values.returns, batch.values.reward, batch.values.action, action_loss, batch.values.param, batch.values.state)
        self.step_optimizer(value_loss * self.args.value_loss_coef + action_loss + entropy_loss, RL=0)
        # print(value_loss, action_loss, dist_entropy, entropy_loss)
        return PolicyLoss(value_loss, action_loss, dist_entropy, None, entropy_loss, log_probs)



learning_algorithms = {'ppo': PPO_optimizer, 'dqn': DQN_optimizer, 'a2c': A2C_optimizer, 'her': HER_optimizer, 'gsr': GSR_optimizer}
