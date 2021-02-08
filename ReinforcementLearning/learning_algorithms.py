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


class LearningOptimizer():
    def __init__(self, args, option):
        self.option = option
        self.policy = self.option.policy
        self.optimizer = self.initialize_optimizer(args, self.option.policy)
        self.args = args
        self.i = 0

    def initialize_optimizer(self, args, policy):
        # if args.model_form == "population":
        #     return PopOptim(model, args)
        if args.optim == "SGD":
            return torch.optim.SGD(policy.parameters(), lr=args.lr)
        elif args.optim == "RMSprop":
            print("optimparams", args.lr, args.eps, args.alpha)
            return optim.RMSprop(policy.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        elif args.optim == "Adam":
            return optim.Adam(policy.parameters(), args.lr)
            # return optim.Adam(policy.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError("Unimplemented optimization")

    def step_optim(self, loss, q_values):
        # print("maxgrad", self.args.max_grad_norm)
        self.optimizer.zero_grad()
        (loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
        # print("Q-values", q_values.grad)
        self.optimizer.step()

    def step_optimizer(self, loss, q_values, RL=0):
        '''
        steps the optimizer. This is shared between RL algorithms
        '''
        if RL == 0:
            self.step_optim(loss, q_values)
        else:
            raise NotImplementedError("Check that Optimization is appropriate")
        if self.args.double_Q > 0: # double Q learning targets
            if self.args.double_Q > 1: # double Q is a schedule
                if self.i % self.args.double_Q == 0 and self.i != 0:
                    new_params = self.policy.state_dict().copy()
                    self.option.policy.load_state_dict(new_params)
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
        value_estimate = (next_value.detach().squeeze() * self.args.gamma) * (1-batch.values.done.squeeze()) + batch.values.reward.squeeze()
        
        # print(time.time() - start)
        # start = time.time()
        # print(value_estimate, q_values)
        q_loss = F.smooth_l1_loss(value_estimate, q_values)#.norm(p=2) / self.args.batch_size
        # print(torch.stack((batch.values.object_state.squeeze(), batch.values.next_object_state.squeeze(), batch.values.param.squeeze(), batch.values.action.squeeze(), next_value.squeeze(), value_estimate), dim=1))
        # print("before", torch.cat((output.Q_vals, next_value.unsqueeze(1)), dim=1))
        # print(batch.values.reward.squeeze(), (next_value.detach().squeeze() * self.args.gamma + batch.values.reward.squeeze() - q_values[0].squeeze()).pow(2))
        # print("loss", q_loss, "reward", batch.values.reward.squeeze(), "q_values", q_values, "next_value", next_value.detach() * self.args.gamma)
        # print(time.time() - start)
        # start = time.time()
        return q_loss, q_values

    # def record_state(self, state, next_state, action_chain, rl_outputs, param, rewards, dones):
    #     # use record state to measure number of time steps for double-Q updates
    #     if self.double_Q_counter == self.args.double_Q:
    #         params = pytorch_model.upwrap(self.policy.get_parameters())


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
        if self.last_done == self.resample_timer or dones[-1] == 1: # either end of episode or end of timer
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
                # print("inserting", self.option.reward.get_reward(input_state, object_state, param))
                self.last_done = 0 # resamples as long 

    def step(self, rollouts, use_range=None):
        total_loss = 0
        for _ in range(self.args.grad_epoch):
            if np.random.random() > self.select_positive:
                idxes, batch = rollouts.get_batch(self.args.batch_size)
            else:
                idxes, batch = self.rollouts.get_batch(self.args.batch_size)
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
            q_loss, q_values = self.DQN_loss(batch)
            self.step_optimizer(q_loss, q_values, RL=0)
            rloutnext = self.policy.forward(batch.values.state, batch.values.param)
            # print("after", rloutnext.Q_vals)
            # total_loss += q_loss
        print(q_values[0], q_values.grad[0])
        return PolicyLoss(total_loss/self.args.grad_epoch, None, q_loss, None, None, None)


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
