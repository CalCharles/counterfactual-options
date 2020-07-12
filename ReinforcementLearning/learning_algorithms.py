import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from ReinforcementLearning.train_RL import sample_actions
from ReinforcementLearning.policy import pytorch_model
import cma, cv2
import time

class PolicyLoss():
    def __init__(self, value_loss, action_loss, q_loss, dist_entropy, entropy_loss, action_log_probs):
        self.value_loss, self.action_loss, self.q_loss, self.dist_entropy, self.entropy_loss, self.action_log_probs = value_loss, action_loss, q_loss, dist_entropy, entropy_loss, action_log_probs

class LearningOptimizer():
    def __init__(self, args, option):
        self.option = option
        self.optimizer = self.initialize_optimizer(args, self.option.policy)
        self.args = args

    def initialize_optimizer(self, args, policy):
        # if args.model_form == "population":
        #     return PopOptim(model, args)
        if args.optim == "RMSprop":
            return optim.RMSprop(policy.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        elif args.optim == "Adam":
            return optim.Adam(policy.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError("Unimplemented optimization")

    def step_optim(self, loss):
        self.optimizer.zero_grad()
        (loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def step_optimizer(self, i, loss, RL=0):
        '''
        steps the optimizer. This is shared between RL algorithms
        '''
        if RL == 0:
            self.step_optim(i, loss)
        if RL == 1: # double Q learning targets
            self.step_optim(i, loss)
            self.options[i] = (1-self.learner_args.tau) * self.options[i].parameters() + self.learner_args.tau * self.mains[i].parameters()
        else:
            raise NotImplementedError("Check that Optimization is appropriate")

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


class DQN_optimizer(LearningOptimizer):
    def step(self, args, rollouts, use_range=None):
        # # self.step_counter += 1
        total_loss = 0
        tpl = 0
        for _ in range(args.grad_epoch):
            batch = rollouts.get_batch(self.args.batch_size)
            output = self.option.forward(batch.values.state, batch.values.parameter)
            next_output = self.option.forward(batch.values.next_state, batch.values.parameter)
            q_values = self.option.get_action(batch.values.action, output.values.q_values)
            next_value = next_output.values.value
            q_loss = (next_value.detach() + batch.values.reward - q_values.squeeze()).norm().pow(2)
            self.step_optimizer(self.optimizer, self.option, q_loss, RL=self.RL)
            total_loss += q_loss
        return PolicyLoss(total_loss/args.grad_epoch, None, q_loss, None, None, torch.log(action_probs))

class PPO_optimizer(LearningOptimizer):
    def step(self, rollouts, use_range=None):
        for _ in range(self.args.grad_epoch):
            batch = rollouts.get_batch(self.args.batch_size)
            output = self.option.forward(batch.state, batch.parameter)
            values, probs = self.option.get_action(batch.action, output.values.value, output.values.probs)
            log_probs, old_log_probs = torch.log(probs + 1e-10).gather(1, batch.values.actions), torch.log(old_probs + 1e-10).gather(1, batch.values.actions)
            advantages = batch.returns.view(-1, 1) - values
            a = (advantages - advantages.mean())
            advantages = a / (advantages.std() + 1e-5)
            ratio = torch.exp(log_probs - old_log_probs.detach()).squeeze()
            surr1 = ratio * advantages.squeeze().detach()
            surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages.squeeze().detach()
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
            value_loss = (batch.returns - values).pow(2).mean()
            log_output_probs = torch.log(torch.sum(probs, dim=0) / probs.size(0) + 1e-10)
            dist_entropy = -(log_probs * probs).sum(-1).mean()
            self.step_optimizer(self.optimizer, self.option, value_loss * args.value_loss_coef + action_loss + entropy_loss, RL=self.RL)
        return PolicyLoss(value_loss, action_loss, dist_entropy, None, entropy_loss, action_log_probs)

learning_algorithms = {'ppo': PPO_optimizer, 'dqn': DQN_optimizer}
