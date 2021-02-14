import os, collections, time
import gc, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from collections import deque

from Environments.environment_specification import ProxyEnvironment
from ReinforcementLearning.Policy.policy import pytorch_model
from file_management import save_to_pickle
from ReinforcementLearning.train_RL import Logger

def testRL(args, rollouts, logger, environment, environment_model, option, names, graph):
    full_state = environment_model.get_factored_state()
    last_full_state = environment_model.get_factored_state()
    if option.object_name == 'Raw':
        stack = torch.zeros([4,84,84])
        if args.cuda:
            stack = stack.cuda()
        stack = stack.roll(-1,0)
        stack[-1] = pytorch_model.wrap(environment.render_frame(), cuda=args.cuda)
    true_rewards = deque(maxlen=1000)
    true_dones = deque(maxlen=1000)
    last_done = True
    done = True
    success = [1]
    for i in range(args.num_iters):
        param, mask = option.get_param(full_state, last_done)
        
        action_chain, rl_outputs = option.sample_action_chain(full_state, param)
        option.step(full_state, action_chain)
        frame, factored, true_done = environment.step(action_chain[0].cpu().numpy())
        # Record information about next state
        next_full_state = environment_model.get_factored_state()
        true_rewards.append(environment.reward), true_dones.append(true_done)
        dones, rewards = option.terminate_reward(next_full_state, param, action_chain)
        done, last_done = dones[-1], dones[-2]
        option.step_timer(done)

        # refactor this into the option part
        option.record_state(full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
        print(param, option.timer, option.time_cutoff, option.get_state(full_state), action_chain[-1], last_done)
        # print(option.rollouts.get_values("done")[-1], option.rollouts.get_values("next_state")[-1], option.rollouts.get_values("param")[-1])

        full_state = next_full_state
        if args.return_form != "none":
            input_state = option.get_state(full_state, form=1, inp=0)
            next_value = option.forward(input_state.unsqueeze(0), param.unsqueeze(0)).values
            option.compute_return(args.gamma, option.rollouts.at-1, 1, next_value, return_max = 128, return_form=args.return_form)
        logger.log(i, option.behavior_policy.epsilon, args.log_interval, 1, None, option.rollouts, true_rewards, true_dones, success)

