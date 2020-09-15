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
from ReinforcementLearning.policy import pytorch_model
from file_management import save_to_pickle

def sample_actions(probs, deterministic):  # TODO: why is this here?
    if deterministic is False:
        cat = torch.distributions.categorical.Categorical(probs.squeeze())
        action = cat.sample()
        action = action.unsqueeze(-1).unsqueeze(-1)
    else:
        action = probs.max(1)[1]
    return action


def unwrap_or_none(val):
    if val is not None:
        if type(val) == torch.tensor:
            return pytorch_model.unwrap(val)
        return val
    else:
        return -1.0

def true_action_prob(actions, num_actions):
    vals = torch.zeros((num_actions,))
    total = torch.zeros(num_actions)
    for action in actions:
        a = vals.clone()
        a[action.long()] = 1
        total += a
    return total, len(actions)

class Logger:
    def __init__(self, args):
        self.args = args
        self.num_steps = args.num_steps
        self.start = time.time()
        self.total = torch.zeros(4)
        self.length = 0

    def log(self, i, interval, policy_loss, rollouts, true_rewards, true_dones):
        true_rewards = np.array(true_rewards)
        total_elapsed = i * self.num_steps
        end = time.time()
        final_rewards = rollouts.get_values('reward')
        el, vl, al = policy_loss.entropy_loss, policy_loss.value_loss, policy_loss.action_loss
        logvals = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}, true_reward median: {}, mean: {}, episode: {}".format(
            i,
            total_elapsed,
            int(self.args.num_steps*interval / (end - self.start)),
            final_rewards.mean(),
            final_rewards.median(),
            final_rewards.min(),
            final_rewards.max(),
            el,
            vl,
            al,
            np.median(true_rewards),
            np.mean(true_rewards),
            np.sum(true_rewards) / np.sum(true_dones)
        )
        addtotal, length = true_action_prob(rollouts.values.true_action, 4) # disable this if actions not corresponding
        self.total += addtotal
        self.length += length
        if i % interval == 0:
            self.start = end
            print(logvals)
            print("action_probs", self.total / self.length)
            print("probs", rollouts.get_values('probs')[-1])
            print("Q_vals", rollouts.get_values('Q_vals')[-1])
            self.total = torch.zeros(4)
            self.length = 0




def testRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph):
    input_state = pytorch_model.wrap(environment_model.get_flattened_state(names=names), cuda=args.cuda)
    diff_state = pytorch_model.wrap(environment_model.get_flattened_state(names=[names[0]]), cuda=args.cuda)
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
    for i in range(args.num_iters):
        for j in range(args.num_steps):
            if last_done:
                if option.object_name == 'Raw':
                    param, mask = torch.tensor([1]), torch.tensor([1])
                else:
                    param, mask = option.dataset_model.sample(full_state, 1, name=option.object_name)

                return_update = rollouts.filled
                if args.cuda:
                    param, mask = param.cuda(), mask.cuda()
            action_chain, rl_output = option.sample_action_chain(full_state, param)
            if args.option_type == 'raw':
                input_state = stack.clone()
            frame, factored, true_done = environment.step(action_chain[0].cpu().numpy())
            next_full_state = environment_model.get_factored_state()
            true_rewards.append(environment.reward), true_dones.append(true_done)
            next_input_state = pytorch_model.wrap(environment_model.get_flattened_state(names=names), cuda=args.cuda)
            if args.option_type == 'raw':
                stack = stack.roll(-1,0)
                stack[-1] = pytorch_model.wrap(frame, cuda=args.cuda)
                next_input_state = stack.clone()
            done, reward, diff, last_done = option.terminate_reward(next_full_state, param, action_chain)
            rollouts.append(**{'state': input_state,
                                'next_state': next_input_state,
                             'state_diff': diff, 
                             'true_action': action_chain[0],
                             'action': action_chain[-1],
                             'probs': rl_output.probs,
                             'Q_vals': rl_output.Q_vals,
                             'param': param, 
                             'mask': mask, 
                             'reward': reward, 
                             'done': done})
            input_state = next_input_state
            full_state = next_full_state
        next_value = option.forward(input_state.unsqueeze(0), param).values
        rollouts.compute_return(args.gamma, return_update, args.num_steps, next_value, return_max = 128)
        logger.log(i, args.log_interval, policy_loss, rollouts, true_rewards, true_dones)

