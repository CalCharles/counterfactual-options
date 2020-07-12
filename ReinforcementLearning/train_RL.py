import os, collections, time
import gc, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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


class Logger:
    def __init__(self, args):
        self.args = args
        self.num_iters = args.num_iters
        self.start = time.time()

    def log(self, i, policy_loss, rollouts):
        total_elapsed = i * self.num_iters
        end = time.time()
        final_rewards = rollouts.values.reward
        el, vl, al = policy_loss.values.entropy_loss
        logvals = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}, average_reward {}, true_reward median: {}, mean: {}, max: {}".format(
            i,
            total_elapsed,
            int(total_elapsed / (end - self.start)),
            rollouts.get_value('reward').mean(),
            np.median(final_rewards.cpu()),
            final_rewards.min(),
            final_rewards.max(),
            el,
            vl,
            al,
            torch.stack(average_rewards).sum() / acount,
            true_reward,
            mean_reward,
            best_reward,
        )
        self.start = end


def trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names):
    state = pytorch_model.wrap(environment_model.get_flattened_state(names=names), cuda=args.cuda)
    fullstate = environment_model.get_factored_state()
    print(state)
    last_state = torch.zeros(state.shape)
    if args.cuda:
        last_state = last_state.cuda()
    for i in range(args.num_iters):
        param, mask = option.model.sample(fullstate, 1, name=option.object_name)
        for j in range(args.num_steps):
            # store action, currently not storing action probabilities, might be necessary
            diff = state - last_state
            action_chain, done, reward = option.sample_action_chain(state, diff, param)
            rollouts.append({'state': environment_model.flatten_factored_state(state),
                             'state_diff': diff, 
                             'action': action_chain[-1], 
                             'param': param, 
                             'mask': mask, 
                             'reward': reward, 
                             'done': done})

            # managing the next state
            last_state = state.clone()
            environment.step(action_chain[0])
            state = environment_model.get_flattened_state(names=names)
        fullstate = environment_model.get_factored_state()
        policy_loss = learning_algorithm.step(rollouts)
        logger.log(i, policy_loss, proxy_environment.rollouts)
        option.save_network()