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
from ReinforcementLearning.rollouts import RLRollouts
from file_management import save_to_pickle

class PolicyLoss():
    def __init__(self, value_loss, action_loss, q_loss, dist_entropy, entropy_loss, action_log_probs):
        self.value_loss, self.action_loss, self.q_loss, self.dist_entropy, self.entropy_loss, self.action_log_probs = value_loss, action_loss, q_loss, dist_entropy, entropy_loss, action_log_probs

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
    def __init__(self, args, option):
        self.args = args
        self.num_steps = args.num_steps
        self.start = time.time()
        self.total = torch.zeros(4)
        self.length = 0
        self.logger_rollout = RLRollouts(args.log_interval * args.num_steps, option.rollouts.shapes)


    def log(self, i, interval, steps, policy_loss, rollouts, true_rewards, true_dones):
        true_rewards = np.array(true_rewards)
        total_elapsed = i * self.num_steps
        end = time.time()
        addtotal, length = true_action_prob(rollouts.get_values('true_action')[-steps:], 4) # disable this if actions not corresponding
        self.logger_rollout.insert_rollout(rollouts, a=1, add=True)
        self.total += addtotal
        self.length += length
        if i % interval == 0:
            final_rewards = self.logger_rollout.get_values('reward')
            el, vl, al = policy_loss.entropy_loss, policy_loss.value_loss, policy_loss.action_loss
            logvals = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.3f}/{:.3f}, min/max reward {:.1f}/{:.1f}, entropy {}, value loss {}, policy loss {}, true_reward median: {}, mean: {}, episode: {}".format(
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
            self.start = end
            print(logvals)
            print("action_probs", self.total / self.length)
            print("probs", rollouts.get_values('probs')[-1])
            print("Q_vals", rollouts.get_values('Q_vals')[-1])
            self.total = torch.zeros(4)
            self.length = 0

def trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph):
    # initialize states. input state/stack goes to the policy, diff state keeps last two states
    # full state, last full state goes into the ground truth forward model
    input_state = pytorch_model.wrap(environment_model.get_flattened_state(names=names), cuda=args.cuda)
    diff_state = pytorch_model.wrap(environment_model.get_flattened_state(names=[names[0]]), cuda=args.cuda)
    full_state = environment_model.get_factored_state()
    last_full_state = environment_model.get_factored_state()


    # refactor this into the option section
    if option.object_name == 'Raw':
        stack = torch.zeros([4,84,84]).detach()
        if args.cuda:
            stack = stack.cuda()
        stack = stack.roll(-1,0)
        stack[-1] = pytorch_model.wrap(environment.render_frame(), cuda=args.cuda).detach()

    # initialize information about ground truth rewards and dones for recording
    true_rewards = deque(maxlen=1000)
    true_dones = deque(maxlen=1000)
    done_lengths = deque(maxlen=1000)
    done_count = 0

    # 
    object_state, next_object_state = None, None
    last_done = True
    done = True

    # policy loss is for logging purposes only
    policy_loss = PolicyLoss(None, None, None, None, None, None)
    if args.warm_up > 0:
        option.set_behavior_epsilon(1)

    # an iteration corresponds to a learning step
    for i in range(args.num_iters):
        return_update = rollouts.at
        for j in range(args.num_steps):
            start = time.time()

            # resamples parameter and mask if last_done is true
            param, mask = option.get_param(full_state, last_done)

            # store action, currently not storing action probabilities, might be necessary
            # TODO: include the diffs for each level of the option internally
            # use the correct termination for updating the diff (internaly in option?)
            action_chain, rl_outputs = option.sample_action_chain(full_state, param)

            # step the environment
            option.step(full_state, action_chain)
            frame, factored, true_done = environment.step(action_chain[0].cpu().numpy())
            # Record information about next state
            next_full_state = environment_model.get_factored_state()
            true_rewards.append(environment.reward), true_dones.append(true_done)

            # progress the option and then calculate if terminated
            dones, rewards = option.terminate_reward(next_full_state, param, action_chain)
            done, last_done = dones[-1], dones[-2]

            # done count measures number of terminations
            option.step_timer(done)
            done_count += 1
            if done:
                done_lengths.append(done_count)
                done_count = 0

            # refactor this into the option part
            option.record_state(full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
            if args.train: # it may be necessary for the learning algorithm to in parallel store state information
                learning_algorithm.record_state(full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
            full_state = next_full_state


        if args.return_form != "none":
            input_state = option.get_state(full_state, form=1, inp=0)
            next_value = option.forward(input_state.unsqueeze(0), param.unsqueeze(0)).values
            option.compute_return(args.gamma, return_update, args.num_steps, next_value, return_max = 128, return_form=args.return_form)
        if option.rollouts.get_values("reward").sum() > 0:
            print("located reward: ", option.timer)

        if i >= args.warm_up:
            if args.epsilon_schedule > 0 and i % args.epsilon_schedule == 0:
                args.epsilon = args.epsilon * .5
                print("epsilon", args.epsilon)
            option.set_behavior_epsilon(args.epsilon)
            logger.log(i, args.log_interval, args.num_steps, policy_loss, option.rollouts, true_rewards, true_dones)
            # print("log", time.time() - start)
            # start = time.time()
            if args.train:
                policy_loss = learning_algorithm.step(option.rollouts)
                # print("train", time.time() - start)
                # start = time.time()
            if args.save_interval > 0 and (i+1) % args.save_interval == 0:
                option.save(args.save_dir)
                graph.save_graph(args.save_graph, [args.object])

    return done_lengths