import os, collections, time
import gc, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from collections import deque
import tianshou as ts

from Environments.environment_specification import ProxyEnvironment
from Networks.network import pytorch_model
from file_management import save_to_pickle, load_from_pickle

forced_actions = [0, 3, 2, 3, 1, 0, 1, 1, 0, 2, 1, 2, 2, 1, 1, 3, 1, 1, 3, 3, 2, 3, 1, 0, 0, 0, 3, 1, 3, 0, 1, 2, 0, 0, 3, 2, 3, 3, 2, 1, 1, 3, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 1, 2, 3, 0, 3, 1, 2, 3, 2, 0, 3, 1, 3, 1, 1, 2, 1, 3, 0, 1, 2, 0, 0, 2, 2, 3, 2, 1, 0, 0, 1, 3, 2, 0, 2, 2, 3, 1, 1, 0, 2, 1, 3, 2, 1, 2, 2, 2, 0]

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
        self.length = 0
        self.option = option
        self.action_tensor()
        self.logger_rollout = RLRollouts(args.log_interval * args.num_steps, option.rollouts.shapes)

    def action_tensor(self):
        if self.option.environment_model.environment.discrete_actions:
            self.total = pytorch_model.wrap(torch.zeros(self.option.environment_model.environment.num_actions), cuda=self.args.cuda)
        else:
            self.total = pytorch_model.wrap(torch.zeros(self.option.environment_model.environment.action_shape), cuda=self.args.cuda)


    def log(self, i, epsilon, interval, steps, policy_loss, rollouts, true_rewards, true_dones, true_success, continuous_actions = False):
        true_rewards = np.array(true_rewards)
        total_elapsed = i * self.num_steps
        end = time.time()
        if continuous_actions:
            actions = rollouts.get_values('true_action')[-steps:]
            length = len(actions)
            addtotal = actions.mean(dim=0)
        else:
            addtotal, length = true_action_prob(rollouts.get_values('true_action')[-steps:], 4) # disable this if actions not corresponding
        self.logger_rollout.insert_rollout(rollouts, a=1, add=True)
        self.total += addtotal
        self.length += length
        if i % interval == 0:
            final_rewards = self.logger_rollout.get_values('reward')
            if policy_loss is not None:
                el, vl, al = policy_loss.entropy_loss, policy_loss.value_loss, policy_loss.action_loss
            else:
                el, vl, al = None, None, None
            logvals = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.3f}/{:.3f}, min/max reward {:.2f}/{:.2f}, entropy {}, value loss {}, policy loss {}, true_reward median: {:.3f}, mean: {:.3f}, episode: {:.2f}, success: {:.2f}, epsilon: {}".format(
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
                np.sum(true_rewards) / np.sum(true_dones),
                np.sum(true_success) / 100,
                epsilon
            )
            self.start = end
            print(logvals)
            print("action_probs", self.total / self.length)
            if not continuous_actions:
                print("probs", rollouts.get_values('probs')[-1])
            print("Q_vals", rollouts.get_values('Q_vals')[-1])
            self.action_tensor()
            self.length = 0

def TSTrainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph):
    train_collector = ts.data.Collector(option.policy, environment, ts.data.ReplayBuffer(args.buffer_len, 1), exploration_noise=True)
    test_collector = ts.data.Collector(option.policy, environment)
    # train_collector = ts.data.Collector(option.policy, environment, ts.data.ReplayBuffer(args.buffer_len), preprocess_fn=option.get_env_state, exploration_noise=True)

    train_collector.collect(n_step=args.pretrain_iters, random=True)

    total_steps = 0
    for i in range(args.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.num_steps)
        total_steps += args.num_steps
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        if i % args.log_interval == 0:
            # option.policy.set_eps(0.05)
            # test_perf = list()
            # for j in range(10):
            #     result = test_collector.collect(n_episode=1)
            #     test_perf.append(result["rews"].mean())
            # print(f'Test mean returns: {np.array(test_perf).mean()}')

            result = test_collector.collect(n_episode=10)
            print("Steps: ", total_steps)
            print(f'Test mean returns: {result["rews"].mean()}')
            # print(environment.spec.reward_threshold)
            # if result['rews'].eman() >= environment.spec.reward_threshold: 
            #     print(f'Finished training! Test mean returns: {result["rews"].mean()}')
            #     break
            # else:
            #     # back to training eps
            #     option.policy.set_eps(0.1)
        # if i == 10:
        #     print(train_collector.buffer.act[:100], train_collector.buffer.obs[:100], train_collector.buffer.rew[:100], train_collector.buffer.done[:100])

        # train option.policy with a sampled batch data from buffer
        losses = option.policy.update(args.batch_size, train_collector.buffer)
        if i % args.log_interval == 0:
            print(losses, result['rews'].mean())
    train_collector.buffer.save_hdf5("data/working_rollouts.hdf5")

def collect_test_trials(args, test_collector, i, total_steps, test_perf, suc, random=False):
    '''
    TODO: still need an objective measure for performance
    '''
    test_collector.reset()
    trials = args.test_trials
    if random:
        trials = args.test_trials * 10
    for j in range(trials):
        if args.max_steps > 0: result = test_collector.collect(n_episode=1, n_step=args.max_steps, new_param=True, random=random)
        else: result = test_collector.collect(n_episode=1, new_param=True, random=random)
        test_perf.append(result["rews"].mean())
        suc.append(float(result["n/st"] != args.max_steps and result["true_done"] < 1))
    if random:
        print("Initial trials: ", trials)
    else:
        print("Iters: ", i, "Steps: ", total_steps)
    mean_perf, mean_suc = np.array(test_perf).mean(), np.array(suc).mean()
    print(f'Test mean returns: {mean_perf}', f"Success: {mean_suc}")
    return mean_perf, mean_suc


def trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph):
    # full_state = environment_model.get_state()
    last_done = 0 # assumes that the first state is not done
    train_collector.collect(n_step=args.pretrain_iters, random=True) # param doesn't matter with random actions
    initial_perf, initial_suc = list(), list()
    initial_perf, initial_suc = collect_test_trials(args, test_collector, 0, 0, initial_perf, initial_suc)

    total_steps = 0
    # TODO: might need to put interaction on a schedule also
    if args.epsilon_schedule:
        epsilon = 1
        option.policy.set_eps(epsilon)
    test_perf, suc = deque(maxlen=200), deque(maxlen=200)
    print("begin train loop")
    for i in range(args.num_iters):  # total step
        # print("training collection")
        # full_state = environment_model.get_state()
        collect_result = train_collector.collect(n_step=args.num_steps) # TODO: make n-episode a usable parameter for collect
        total_steps, last_done = collect_result['n/st'] + total_steps, collect_result["done"]
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        if i % args.log_interval == 0:
            print("testing collection")
            # option.policy.set_eps(0.05)
            test_collector.reset()
            for j in range(args.test_trials):
                if args.max_steps > 0: result = test_collector.collect(n_episode=1, n_step=args.max_steps, new_param=True)
                else: result = test_collector.collect(n_episode=1, new_param=True)
                print(result["n/st"])
                test_perf.append(result["rews"].mean())
                suc.append(float(result["n/st"] != args.max_steps and result["true_done"] < 1))
            print("Iters: ", i, "Steps: ", total_steps)
            print(f'Test mean returns: {np.array(test_perf).mean()}', f"Success: {np.array(suc).mean()}")
            train_collector.reset_env() # because test collector and train collector share the same environment

            # result = test_collector.collect(n_episode=10)
            # print("Steps: ", total_steps)
            # print(f'Test mean returns: {result["rews"].mean()}')
            # train_collector.env.workers[0].env.state = train_collector.other_buffer[train_collector.at].obs
            # print(train_collector.other_buffer[train_collector.at].obs)
            # train_collector.data.obs = environment.env.state

        # if i == 5:
        #     error
        #     print(train_collector.buffer.act[:100], train_collector.buffer.obs[:100], train_collector.buffer.rew[:100], train_collector.buffer.done[:100])
            # error
        # train option.policy with a sampled batch data from buffer
        losses = option.policy.update(args.batch_size, train_collector.buffer)
        if i % args.log_interval == 0:
            print(losses)
            print("epsilon", epsilon)
        if args.epsilon_schedule > 0:
            epsilon = args.epsilon + (1-args.epsilon) * (np.exp(-1.0 * (i*args.num_steps)/args.epsilon_schedule))
            option.policy.set_eps(epsilon)
        if args.save_interval > 0 and (i+1) % args.save_interval == 0:
            option.save(args.save_dir)
            graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)

    if args.save_interval > 0:
        option.save(args.save_dir)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
    final_perf, final_suc = list(), list()
    final_perf, final_suc = collect_test_trials(args, test_collector, 0, 0, final_perf, final_suc)

    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False

# def trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph):
#     # initialize states. input state/stack goes to the policy, diff state keeps last two states
#     # full state, last full state goes into the ground truth forward model
#     # input_state = pytorch_model.wrap(environment_model.get_flattened_state(names=names), cuda=args.cuda)
#     # diff_state = pytorch_model.wrap(environment_model.get_flattened_state(names=[names[0]]), cuda=args.cuda)
#     full_state = environment_model.get_factored_state()
#     last_full_state = environment_model.get_factored_state()

#     # refactor this into the option section
#     if option.object_name == 'Stack':
#         stack = torch.zeros([4,84,84]).detach()
#         if args.cuda:
#             stack = stack.cuda()
#         stack = stack.roll(-1,0)
#         stack[-1] = pytorch_model.wrap(environment.render_frame(), cuda=args.cuda).detach()

#     # initialize information about ground truth rewards and dones for recording
#     true_rewards = deque(maxlen=1000)
#     true_dones = deque(maxlen=1000)
#     done_lengths = deque(maxlen=1000)
#     success = deque(maxlen=100)
#     done_count = 0

#     # set some base values
#     last_done = True
#     done = True
#     ep = args.epsilon

#     # policy loss is for logging purposes only
#     policy_loss = PolicyLoss(None, None, None, None, None, None)
#     # if args.warm_up > 0:
#     #     option.set_behavior_epsilon(1)

#     all_sa = load_from_pickle("data/all_sa.pkl")
#     # an iteration corresponds to a learning step
#     for i in range(args.num_iters):
#         return_update = rollouts.at
#         for j in range(args.num_steps):
#             start = time.time()

#             # resamples parameter and mask if last_done is true
#             param, mask = option.get_param(full_state, last_done)

#             # store action, currently not storing action probabilities, might be necessary
#             # TODO: include the diffs for each level of the option internally
#             # use the correct termination for updating the diff (internaly in option?)
#             action_chain, rl_outputs = option.sample_action_chain(full_state, param) # uncomment this line
#             # action_chain, rl_outputs = option.sample_action_chain(full_state, param, learning_algorithm.policy) # REMOVE THIS LINE LATER
#             # print(full_state)
#             # # REMOVE forced actions
#             # action_chain[-1] = torch.tensor(forced_actions[j + i * args.num_steps]).cuda()
#             # # REMOVE ABOVE
#             # print(action_chain)
#             # print(option.policy)

#             # step the environment
#             # print(action_chain)
#             option.step(full_state, action_chain)
#             # print(action_chain[0].cpu().numpy())
#             frame, factored, true_done = environment.step(action_chain[0].cpu().numpy())
#             # Record information about next state
#             next_full_state = environment_model.get_factored_state()
#             true_rewards.append(environment.reward), true_dones.append(true_done)

#             # done count measures number of terminations
#             option.step_timer(done)
#             # progress the option and then calculate if terminated
#             dones, rewards = option.terminate_reward(next_full_state, param, action_chain)
#             done, last_done = dones[-1], 1 if len(dones) < 2 else dones[-2] # use the last option's done if it exists

#             # full_state['State'] = all_sa[i*args.num_steps+j][0]
#             # full_state['Action'] = all_sa[i*args.num_steps+j][1]
#             # full_state['Reward'] = all_sa[i*args.num_steps+j][2]
#             # full_state['Done'] = all_sa[i*args.num_steps+j][3]
#             # dones[0][0] = all_sa[i*args.num_steps+j][3]
#             # done = all_sa[i*args.num_steps+j][3]
#             # rewards[0][0] = all_sa[i*args.num_steps+j][2]
#             # action_chain[0][0] = all_sa[i*args.num_steps+j][1]

#             # refactor this into the option part
#             option.record_state(full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
#             if args.train and i >= args.warm_up: # it may be necessary for the learning algorithm to in parallel store state information
#                 learning_algorithm.record_state(i, full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
#             # state = pytorch_model.unwrap(option.rollouts.values.state[option.rollouts.at-1].reshape(20,20,3))
#             # state = pytorch_model.unwrap(full_state["Frame"])
#             # state[0,0,1] = 1
#             # cv2.imshow('Example - Show image in window',state * 255)
#             # cv2.imshow('Example - Show image in window',frame* 255)
#             # cv2.waitKey(100)
#             # print("showing", i)
#             # print(i*args.num_iters + j, option.get_state(full_state), dones, rewards, action_chain)
#             # print("true", option.rollouts.get_values("action")[-5:], option.rollouts.get_values("state")[-5:], option.rollouts.get_values("next_state")[-5:], "obj", option.rollouts.get_values("object_state")[-5:], option.rollouts.get_values("next_object_state")[-5:], "param", option.rollouts.get_values("param")[-5:], option.rollouts.get_values("done")[-5:], option.rollouts.get_values("reward")[-5:])
#             # print("learning", learning_algorithm.rollouts.get_values("action")[-5:], learning_algorithm.rollouts.get_values("state")[-5:], learning_algorithm.rollouts.get_values("next_state")[-5:], "obj", learning_algorithm.rollouts.get_values("object_state")[-5:], learning_algorithm.rollouts.get_values("next_object_state")[-5:], "param", learning_algorithm.rollouts.get_values("param")[-5:], learning_algorithm.rollouts.get_values("done")[-5:], learning_algorithm.rollouts.get_values("reward")[-5:])
#             done_count += 1
#             if done:
#                 done_lengths.append(done_count)
#                 # print("r1", option.rollouts.get_values("reward")[-done_count:])
#                 # print(option.rollouts.get_values("done")[-done_count:])
#                 # print(option.rollouts.get_values("state")[-dx`xone_count:])
#                 # print(option.rollouts.get_values("param")[-done_count:])
#                 done_count = 0
#                 # print(dones, j, environment.done)
#                 if j == args.num_steps - 1:
#                     success.append(0)
#                 else:
#                     success.append(1)
#                     if args.true_environment: # end on done
#                         break
            
#             full_state = next_full_state
#             # if done:
#             #     print("r2", learning_algorithm.rollouts.get_values("reward")[-done_count:])
#             #     print(learning_algorithm.rollouts.get_values("done")[-done_count:])
#             #     error
#         if args.return_form != "none":
#             input_state = option.get_state(full_state, form=1, inp=0)
#             next_value = option.forward(input_state.unsqueeze(0), param.unsqueeze(0)).values
#             option.compute_return(args.gamma, return_update, args.num_steps, next_value, return_max = 128, return_form=args.return_form)
#         # if option.rollouts.get_values("done")[-args.num_steps:].sum() > 0:
#         #     print("located reward: ", option.timer)
#         #     print(option.rollouts.get_values("reward"))
#         #     print(learning_algorithm.rollouts.get_values("reward"))

#         if i >= args.warm_up:
#             # batch = option.rollouts.get_values("state").reshape(-1,20,20,3)
#             # for i in range(100):
#             #     print(i)
#             #     cv2.imshow("buffer states", pytorch_model.unwrap(batch[i]))
#             #     cv2.waitKey(200)
#             if i == args.warm_up:
#                 random_episode_reward = option.rollouts.get_values("reward").sum()/option.rollouts.get_values("done").sum()

#                 args.epsilon = ep
#                 print("warm updating", args.epsilon)
#                 option.policy.set_mean(option.rollouts)
#                 if args.train: # warm updates trains for num steps with the warm-up data
#                     for _ in range(args.warm_updates):

#                         policy_loss = learning_algorithm.step(option.rollouts)

#             if args.epsilon_schedule > 0:
#                 args.epsilon = ep + (1-ep) * np.exp(-1*(i - args.warm_up)/args.epsilon_schedule)
#                 # print("epsilon", args.epsilon)
#             option.set_behavior_epsilon(args.epsilon)
#             logger.log(i, args.epsilon, args.log_interval, args.num_steps, policy_loss, option.rollouts, true_rewards, true_dones, success, continuous_actions = args.continuous)
#             # print("log", time.time() - start)
#             # start = time.time()
#             if args.train:
#                 # if i == args.warm_up:
#                 #     print(learning_algorithm.policy.critic.l1.weight.data)
#                 #     print(learning_algorithm.policy.critic.QFunction.l1.weight.data)
#                 #     print(learning_algorithm.policy.critic.QFunction.l2.weight.data)
#                 policy_loss = learning_algorithm.step(option.rollouts)
#                 # error
#                 # print("train", time.time() - start)
#                 # start = time.time()
#             if args.save_interval > 0 and (i+1) % args.save_interval == 0:
#                 option.save(args.save_dir)
#                 graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
#         # elif args.warm_up > 0: # if warming up, take random actions
#             # args.epsilon = 1.0
#             # option.set_behavior_epsilon(args.epsilon)
#         # if i == args.warm_up + 50:
#         #     print(option.policy.critic.QFunction.l2.weight.data)
#         #     error
#         # NO: end episode after num_steps if true_environment, true environment handles its own resets
#         # if args.true_environment:
#         #     environment.reset()

#     # computes success based on the last 2000 time steps
#     trained = False
#     final_episode_reward = option.rollouts.get_values("reward")[max(option.rollouts.filled - 2000, 0):].sum() / option.rollouts.get_values("done")[max(option.rollouts.filled - 2000, 0):].sum()
#     if final_episode_reward - random_episode_reward > args.train_reward_significance:
#         trained = True
#     return done_lengths, trained