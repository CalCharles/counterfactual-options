import os, collections, time, copy
import gc, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from collections import deque
import imageio as imio

from Environments.environment_specification import ProxyEnvironment
from ReinforcementLearning.Policy.policy import pytorch_model
from file_management import save_to_pickle

def testRL(args, test_collector, environment, environment_model, option, names, graph):
    test_perf, suc = deque(maxlen=200), deque(maxlen=200)
    option.zero_epsilon()
    total_steps = 0
    for i in range(args.num_iters):
        print("testing collection")
        # option.policy.set_eps(0.05)
        test_collector.reset()
        hit_miss = list()
        for j in range(args.test_trials):
            if args.max_steps > 0: result = test_collector.collect(n_episode=1, n_term=1, n_step=args.max_steps, visualize_param=args.visualize_param)
            print(result["n/st"])
            total_steps += result["n/st"]
            hit_miss.append(result['n/h'])
            test_perf.append(result["rews"].mean())
            suc.append(float(result["terminate"]))
        print("Iters: ", i, "Steps: ", total_steps)
        mean_hit = sum(hit_miss)/ max(1, len(hit_miss))
        print(f'Test mean returns: {np.array(test_perf).mean()}', f"Success: {np.array(suc).mean()}", f"Hit Miss: {mean_hit}")
        test_collector.reset_env() # because test collector and train collector share the same environment


# def testRL(args, rollouts, logger, environment, environment_model, option, names, graph):
#     full_state = environment_model.get_factored_state()
#     last_full_state = environment_model.get_factored_state()
#     if option.object_name == 'Raw':
#         stack = torch.zeros([4,84,84])
#         if args.cuda:
#             stack = stack.cuda()
#         stack = stack.roll(-1,0)
#         stack[-1] = pytorch_model.wrap(environment.render_frame(), cuda=args.cuda)
#     true_rewards = deque(maxlen=1000)
#     true_dones = deque(maxlen=1000)
#     last_done = True
#     done = True
#     success = [1]
#     for i in range(args.num_iters):
#         param, mask = option.get_param(full_state, last_done)
#         action_chain, rl_outputs = option.sample_action_chain(full_state, param)
#         option.step(full_state, action_chain)
#         frame, factored, true_done = environment.step(action_chain[0].cpu().numpy())
#         # Record information about next state
#         next_full_state = environment_model.get_factored_state()
#         true_rewards.append(environment.reward), true_dones.append(true_done)
#         dones, rewards = option.terminate_reward(next_full_state, param, action_chain)
#         done, last_done = dones[-1], dones[-2]
#         option.step_timer(done)

#         # refactor this into the option part
#         option.record_state(full_state, next_full_state, action_chain, rl_outputs, param, rewards, dones)
#         print(param, option.timer, option.time_cutoff, option.get_state(full_state), action_chain[-1], last_done)
#         # print(option.rollouts.get_values("done")[-1], option.rollouts.get_values("next_state")[-1], option.rollouts.get_values("param")[-1])
        
#         # # visualization, but only paddle
#         # frame = copy.deepcopy(environment.frame)
#         # frame1 = copy.deepcopy(environment.frame)
#         # frame2 = copy.deepcopy(environment.frame)
#         # paddle = environment.paddle
#         # frame1[71:71+paddle.height, int(param[1] + 1.5):int(param[1] + 1.5)+paddle.width] = 255
#         # frame = np.stack([frame, frame1, frame2], axis = 2)
#         # # imio.imsave(os.path.join("data", "target_frames", "state" + str(i) + ".png"), frame)
#         # cv2.imshow("window", frame)
#         # cv2.waitKey(50)
        
#         full_state = next_full_state
#         if args.return_form != "none":
#             input_state = option.get_state(full_state, form=1, inp=0)
#             next_value = option.forward(input_state.unsqueeze(0), param.unsqueeze(0)).values
#             option.compute_return(args.gamma, option.rollouts.at-1, 1, next_value, return_max = 128, return_form=args.return_form)
#         logger.log(i, option.behavior_policy.epsilon, args.log_interval, 1, None, option.rollouts, true_rewards, true_dones, success)

