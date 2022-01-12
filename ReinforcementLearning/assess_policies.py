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

def set_environment(test_collector, option, environment, environment_model, stored_data):
    test_collector.env.workers[0].env = environment
    test_collector.environment_model = environment_model
    option.sampler.current_environment_model = environment_model
    if stored_data is not None:
        test_collector.data = copy.deepcopy(stored_data)

def array_full(state):
    for k in state['factored_state'].keys():
        state['factored_state'][k] = np.array(state['factored_state'][k])
    return state['factored_state']

def run_policy(policy, environment, environment_model, entity_selector, act, option, initial_blocks):
    policy.reset_screen(environment)
    state = environment.get_state()
    inter_state = entity_selector(array_full(state))
    action = 0
    # print(state["factored_state"]["Ball"])
    while not np.any(option.dataset_model.hypothesize(inter_state)[0]):
        # frame = self.render_frame()
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(10) & 0xFF == ord(' ') & 0xFF == ord('c'):
        #     continue
        action = policy.act(environment, angle=act)
        if action == -1: # signal to quit
            break
        state, reward, done, info = environment.step(action)
        inter_state = entity_selector(array_full(state))
        # print(state["factored_state"]["Ball"])
    state, reward, done, info = environment.step(action)
    final_blocks = np.array([state["factored_state"]["Block" + str(i)][4] for i in range(20)])
    val = np.nonzero(initial_blocks - final_blocks)
    print("policy hit", val, initial_blocks, final_blocks)
    return val

def assess_policies(args, test_collector, environment, environment_model, option, policy, names, graph):
    # TODO: currently relies on being able to copy an environment to reset it
    # TODO: currently only compares an option with a policy
    test_perf, suc = deque(maxlen=200), deque(maxlen=200)
    option.zero_epsilon()
    total_steps = 0
    entity_selector = environment_model.create_entity_selector(["Ball", "Block"]) # TODO: hardcoded hypothesis model information
    match_num, total_num = 0, 0
    stored_data = None
    for i in range(args.num_iters):
        print("testing collection")
        # option.policy.set_eps(0.05)
        # test_collector.reset()
        hit_count = list()
        miss_count = list()
        base_environment_model = copy.deepcopy(environment_model)
        rc = np.random.randint(option.action_map.num_actions)

        stepped_model = copy.deepcopy(environment_model)
        state, _, _, _ = stepped_model.step(0)
        initial_blocks = np.array([state["factored_state"]["Block" + str(i)][4] for i in range(20)]) # give one step to restore any base blocks
        for j in range(option.action_map.num_actions):
            # TODO: only handles discrete actions
            next_environment_model = copy.deepcopy(base_environment_model)
            next_environment = next_environment_model.environment

            set_environment(test_collector, option, next_environment, next_environment_model, stored_data)
            # print(stored_data.full_state['factored_state']['Ball'] if stored_data is not None else None, test_collector.data.full_state['factored_state']['Ball'])
            state = next_environment.get_state()
            if args.max_steps > 0: result = test_collector.collect(n_episode=1, n_term=1, n_step=args.max_steps, visualize_param=args.visualize_param, force=j)
            state = next_environment.get_state()
            final_blocks = np.array([state["factored_state"]["Block" + str(i)][4] for i in range(20)])
            option_val = np.nonzero(initial_blocks - final_blocks)
            print("option hit", option_val, initial_blocks, final_blocks, state["factored_state"]["Block" + str(option_val[0][0])])

            next_environment_model = copy.deepcopy(base_environment_model)
            next_environment = next_environment_model.environment
            policy_val = run_policy(policy, next_environment, next_environment_model, entity_selector, j, option, initial_blocks)

            matched = False
            for v in option_val:
                for p in policy_val:
                    if len(p) == len(v) == 0:
                        matched = True
                        break
                    if v[0] == p[0]:
                        matched = True
                        break
                if matched:
                    break
            match_num += int(matched)
            total_num += 1

            if rc == j: 
                next_base_environment_model = copy.deepcopy(next_environment_model)
                next_stored_data = copy.deepcopy(test_collector.data)
                print("recorded", next_base_environment_model.get_state()['factored_state']['Ball'])

            print("iter", j, "num steps", result["n/st"])
            total_steps += result["n/st"]
            hit_count.append(result['n/h'])
            miss_count.append(result['n/m'])
            test_perf.append(result["rews"].mean())
            suc.append(float(result["terminate"]))
            print("matching", match_num / total_num)
        stored_data = next_stored_data
        environment_model = next_base_environment_model
        environment = environment_model.environment
        print("Iters: ", i, "Steps: ", total_steps)
        mean_hit = sum(hit_count)/ max(1, sum(miss_count) + sum(hit_count))
        print(f'Test mean returns: {np.array(test_perf).mean()}', f"Success: {np.array(suc).mean()}", f"Hit Miss: {mean_hit}")
        