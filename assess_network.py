import numpy as np
from file_management import ObjDict, save_to_pickle, load_from_pickle
from Networks.network import PairNetwork, pytorch_model, BasicMLPNetwork
from Environments.SelfBreakout.breakout_screen import Screen, AnglePolicy
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import copy
from test_network import TargetingPolicy, construct_state
from visualizer import draw_target

if __name__ == "__main__":
    args = ObjDict()
    args.num_iters = 100
    args.num_steps = 1000
    args.num_rows = 2 # row counter mode
    args.num_samples = 20 # row counter mode
    args.num_outputs = 4 # row counter mode
    args.num_columns = 10
    args.negative_mode = "" 
    args.num_actions = 4
    args.no_breakout = True
    args.hit_reset = 15
    args.filename = "ball_test_policy_small"

    screen = Screen(num_rows = args.num_rows, num_columns = args.num_columns, max_block_height=4, random_exist=args.num_samples, negative_mode = args.negative_mode, no_breakout = args.no_breakout, hit_reset=args.hit_reset)
    net = torch.load("data/breakout/network_test/net_"+ args.filename + ".pt")
    policy = TargetingPolicy(screen, net, construct_state, args.num_samples)
    for j in range(args.num_iters):
        for i in range(args.num_steps):
            state = screen.get_state()
            action = policy.forward(state)
            screen.step(action)
            
            target = policy.target
            im = screen.render()
            im = np.stack([im.copy() for _ in range(3)], axis=-1)
            draw_target(im, target)
            cv2.imshow('frame',im)
            if cv2.waitKey(30) & 0xFF == ord(' ') & 0xFF == ord('c'):
                continue
        print("Iters: ", j, policy.hit / (policy.hit + policy.miss), policy.hit, (policy.hit + policy.miss))
