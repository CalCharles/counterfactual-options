import argparse

import torch

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument('--algo', default='a2c',
    #                     help='algorithm to use: a2c, ppo, evo')
    parser.add_argument('--seed', type=int, default=4,
                        help='random seed (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='no cuda when cuda is avaliable')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded (when adding edges, where data was recorded to compute min/max)')
    parser.add_argument('--save-interval', type=int, default=-1,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n episodes (default: 10)')
    parser.add_argument('--final-instanced', action='store_true', default=False,
                    help='if the final layer is instanced (will generate a pairnet)')

    parser.add_argument('--lookahead', type=int, default=1,
                        help='optimization steps to look ahead (default: 1)')
    parser.add_argument('--input-norm', action='store_true', default=False,
                    help='normalize the inputs by the sample mean variance from the buffer')
    parser.add_argument('--printout', action='store_true', default=False,
                    help='debugging printouts')
    parser.add_argument('--epsilon', type=float, default=0,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--epsilon-schedule', type=float, default=-1,
                    help='uses exp (-steps/epsilon-schedule) to compute epsilon at a given step, -1 for no schedule')
    parser.add_argument('--grad-epoch', type=int, default=5,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='number of episodes to run (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, not used if actor and critic learning rate used for algo (default: 1e-6)')
    parser.add_argument('--actor-lr', type=float, default=-1,
                        help='actor learning rate (default: -1 (replace wih lr))')
    parser.add_argument('--critic-lr', type=float, default=-1,
                        help='critic learning rate (default: -1 (replace with lr))' +
                        'overloaded to also be the interaction model lr')
    parser.add_argument('--max-critic', type=float, default=-1,
                    help='bounds the critic values between -max-critic and max-critic (-1 is not used)')
    parser.add_argument('--bound-output', type=float, default=0,
                    help='bounds the output between -bound-output and bound-output (0 is not used)')
    parser.add_argument('--policy-type', default="basic",
                        help='choose the model form for the policy, which is defined in Policy.policy, overloaded to also specify the kind of network when training the hypothesis model')
    parser.add_argument('--learning-type', default="ddpg",
                        help='choose the learning algorithm, default ddpg')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number to use (default: 0)')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256],
                        help='sizes of the internal hidden layers')
    parser.add_argument('--no-keep-instanced', action = 'store_true', default=False,
                    help='if true, then use a reduced state for non-top of HAC')
    parser.add_argument('--init-form', default="none",
                        help='choose the initialization for the weights')    
    parser.add_argument('--use-layer-norm', action='store_true', default=False,
                    help='uses layer norm in the network')
    parser.add_argument('--activation', default="relu",
                        help='choose the activation function (TODO: not used at the moment)')    
    parser.add_argument('--tau', type=float, default=0.005,
                        help='parameter for target network updates (default: 0.95)')
    parser.add_argument('--buffer-len', type=int, default=100000,
                        help='replay buffer length')
    parser.add_argument('--prioritized-replay', type=float, nargs='*', default=[.6, .4],
                        help='alpha and beta values for prioritized replay')

    parser.add_argument('--epsilon-close', type=float, nargs='*', default=[1e-2],
                    help='how close to the target state is considered a goal')
    parser.add_argument('--lamda', type=float, default=0.3,
                    help='ratio to use subgoal testing in HAC')
    parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
    parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch sizes')
    parser.add_argument('--k-level', type=int, default=3,
                    help='number of HAC layers')
    parser.add_argument('--H', type=int, default=20,
                    help='HAC time horizon')
    parser.add_argument('--env', default="SelfBreakout",
                        help='the environment')
    parser.add_argument('--block-shape', type=int, nargs=5, default=(5, 20, 4, 0, 0),
                        help='shape of the blocks, number of blocks high, number of blocks wide, max block height, no_breakout(flag), max number of hits (default: (5,20,4,0,0))')
    parser.add_argument('--drop-stopping', action = 'store_true', default=False,
                    help='returns done when the ball is dropped')
    parser.add_argument('--target-mode', action='store_true', default=False,
                    help='in breakout, induces a domain where there is a single block to target')
    parser.add_argument('--breakout-variant', default='',
                        help='name of a specialized variant of breakout')

    # # HAC parameters:
    # k_level = 2                 # num of levels in hierarchy
    # H = 20                      # time horizon to achieve subgoal
    # lamda = 0.3                 # subgoal testing parameter
    
    # # DDPG parameters:
    # gamma = 0.95                # discount factor for future rewards
    # n_iter = 100                # update policy n_iter times in one DDPG update
    # batch_size = 100            # num of transitions sampled from replay buffer
    # lr = 0.001


    args = parser.parse_args()
    args.epsilon_close = np.array(args.epsilon_close) if len(args.epsilon_close) > 1 else args.epsilon_close[0]
    args.discount_factor = args.gamma # TianShou Support
    # args.max_critic = args.H if args.max_critic == -1 else args.max_critic
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.critic_lr = args.lr if args.critic_lr < 0 else args.critic_lr
    args.actor_lr = args.lr if args.actor_lr < 0 else args.actor_lr
    args.keep_instanced = not args.no_keep_instanced

    return args