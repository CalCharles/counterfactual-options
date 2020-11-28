import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument('--algo', default='a2c',
    #                     help='algorithm to use: a2c, ppo, evo')

    # environment hyperparameters
    parser.add_argument('--true-environment', action='store_true', default=False,
                        help='triggers the baseline methods')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='number of frames to skip (default: 1 (no skipping))')
    parser.add_argument('--object', default='',
                        help='name of the object whose options are being investigated')
    parser.add_argument('--target', default='',
                        help='name of the object filtering for')
    # # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (definedfault: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    # cost function hyperparameters
    parser.add_argument('--return-form', default='value',
                        help='determines what return equation to use. true is true returns, gae is gae (not implemented), value uses the value function, none avoids return computation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=1e-2,
                        help='entropy loss term coefficient (default: 1e-7)')
    parser.add_argument('--high-entropy', type=float, default=0,
                        help='high entropy (for low frequency) term coefficient (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    # model hyperparameters
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of layers for network. When using basis functions, defines independence relations (see ReinforcementLearning.basis_models.py)')
    parser.add_argument('--factor', type=int, default=4,
                        help='decides width of the network')
    parser.add_argument('--optim', default="RMSprop",
                        help='optimizer to use: Adam, RMSprop, Evol')
    parser.add_argument('--activation', default="relu",
                        help='activation function for hidden layers: relu, sin, tanh, sigmoid')
    parser.add_argument('--init-form', default="uni",
                    help='initialization to use: uni, xnorm, xuni, eye')
    parser.add_argument('--last-param', action='store_true', default=False,
                    help='use the parameter at the last layer, otherwise, does it at the first layer')
    parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalizes the inputs using 84')
    parser.add_argument('--predict-dynamics', action='store_true', default=False,
                    help='predict the dynamics instead of the next state')
    parser.add_argument('--action-shift', action='store_true', default=False,
                    help='shift the actions back one time step so the action is applied at the last time step')
    # Meta-Hyperparameters
    parser.add_argument('--policy-type', default="basic",
                        help='choose the model form for the policy, which is defined in Policy.policy')
    parser.add_argument('--terminal-type', default="param",
                        help='choose the way the terminal condition is defined, in Option.Termination.termination')
    parser.add_argument('--reward-type', default="bin",
                        help='choose the way the reward is defined, in Option.Reward.reward')
    parser.add_argument('--option-type', default="discrete",
                        help='choose the way the option is defined, in Option.option')
    parser.add_argument('--behavior-type', default="prob",
                        help='choose the way the behavior policy is defined, in ReinforcementLearning.behavior_policy')
    parser.add_argument('--learning-type', default='ppo',
                        help='defines the algorithm used for learning')
    # Behavior policy parameters
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='When the policy outputs a continuous distribution')
    parser.add_argument('--epsilon', type=float, default=0.1,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--epsilon-schedule', type=int, default=-1,
                    help='halves the percentage of epsilon after this many time steps')
    # termination set parameters
    parser.add_argument('--epsilon-close', type=float, default=0.1,
                    help='minimum distance for states to be considered the same')

    # Learning settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--lag-num', type=int, default=2,
                        help='lag between states executed and those used for learning, to delay for reward computation TODO: 1 is the minimum for reasons... (default: 1)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of reward checks before update (default: 1)')
    parser.add_argument('--grad-epoch', type=int, default=5,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--reward-check', type=int, default=5,
                        help='steps between a check for reward, (default 1)')
    parser.add_argument('--num-iters', type=int, default=int(2e5),
                        help='number of iterations for training (default: 2e5)')
    parser.add_argument('--interaction-iters', type=int, default=0,
                        help='number of iterations for training the interaction network with true values (default: 0)')
    parser.add_argument('--pretrain-iters', type=int, default=int(2e4),
                        help='number of iterations for training (default: 2e4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--Q-critic', action='store_true', default=False,
                        help='use the q function to compute the state value')
    parser.add_argument('--warm-up', type=int, default=0,
                        help='warm up updates to fill buffer (default: 0)')
    parser.add_argument('--ratio', type=float, default=0.9,
                    help='ratio of training samples to testing ones')

    # PPO settings
    parser.add_argument('--clip-param', type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')
    # goal search replay settings
    parser.add_argument('--search-rate', type=float, default=0.7,
                    help='rate at which a high reward parameter is chosen (default: 0.7)')

    # Replay buffer settings
    parser.add_argument('--match-option', action='store_true', default=False,
                        help='use data only from the option currently learning (default False')
    parser.add_argument('--buffer-steps', type=int, default=32,
                        help='number of buffered steps in the record buffer (default: 32)')
    parser.add_argument('--buffer-clip', type=int, default=20,
                        help='backwards return computation (strong effect on runtime')
    parser.add_argument('--weighting-lambda', type=float, default=1e-3,
                        help='lambda for the sample weighting in prioritized replay (default = 1e-2)')
    parser.add_argument('--prioritized-replay', default="",
                        help='different prioritized replay schemes, (TD (Q TD error), return, recent, ""), default: ""')
    # Option Chain Parameters
    parser.add_argument('--base-node', default="Action",
                        help='The name of the lowest node in the option chain (generally should be Action)')
    parser.add_argument('--use-both', type=int, default=1,
                        help='enum for which part to use as parameter (0: state, 1: state difference, 2: both state and state difference)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='number of samples to take for all_state_next')
    # termination condition parameters
    parser.add_argument('--min-use', type=int, default=5,
                    help='minimum number of seen states to use as a parameter')

    # logging settings
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=-1,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='data/new_net/',
                        help='directory to save data when adding edges')
    parser.add_argument('--save-graph', default='',
                        help='directory to save graph data. If empty, does not save the graph')
    parser.add_argument('--save-recycle', type=int, default=-1,
                        help='only saves the last n timesteps (-1 if not used)')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded (when adding edges, where data was recorded to compute min/max)')
    parser.add_argument('--graph-dir', default='./data/optgraph/',
                        help='directory to load graph')
    parser.add_argument('--dataset-dir', default='./data/',
                        help='directory to save/load dataset')
    parser.add_argument('--unique-id', default="0",
                        help='a differentiator for the save path for this network')
    parser.add_argument('--save-past', type=int, default=-1,
                    help='save past, saves a new net at the interval, -1 disables, must be a multiple of save-interval (default: -1)')
    parser.add_argument('--display-focus', action ='store_true', default=False,
                        help='shows an image with the focus at each timestep like a video')
    parser.add_argument('--save-raw', action ='store_true', default=False,
                        help='shows an image with the focus at each timestep like a video')
    parser.add_argument('--single-save-dir', default="",
                        help='saves all images to a single directory with name all')
    # environmental variables
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number to use (default: 0)')
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env', default='SelfBreakout',
                        help='environment to train on (default: SelfBreakout)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
    parser.add_argument('--set-time-cutoff', action ='store_true', default=False,
                        help='runs the algorithm without time cutoff to set it')

    # load variables
    parser.add_argument('--load-weights', action ='store_true', default=False,
                        help='load the options for the existing network')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_edge(edge):
    head = edge.split("->")[0]
    tail = edge.split("->")[1]
    head = head.split(",")
    return head, tail[0]
