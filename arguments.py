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
    # # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    # cost function hyperparameters
    parser.add_argument('--return-form', default='true',
                        help='determines what return equation to use. true is true returns, gae is gae (not implemented), value uses the value function')
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
    parser.add_argument('--optim', default="Adam",
                        help='optimizer to use: Adam, RMSprop, Evol')
    parser.add_argument('--activation', default="relu",
                        help='activation function for hidden layers: relu, sin, tanh, sigmoid')
    parser.add_argument('--init-form', default="uni",
                    help='initialization to use: uni, xnorm, xuni, eye')
    parser.add_argument('--model-form', default="",
                        help='choose the model form, which is defined in Models.models')
    # Behavior policy parameters
    parser.add_argument('--greedy-epsilon', type=float, default=0.1,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--min-greedy-epsilon', type=float, default=0.1,
                    help='minimum percentage of random actions in epsilon greedy (if decaying)')
    parser.add_argument('--greedy-epsilon-decay', type=float, default=-1,
                    help='greedy epsilon decays by half every n updates (-1 is for no use)')
    parser.add_argument('--behavior-policy', default='',
                        help='defines the behavior policy, as defined in BehaviorPolicies.behavior_policies')

    # Learning settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--lag-num', type=int, default=2,
                        help='lag between states executed and those used for learning, to delay for reward computation TODO: 1 is the minimum for reasons... (default: 1)')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='number of reward checks before update (default: 1)')
    parser.add_argument('--num-grad-states', type=int, default=-1,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--reward-check', type=int, default=5,
                        help='steps between a check for reward, (default 1)')
    parser.add_argument('--num-iters', type=int, default=int(2e3),
                        help='number of iterations for training (default: 2e3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Replay buffer settings
    parser.add_argument('--match-option', action='store_true', default=False,
                        help='use data only from the option currently learning (default False')
    parser.add_argument('--buffer-steps', type=int, default=-1,
                        help='number of buffered steps in the record buffer, -1 implies it is not used (default: -1)')
    parser.add_argument('--buffer-clip', type=int, default=20,
                        help='backwards return computation (strong effect on runtime')
    parser.add_argument('--weighting-lambda', type=float, default=1e-2,
                        help='lambda for the sample weighting in prioritized replay (default = 1e-2)')
    parser.add_argument('--prioritized-replay', default="",
                        help='different prioritized replay schemes, (TD (Q TD error), return, recent, ""), default: ""')
    # logging settings
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='',
                        help='directory to save data when adding edges')
    parser.add_argument('--save-graph', default='graph',
                        help='directory to save graph data. Use "graph" to let the graph specify target dir, empty does not train')
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
    parser.add_argument('--save-models', action ='store_true', default=False,
                        help='Saves environment and models to option chain directory if true')
    parser.add_argument('--display-focus', action ='store_true', default=False,
                        help='shows an image with the focus at each timestep like a video')
    parser.add_argument('--single-save-dir', default="",
                        help='saves all images to a single directory with name all')
    # Option Chain Parameters
    parser.add_argument('--base-node', default="Action",
                        help='The name of the lowest node in the option chain (generally should be Action)')
    # environmental variables
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number to use (default: 0)')
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env', default='SelfBreakout',
                        help='environment to train on (default: SelfBreakout)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
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
