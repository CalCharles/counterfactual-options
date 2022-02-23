import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument('--algo', default='a2c',
    #                     help='algorithm to use: a2c, ppo, evo')

    # environment hyperparameters
    parser.add_argument('--true-environment', action='store_true', default=False,
                        help='triggers the baseline methods')
    parser.add_argument('--primitive-actions', action='store_true', default=False,
                        help='take primitive actions instead of next level options')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='number of frames to skip (default: 1 (no skipping))')
    parser.add_argument('--num-obstacles', type=int, default=0,
                        help='number of obstacles for pushing domain (default: 0)')
    parser.add_argument('--object', default='',
                        help='name of the object whose options are being investigated')
    parser.add_argument('--target', default='',
                        help='name of the object filtering for')
    parser.add_argument('--temporal-extend', type=int, default=-1,
                        help='take temporally extended actions, max number of steps to extend before resampling (default: -1 (no extension))')
    parser.add_argument('--env-timelimit', type=int, default=-1,
                        help='amount of time for the environment to expire (default: -1 (no extension))')
    parser.add_argument('--env-reset', action='store_true', default=False,
                    help='collector will call reset whenever a "done" occurs unless this is true')
    parser.add_argument('--target-mode', action='store_true', default=False,
                    help='in breakout, induces a domain where there is a single block to target')
    parser.add_argument('--block-shape', type=int, nargs=5, default=(5, 20, 4, 0, 0),
                        help='shape of the blocks, number of blocks high, number of blocks wide, max block height, no_breakout(flag), max number of hits (default: (5,20,4,0,0))')
    parser.add_argument('--breakout-variant', default='',
                        help='name of a specialized variant of breakout')
    parser.add_argument('--joint-mode', action='store_true', default=False,
                    help='in robosuite, changes the action space to be joint level control')
    parser.add_argument('--planar-mode', action='store_true', default=False,
                    help='in robosuite, changes the action space to be joint level control')
    parser.add_argument('--hard-obstacles', action='store_true', default=False,
                    help='in robosuite, changes the obstacles from soft negative reward regions to physical blocks ')    
    # # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, not used if actor and critic learning rate used for algo (default: 1e-6)')
    parser.add_argument('--actor-lr', type=float, default=-1,
                        help='actor learning rate (default: -1 (replace wih lr))')
    parser.add_argument('--critic-lr', type=float, default=-1,
                        help='critic learning rate (default: -1 (replace with lr))' +
                        'overloaded to also be the interaction model lr')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='parameter for target network updates (default: 0.95)')
    parser.add_argument('--lookahead', type=int, default=1,
                        help='optimization steps to look ahead (default: 1)')
    # SAC hyperparameters 
    parser.add_argument('--sac-alpha', type=float, default=0.2,
                        help='entropy constant for SAC (default: 0.2), use -1 for adaptive SAC')
    parser.add_argument('--not-deterministic-eval', action='store_true', default=False,
                        help='if true, deterministic evaluation for SAC is false')
    # cost function hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)') 
    # Iterated Supervised Learning parameters
    parser.add_argument('--label-smoothing', type=float, default=0.00,
                        help='smooths the one-hot labels for behavior cloning(default: 0.00)') 

    # rainbow hyperparameters
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='number of atoms to be used for C51 in Rainbow (default: 51)')
    parser.add_argument('--is-dueling', action='store_true', default=False,
                        help='using dueling DQN')
    # model hyperparameters
    parser.add_argument('--true-actions', action='store_true', default=False,
                        help='short circuits option framework to just take true actions')
    parser.add_argument('--discretize-actions', action='store_true', default=False,
                        help='converts a continuous action space to a discrete one (TODO: currently requires relative-action)')
    parser.add_argument('--observation-setting', type=int, nargs='+', default=(0,0,0,0,0,0,0,0,0),
                        help='sets the components of the input state to the policy (default: (1,0,0,1,0,0,0,0,0))' +
                        'components in this order: interaction_state, target state, full flattened state' +
                        'param, target state relative to param, relative interaction state, action state, diff,' +
                        'used in interation model to indicate relative state if position 0 is 1')
    parser.add_argument('--param-contained', action='store_true', default=False,
                        help='shifts the parameter to the front of the input, for pairnet implementations'+
                        'OVERLOADED to also move the target first in distributional models')
    parser.add_argument('--target-contained', action='store_true', default=False,
                        help='has the target at the start of the representation, for pairnet implementations where the target is NOT instanced')
    parser.add_argument('--interleave', action='store_true', default=False,
                        help='interweaves the interaction state and relative state for pairnet implementations')
    parser.add_argument('--keep-target', action='store_true', default=False,
                        help='also appends the target state to param_relative state')

    parser.add_argument('--relative-action', type=float, default=-1,
                    help='the model computes actions relative to the current object position (-1 is not used)')
    parser.add_argument('--max-critic', type=float, default=-1,
                    help='bounds the critic values between -max-critic and max-critic (-1 is not used)')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256],
                        help='sizes of the internal hidden layers')
    parser.add_argument('--init-form', default="none",
                        help='choose the initialization for the weights')    
    parser.add_argument('--use-layer-norm', action='store_true', default=False,
                    help='uses layer norm in the network')
    parser.add_argument('--bound-output', action='store_true', default=False,
                    help='bounds the critic output to [-time_cutoff, 0]')
    parser.add_argument('--input-norm', action='store_true', default=False,
                    help='normalize the inputs by the sample mean variance from the buffer')
    parser.add_argument('--hardcode-norm', type=str, nargs='+', default=list(),
                        help='use hardcoded input normalization, of format "env name" "layer num" "scale" (default: empty list (not used))')
    parser.add_argument('--use-pair-gamma', action='store_true', default=False,
                    help='uses the two objects in obs instead of the full gamma')
    parser.add_argument('--activation', default="relu",
                        help='choose the activation function (TODO: not used at the moment)')    
    parser.add_argument('--reward-normalization', action='store_true', default=False,
                        help='have the policy normalize the reward function')
    # dynamics model learning parameters
    parser.add_argument('--predict-dynamics', action='store_true', default=False,
                    help='predict the dynamics instead of the next state')
    parser.add_argument('--action-shift', action='store_true', default=False,
                    help='shift the actions back one time step so the action is applied at the last time step')
    # Meta-Hyperparameters
    parser.add_argument('--policy-type', default="basic",
                        help='choose the model form for the policy, which is defined in Policy.policy, overloaded to also specify the kind of network when training the hypothesis model')
    parser.add_argument('--terminal-type', default="param",
                        help='choose the way the terminal condition is defined, in Option.Termination.termination')
    parser.add_argument('--reward-type', default="bin",
                        help='choose the way the reward is defined, in Option.Reward.reward')
    parser.add_argument('--option-type', default="model",
                        help='choose the way the option is defined, in Option.option')
    parser.add_argument('--behavior-type', default="ts",
                        help='choose the way the behavior policy is defined, in ReinforcementLearning.behavior_policy')
    parser.add_argument('--learning-type', default='',
                        help='defines the algorithm used for learning, includes ddpg, sac, dqn, her can be added as a prefix to any of these')
    parser.add_argument('--sampler-type', default='uni',
                        help='defines the function used to sample param targets')
    # Behavior policy parameters
    # parser.add_argument('--continuous', action='store_true', default=False,
    #                     help='When the policy outputs a continuous distribution')
    parser.add_argument('--epsilon', type=float, default=0,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--epsilon-schedule', type=float, default=-1,
                    help='uses exp (-steps/epsilon-schedule) to compute epsilon at a given step, -1 for no schedule')
    # termination set parameters
    parser.add_argument('--max-distance-epsilon', type=float, default=-1,
                    help='minimum distance for multi-instanced reward/termination (default: -1)'+
                        'OVERLOADED to the maximum proximity to be considered proximal')
    parser.add_argument('--param-norm', type=int, default=1,
                    help='p-norm p value to use for the norm (default 1)')
    parser.add_argument('--epsilon-close', type=float, default=0.1,
                    help='minimum distance for states to be considered the same (default 0.1)')
    parser.add_argument('--negative-epsilon-close', type=float, default=0.1,
                    help='minimum distance for negative reward regions (robosuite pushing)')
    parser.add_argument('--epsilon-min', type=float, default=0.1,
                    help='minimum distance to end for ring schedule')
    parser.add_argument('--epsilon-close-schedule', type=float, default=0.0,
                    help='minimum distance for states to be considered the same')
    parser.add_argument('--pretrain-active', type=int, default=-1,
                    help='pretrain the active model with the passive model for the given number of iterations (-1 not used)')
    parser.add_argument('--intrain-passive', action = 'store_true', default=False,
                    help='trains the passive model on its own bad states')
    parser.add_argument('--passive-weight-interaction-iters', type=int, default=-1,
                        help='if positive, the number of steps to use the passive error as a proxy for training the interaction model (default: -1 (not used))')
    parser.add_argument('--param-interaction', action = 'store_true', default=False,
                    help='Only terminates when the param and interaction co-occur')
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='number of steps before forcing the end of episode flag (default: -1 (not used))')
    parser.add_argument('--param-recycle', type=float, default=0.0,
                    help='probability of choosing the same param after termination')
    parser.add_argument('--norm-variance', type=float, default=5.0,
                    help='variance used for normalization')
    parser.add_argument('--base-variance', type=float, nargs='*', default=[1e-2],
                    help='add a small variance to limit overfitting to easy to predict parts, if a single value, uses the same for all dimensions, if multiple, must be the size of the vector')
    parser.add_argument('--multi-instanced', action = 'store_true', default=False,
                    help='if interaction trains to predict a vector')
    parser.add_argument('--instanced-additional', action = 'store_true', default=False,
                    help='if interaction uses additional instanced information (obstacles)')
    parser.add_argument('--only-termination', action = 'store_true', default=False,
                    help='only stores termination values, not temporal extension ones')
    # sampler parameters
    parser.add_argument('--sample-continuous', type=int, default=0,
                        help='use already existing values if 0, false if 1, true if 2')
    parser.add_argument('--no-combine-param-mask', action='store_true', default=False,
                        help='if true, do NOT multiply the param with the mask')
    parser.add_argument('--check-on-binary', action='store_true', default=False,
                    help='if true, uses the last bin (when there is a bin) for the input state')

    # done check parameters
    parser.add_argument('--use-termination', action = 'store_true', default=False,
                    help='returns done when the option terminates')
    parser.add_argument('--terminate-reset', action = 'store_true', default=False,
                    help='resets the environment when the option terminates')
    parser.add_argument('--terminate-cutoff', action = 'store_true', default=False,
                    help='prevents time cutoff when true done IF the envrionment does not send it')
    parser.add_argument('--no-truncate', action = 'store_true', default=False,
                    help='prevents truncation in all cases')
    parser.add_argument('--not-true-done-stopping', action = 'store_true', default=False,
                    help='if true, will NOT end episode when the environment ends the episode')

    # Learning settings
    parser.add_argument('--seed', type=int, default=4,
                        help='random seed (default: 4)')
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
    parser.add_argument('--buffer-len', type=int, default=int(1e6),
                        help='length of the replay buffer (default: 1e6)')
    parser.add_argument('--prioritized-replay', type=float, nargs='*', default=[],
                        help='alpha and beta values for prioritized replay')
    # Training iterations
    parser.add_argument('--num-iters', type=int, default=int(2e5),
                        help='number of iterations for training (default: 2e5)')
    parser.add_argument('--pretrain-iters', type=int, default=int(2e4),
                        help='number of iterations for training (default: 2e4)')
    parser.add_argument('--inline-iters', type=int, default=int(1),
                        help='number of iterations for training the interaction model WITHIN each normal iteration loop (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ratio', type=float, default=0.9,
                    help='ratio of training samples to testing ones')
    # values for determining if significant things are happening
    parser.add_argument('--model-error-significance', type=float, default=.5,
                    help='amount of difference in l2 norm to determine that prediction is happening')
    parser.add_argument('--feature-step', type=float, default=1,
                    help='amount of step in estimating feature significance')
    parser.add_argument('--train-reward-significance', type=float, default=5,
                    help='amount of difference in per-episode reward to determine control')
    # HER/DQN parameters
    parser.add_argument('--select-positive', type=float, default=0.5,
                    help='For hindsight experience replay, selects the positive reward x percent of the time (default .5)')
    parser.add_argument('--resample-timer', type=int, default=-1,
                        help='how often to resample a goal (default: -1)')
    parser.add_argument('--early-stopping', type=int, default=-1,
                        help='stop adding too many terminal states to HER (default: -1)')
    parser.add_argument('--max-hindsight', type=int, default=500,
                        help='most timesteps to look behind for credit assignment (default: 500)')
    parser.add_argument('--her-only-interact', type=int, default=0,
                        help='only resample if an interaction (1) or change(2) occurs at some point')
    parser.add_argument('--use-interact', action='store_true', default=False,
                        help='only resamples HER when an interaction occurs')
    parser.add_argument('--sample-merged', action='store_true', default=False,
                        help='samples a batch from both the main buffer and the HER buffer and merges the batches')
    parser.add_argument('--keep-proximity', action='store_true', default=False,
                    help='keeps in the HER buffer (for iterated supervised learning) the nearby hits')

    #interaction parameters
    parser.add_argument('--train-pair', type=str, nargs='+', default=list(),
                        help='pair of objects to train interaction on, [source, [additional], target] (default: list')
    parser.add_argument('--true-interaction', action='store_true', default=False,
                        help='instead of using the interaction model, uses the actual interaction terms')
    parser.add_argument('--interaction-iters', type=int, default=0,
                        help='number of iterations for training the interaction network with true values (default: 0)' +
                            'overloaded to use trace instead of interaction model for predicting samples if > 0')
    parser.add_argument('--compare-trace', action='store_true', default=False,
                        help='compares with the trace without training an interaction model')
    parser.add_argument('--interaction-distance', type=int, default=0,
                        help='number of steps within to predict interaction (default: 0)')
    parser.add_argument('--interaction-binary', type=float, nargs='+', default=list(),
                        help='difference between P,A, Active greater than, passive less than  (default: empty list)')
    parser.add_argument('--force-mask', type=float, nargs='+', default=list(),
                        help='a hack to control the parameter mask  (default: empty list)')
    parser.add_argument('--interaction-weight', type=float, default=10,
                        help='the weight given to interaction values compared to normal values (default: 1000)')
    parser.add_argument('--interaction-local', type=float, default=0,
                        help='additional sampling weight in range defined by local (should be odd) (default: 0)')
    parser.add_argument('--interaction-probability', type=float, default=1,
                        help='the minimum probability needed to use interaction as termination 1 means interaction is never (solely) used for termination (default: 1)')
    parser.add_argument('--interaction-boosting', type=float, default=-1,
                        help='weights the interaction sampler with states where the interaction model performs better than interaction-prediction to avoid false positives (default -1)')
    parser.add_argument('--interaction-prediction', type=float, default=0,
                        help=('the minimum distance to define the active set (default: 0.3)' + 
                            'overloaded to also represent the decay rate for interaction probability for termination per step until interaction-probability from 1'))
    parser.add_argument('--active-epsilon', type=float, default=.5,
                        help='the minimum distance for a single dimension in the active set default .5 ')
    parser.add_argument('--sample-schedule', type=int, default=-1,
                    help='changes sampling after a certain number of calls')
    parser.add_argument('--sample-distance', type=float, default=.4,
                        help='the distance to sample local values (default: .4)')
    parser.add_argument('--passive-error-cutoff', type=float, default=2,
                        help='the cutoff for error to weight the value (default: 2)')
    parser.add_argument('--passive-weighting', type=float, default=0,
                        help='weight with the passive error, if 0, then passive weighting is not used')
    parser.add_argument('--change-weighting', type=float, default=0,
                        help='weight with the change in state')

    # reward settings
    parser.add_argument('--parameterized-lambda', type=float, default=.5,
                        help='the value given to parameter hits  (default: .5)')
    parser.add_argument('--interaction-lambda', type=float, default=0,
                        help='the value given to interactions  (default: 0)')
    parser.add_argument('--true-reward-lambda', type=float, default=0,
                        help='the value given to true reward negative component (default: 0)')
    parser.add_argument('--reward-constant', type=float, default=-1,
                        help='constant value to add to the reward (default: -1)')
    parser.add_argument('--sum-rewards', action = 'store_true', default=False,
                    help='takes the sum of rewards for temporal extension, rather than a single value')
    # Option Chain Parameters
    parser.add_argument('--base-node', default="Action",
                        help='The name of the lowest node in the option chain (generally should be Action)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='number of samples to take for all_state_next')
    # termination condition parameters
    parser.add_argument('--min-use', type=int, default=5,
                    help='minimum number of seen states to use as a parameter')

    # logging settings
    parser.add_argument('--print-test', action='store_true', default=False,
                        help='prints out values during the test phase of training')
    parser.add_argument('--print-buffer', action='store_true', default=False,
                        help='prints out buffer values for debugging')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=-1,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='data/new_net/',
                        help='directory to save data when adding edges')
    parser.add_argument('--save-pretrain', default='',
                        help='saves the pretrain data if len > 0, suggested data/temp')
    parser.add_argument('--load-pretrain', default='',
                        help='loads the pretrain data from the location if len > 0')
    parser.add_argument('--test-episode', action='store_true', default=False,
                    help='if true, the test will run full epsiodes instead of just until termination')
    parser.add_argument('--test-trials', type=int, default=10,
                    help='number of episodes to run as a test')
    parser.add_argument('--pretest-trials', type=int, default=1,
                    help='number of episodes of random policy to assess performance')
    parser.add_argument('--save-graph', default='',
                        help='directory to save graph data. If empty, does not save the graph')
    parser.add_argument('--save-recycle', type=int, default=-1,
                        help='only saves the last n timesteps (-1 if not used)')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded (when adding edges, where data was recorded to compute min/max)')
    parser.add_argument('--log-only', action ='store_true', default=False,
                        help='record rollouts only stores logs and no state data')
    parser.add_argument('--render', action='store_true', default=False,
                        help='if false, does not render or save renders')
    parser.add_argument('--no-save-action', action ='store_true', default=False,
                        help='saves the highest option level action in record-rollouts')
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
    parser.add_argument('--visualize-param', default="",
                        help='generates images of the parameterized option, default is empty string for no visualization, "nosave" will not save the visualized param')
    
    # environmental variables
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number to use (default: 0)')
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env', default='SelfBreakout',
                        help='environment to train on (Nav2D, SelfBreakout) (default: SelfBreakout)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
    parser.add_argument('--set-time-cutoff', action ='store_true', default=False,
                        help='runs the algorithm without time cutoff to set it')
    parser.add_argument('--time-cutoff', type=int, default=-1,
                        help='sets the duration to switch to the next option (default -1 means no time cutoff)')
    # ONLY FOR BREAKOUT
    parser.add_argument('--drop-stopping', action = 'store_true', default=False,
                    help='returns done when the ball is dropped')

    # load variables
    parser.add_argument('--load-weights', action ='store_true', default=False,
                        help='load the options for the existing network')
    parser.add_argument('--load-intermediate', action ='store_true', default=False,
                        help='load the passive model/interaction to skip passive model training')
    parser.add_argument('--save-loaded-network', action ='store_true', default=False,
                        help='save the graph with the @args load-network as policy to @args save-graph')
    parser.add_argument('--save-intermediate', action ='store_true', default=False,
                        help='save the passive model to skip training later')
    parser.add_argument('--load-network', default="",
                        help='path to network')
    parser.add_argument('--change-option', action ='store_true', default=False,
                        help='switch the option but keep the policy')

    args = parser.parse_args()

    args.discount_factor = args.gamma # TianShou Support
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.critic_lr = args.lr if args.critic_lr < 0 else args.critic_lr
    args.actor_lr = args.lr if args.actor_lr < 0 else args.actor_lr
    args.deterministic_eval = not args.not_deterministic_eval
    args.bound_output = 0 if not args.bound_output else args.time_cutoff
    args.save_action = not args.no_save_action and len(args.record_rollouts) > 0

    return args

def get_edge(edge):
    head = edge.split("->")[0]
    tail = edge.split("->")[1]
    head = head.split(",")
    return head, tail[0]
