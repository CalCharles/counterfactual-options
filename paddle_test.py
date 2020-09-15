import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from ReinforcementLearning.rollouts import RLRollouts, get_RL_shapes
from ReinforcementLearning.train_RL import trainRL, Logger
from ReinforcementLearning.policy import policy_forms
from ReinforcementLearning.behavior_policy import behavior_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from Options.Termination.termination import terminal_forms
from Options.option_graph import OptionGraph, OptionNode
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel
import torch
import numpy as np

if __name__ == '__main__':
    args = get_args()
    args.object = "Paddle"
    args.min_use = 10
    args.epsilon_close = .5
    args.optim = 'SGD'
    args.num_steps = 5
    args.gamma = 0.1
    args.batch_size = 5
    args.num_iters = 5
    args.normalize = True
    args.init_form = 'uni'
    args.lr = 1
    torch.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    np.random.seed(0)
    environment = Screen()
    environment_model = BreakoutEnvironmentModel(environment)
    dataset_model = load_from_pickle(os.path.join("data/random", "dataset_model.pkl"))
    actions = PrimitiveOption(None, None, None, None, environment_model, None, None, "Action", ["Actions"], num_params=environment.num_actions)
    nodes = {'Action': OptionNode('Action', actions, action_shape = (1,), num_params=environment.num_actions)}
    graph = OptionGraph(nodes, dict())
    termination = terminal_forms['param'](use_diff=True, use_both=False, name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close)
    reward = reward_forms['bin'](use_diff=args.use_both==1, use_both=args.use_both==2, epsilon=args.epsilon_close) # using the same epsilon for now, but that could change
    names = [args.object, dataset_model.option_name] 
    option = option_forms["discrete"](None, None, termination, graph.nodes[dataset_model.option_name].option, dataset_model, environment_model, reward, args.object, names, temp_ext=False)
    rl_shape_dict = get_RL_shapes(option, environment_model)
    rollouts = RLRollouts(5, rl_shape_dict)
    print(rl_shape_dict["state"], rl_shape_dict["probs"], rl_shape_dict["param"])
    args.num_inputs, args.num_outputs, args.param_size = rl_shape_dict["state"][0], rl_shape_dict["probs"][0], rl_shape_dict["param"][0]
    args.num_layers = 0
    policy = policy_forms[args.policy_type](**vars(args)) # default args?
    if args.cuda:
        policy.cuda()
        rollouts = rollouts.cuda()
        option.cuda()
    print(policy.action_eval.weight)
    behavior_policy = behavior_forms['prob'](args)
    option.policy = policy
    option.behavior_policy = behavior_policy
    logger = Logger(args)
    learning_algorithm = learning_algorithms['a2c'](args, option)
    trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names)