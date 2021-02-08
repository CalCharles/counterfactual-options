import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from Rollouts.rollouts import ObjDict
from ReinforcementLearning.rollouts import RLRollouts, get_RL_shapes
from ReinforcementLearning.test_RL import testRL, Logger
from ReinforcementLearning.Policy.policy import policy_forms
from ReinforcementLearning.behavior_policy import behavior_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model
from DistributionalModels.distributional_model import load_factored_model
from DistributionalModels.InteractionModels.samplers import samplers
import torch
import numpy as np

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    environment = Screen()
    environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    environment_model = BreakoutEnvironmentModel(environment)
    dataset_model = load_hypothesis_model(args.dataset_dir)
    dataset_model.environment_model = environment_model

    dataset_model.gamma = FeatureSelector([6], {"Paddle": 1})
    dataset_model.delta = FeatureSelector([6], {"Paddle": 1})
    dataset_model.selection_binary = torch.ones((1)).cuda()

    print(dataset_model.selection_binary)
    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule)
    pr, models = ObjDict(), (dataset_model, environment_model, sampler) # policy_reward, featurizers
    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    graph = load_graph(args.graph_dir, args.buffer_steps)
    termination = terminal_forms[args.terminal_type](use_diff=args.use_both==1, use_both=args.use_both==2, name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close, interaction_probability=args.interaction_probability)
    reward = reward_forms[args.reward_type](use_diff=args.use_both==1, epsilon=args.epsilon_close, parameterized_lambda=args.parameterized_lambda, interaction_model=dataset_model.interaction_model, interaction_minimum=dataset_model.interaction_minimum) # using the same epsilon for now, but that could change
    print (dataset_model.name)
    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = args.object in graph.nodes
    print(load_option, args.object)
    option = graph.nodes[args.object].option
    option.assign_models(models)

    rl_shape_dict = get_RL_shapes(option, environment_model)
    rollouts = RLRollouts(args.buffer_steps, rl_shape_dict)
    print(rl_shape_dict["state"], rl_shape_dict["probs"], rl_shape_dict["param"])
    args.num_inputs, args.num_outputs, args.param_size = rl_shape_dict["state"][0], rl_shape_dict["probs"][0], rl_shape_dict["param"][0]
    policy = option.policy
    if args.cuda:
        policy.cuda()
        rollouts = rollouts.cuda()
        option.cuda()
        dataset_model.cuda()
    behavior_policy = behavior_forms[args.behavior_type](args)
    logger = Logger(args, option)
    # if args.set_time_cutoff:
    option.time_cutoff = -1
    done_lengths = testRL(args, rollouts, logger, environment, environment_model, option, names, graph)
