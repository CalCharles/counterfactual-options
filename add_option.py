import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel
from ddpg import DDPG

# remove this line later, along with ddpg file

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.rollouts import RLRollouts, get_RL_shapes
from ReinforcementLearning.train_RL import trainRL, Logger
from ReinforcementLearning.Policy.policy import policy_forms, pytorch_model
from ReinforcementLearning.behavior_policy import behavior_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model, interaction_models
from DistributionalModels.distributional_model import load_factored_model
from DistributionalModels.InteractionModels.samplers import samplers
import torch
import numpy as np

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    args.concatenate_param = True
    args.normalized_actions = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.env == "SelfBreakout":
        args.continuous = False
        environment = Screen()
        args.denorm = pytorch_model.wrap([84, 84, 3,3,1], cuda=args.cuda)
        environment_model = BreakoutEnvironmentModel(environment)
    elif args.env == "Nav2D":
        args.continuous = False
        environment = Nav2D()
        environment_model = Nav2DEnvironmentModel(environment)
        args.denorm = pytorch_model.wrap([environment.N, environment.N, 1], cuda=args.cuda)
    elif args.env == "Pend-Gym":
        args.continuous = True
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name="Pendulum-v0")
        environment.env.seed(args.seed)
        environment_model = GymEnvironmentModel(environment)
        args.denorm = pytorch_model.wrap(environment.action_space.high - environment.action_space.low, cuda=args.cuda)
        args.normalized_actions = True
    environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    if args.true_environment:
        args.concatenate_param = False
        dataset_model = interaction_models["dummy"](environment_model=environment_model)
    else:
        dataset_model = load_hypothesis_model(args.dataset_dir)
        dataset_model.environment_model = environment_model


    # extra line for older dataset models
    # dataset_model.sample_continuous = False
    # dataset_model.use_layer_norm = False
    # if not args.true_environment:
    #     dataset_model.interaction_model.use_layer_norm = False
    # REMOVE above

    # reduced state HACK
    # dataset_model.gamma = FeatureSelector([6], {"Paddle": 1})
    # dataset_model.delta = FeatureSelector([6], {"Paddle": 1})
    # dataset_model.selection_binary = torch.ones((1)).cuda()
    # dataset_model.selection_binary = torch.tensor([0,0,1,1,0])
    # REMOVE above once alternative selection implemented

    # print(dataset_model.selection_binary)
    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule)
    pr, models = ObjDict(), (dataset_model, environment_model, sampler) # policy_reward, featurizers
    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    try:
        graph = load_graph(args.graph_dir, args.buffer_steps)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, models, "Action")
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        graph = OptionGraph(nodes, dict(), dataset_model.controllable)
    termination = terminal_forms[args.terminal_type](use_diff=args.use_both==1, use_both=args.use_both==2, name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close, interaction_probability=args.interaction_probability)
    reward = reward_forms[args.reward_type](use_diff=args.use_both==1, epsilon=args.epsilon_close, parameterized_lambda=args.parameterized_lambda, reward_constant= args.reward_constant, interaction_model=dataset_model.interaction_model, interaction_minimum=dataset_model.interaction_minimum) # using the same epsilon for now, but that could change
    print (dataset_model.name)
    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = not args.train and args.object in graph.nodes
    print(load_option, option_name, args.object)

    # hack to fix old versions REMOVE
    # graph.nodes[option_name].option.output_prob_shape = (graph.nodes[option_name].option.dataset_model.delta.output_size(), )
    
    if not load_option:
        pr.policy, pr.behavior_policy, pr.rollouts, pr.termination, pr.reward, pr.next_option = None, None, None, termination, reward, (None if args.true_environment else graph.nodes[option_name].option)
        print("keys", list(graph.nodes.keys()))
        option = option_forms[args.option_type](pr, models, args.object, temp_ext=False)
        if option_name == "Action" or args.object == "Raw":
            option.discrete = not args.continuous
        else:
            option.discrete = False # assumes that all non-base options are continuous
    else:
        option = graph.nodes[args.object].option
        self.assign_models(models)


    option.time_cutoff = args.time_cutoff
    rl_shape_dict = get_RL_shapes(option, environment_model)
    rollouts = RLRollouts(args.buffer_steps, rl_shape_dict)
    print(rl_shape_dict["state"], rl_shape_dict["probs"], rl_shape_dict["param"])
    args.num_inputs, args.num_outputs, args.param_size, args.reshape = rl_shape_dict["state"][0], rl_shape_dict["probs"][0], rl_shape_dict["param"][0], environment.reshape
    print (args.activation)

    if not load_option:
        policy = policy_forms[args.policy_type](**vars(args)) # default args?
    else:
        policy = option.policy

    #  REMOVE LATER
    # actor = torch.load("data/actor.pt")
    # critic = torch.load("data/critic.pt")
    # policy.actor.l1.weight.data.copy_(actor.linear1.weight.data)
    # policy.actor.l2.weight.data.copy_(actor.linear2.weight.data)
    # policy.actor.action_eval.weight.data.copy_(actor.mu.weight.data)
    # policy.actor.l1.bias.data.copy_(actor.linear1.bias.data)
    # policy.actor.l2.bias.data.copy_(actor.linear2.bias.data)
    # policy.actor.action_eval.bias.data.copy_(actor.mu.bias.data)
    # policy.critic.l1.weight.data.copy_(critic.linear1.weight.data)
    # policy.critic.QFunction.l1.weight.data.copy_(critic.linear2.weight.data)
    # policy.critic.QFunction.l2.weight.data.copy_(critic.V.weight.data)
    # policy.critic.l1.bias.data.copy_(critic.linear1.bias.data)
    # policy.critic.QFunction.l1.bias.data.copy_(critic.linear2.bias.data)
    # policy.critic.QFunction.l2.bias.data.copy_(critic.V.bias.data)
    # print(policy.critic.QFunction.l2.weight.data)


    if args.cuda:
        policy.cuda()
        rollouts = rollouts.cuda()
        option.cuda()
        dataset_model.cuda()
    behavior_policy = behavior_forms[args.behavior_type](args)
    if not load_option:
        option.policy, option.behavior_policy, option.rollouts = policy, behavior_policy, rollouts
        option.rollout_params = (rollouts.length, rollouts.shapes)

        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        graph.load_environment_model(environment_model)
    
    print(load_option, option.policy)
    logger = Logger(args, option)
    learning_algorithm = learning_algorithms[args.learning_type](args, option)
    # if args.set_time_cutoff:
    #     option.time_cutoff = -1
    done_lengths, trained = trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph)
    done_lengths = np.array(done_lengths)
    time_cutoff = 100
    if len(done_lengths) > 0:
        time_cutoff = np.round_(np.quantile(done_lengths, .9))
    if trained: # if trained, add control feature to the graph
        graph.cfs += dataset_model.cfselectors

    if args.train and args.save_interval > 0:
        option.save(args.save_dir)
    if args.set_time_cutoff:
        option.time_cutoff = time_cutoff
    if len(args.save_graph) > 0:
        print(args.object, graph.nodes[args.object].option.time_cutoff)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
    print (time_cutoff)
