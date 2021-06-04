import os, torch
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from EnvironmentModels.environment_model import ModelRollouts, FeatureSelector, ControllableFeature

from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen

from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D

from EnvironmentModels.Pushing.pushing_environment_model import PushingEnvironmentModel
from Environments.Pushing.screen import Pushing

from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel

from Counterfactual.counterfactual_dataset import CounterfactualStateDataset
from Counterfactual.passive_active_dataset import HackedPassiveActiveDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph, OptionEdge
from Options.option import Option, PrimitiveOption
from DistributionalModels.InteractionModels.interaction_model import default_model_args, load_hypothesis_model
from DistributionalModels.InteractionModels.feature_explorer import FeatureExplorer
from Networks.network import pytorch_model
import numpy as np

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=300)
    if args.env == "SelfBreakout":
        args.continuous = False
        environment = Screen()
        environment.seed(args.seed)
        environment_model = BreakoutEnvironmentModel(environment)
    elif args.env == "Nav2D":
        args.continuous = False
        environment = Nav2D()
        environment.seed(args.seed)
        environment_model = Nav2DEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess
    elif args.env.find("2DPushing") != -1:
        args.continuous = False
        environment = Pushing(pushgripper=True)
        if args.env == "StickPushing":
            environment = Pushing(pushgripper=False)
        environment.seed(args.seed)
        environment_model = PushingEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess
    elif args.env.find("RoboPushing") != -1:
        from EnvironmentModels.RobosuitePushing.robosuite_pushing_environment_model import RobosuitePushingEnvironmentModel
        from Environments.RobosuitePushing.robosuite_pushing import RoboPushingEnvironment

        args.continuous = True
        environment = RoboPushingEnvironment(control_freq=2, horizon=30, renderable=False)
        environment.seed(args.seed)
        environment_model = RobosuitePushingEnvironmentModel(environment)
    try:
        graph = load_graph(args.graph_dir)
        controllable_feature_selectors = graph.cfs
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, (None, environment_model), "Action")
        nodes = {'Action': OptionNode('Action', actions, action_shape = environment.action_shape)}
        afs = environment_model.construct_action_selector()
        if environment.discrete_actions:
            controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions-1],1)]
        else:
            controllable_feature_selectors = list()
            for i, af in enumerate(afs):
                step = (environment.action_space.high[i] - environment.action_space.low[i]) / 3
                controllable_feature_selectors.append(ControllableFeature(af, [environment.action_space.low[i],environment.action_space.high[i]],step))

        graph = OptionGraph(nodes, dict(), controllable_feature_selectors)
    graph.load_environment_model(environment_model)
    environment_model.shapes_dict["all_state_next"] = [args.num_samples, environment_model.state_size]

    last_state = None
    for cfs in controllable_feature_selectors:
        # print(cfs.object(), graph.nodes[cfs.object()].option.policy)
        if cfs.object() != "Action":
            graph.nodes[cfs.object()].option.policy.set_eps(0)
    
    data = read_obj_dumps(args.record_rollouts, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)
    # UNCOMMENT above
    # REMOVE LATER: saves rollouts so you don't have to run each time
    # save_to_pickle("data/rollouts.pkl", rollouts)
    # rollouts = load_from_pickle("data/rollouts.pkl")
    # if args.cuda:
    #     rollouts.cuda()
    # print(len(data), rollouts.filled)
    # REMOVE ABOVE

    # cf_state = CounterfactualStateDataset(environment_model)
    # rollouts = cf_state.generate_dataset(rollouts, controllable_feature_selectors)
    
    # Just run selection binary

    # dataset_model.sample_zero = args.sample_zero
    # hypothesis_model = NeuralInteractionForwardModel(environment_model = environment_model, target_name=args.target, contingent_nodes=[n for n in graph.nodes.values() if n.name != args.target])
    # hypothesis_model.train(rollouts)
    if args.train:
        success = False
        if args.load_weights: # retraining an existing model
            hypothesis_model = load_hypothesis_model(args.dataset_dir) 
            hypothesis_model.cpu()
            hypothesis_model.cuda()
            train, test = rollouts.split_train_test(args.ratio)
            if args.cuda:
                hypothesis_model.to("cuda:"+str(args.gpu))
            hypothesis_model.train(train, args, control=hypothesis_model.control_feature, target_name=hypothesis_model.name.split("->")[1])
            success = True
        else:
            model_args = default_model_args(args.predict_dynamics, 10,5, args.policy_type, var=args.norm_variance, basevar = args.base_variance) # input and output sizes should not be needed
            model_args.hidden_sizes, model_args.interaction_binary, model_args.interaction_prediction, model_args.init_form, model_args.activation = args.hidden_sizes, args.interaction_binary, args.interaction_prediction, args.init_form, args.activation
            model_args['controllable'], model_args['environment_model'] = controllable_feature_selectors, environment_model
            feature_explorer = FeatureExplorer(graph, controllable_feature_selectors, environment_model, model_args) # args should contain the model args, might want subspaces for arguments or something since args is now gigantic
            print(rollouts.filled)
            exploration = feature_explorer.search(rollouts, args) # again, args contains the training parameters, but we might want subsets since this is a ton of parameters more 
            if exploration is None:
                print("FAILED TO LOCATE RELATED FEATURES")
                success = False
            else:
                hypothesis_model, delta, gamma = exploration
                success = True
            # save the cfs
        if success:
            hypothesis_model.cpu()
            hypothesis_model.save(args.save_dir)
    else: # if not training, determine and save the active set selection binary
        hypothesis_model = load_hypothesis_model(args.dataset_dir)
        hypothesis_model.control_feature = controllable_feature_selectors[0] if environment.discrete_actions else controllable_feature_selectors
        hypothesis_model.active_epsilon = args.active_epsilon
        hypothesis_model.environment_model = environment_model
        hypothesis_model.cpu()
        hypothesis_model.cuda()
        delta, gamma = hypothesis_model.delta, hypothesis_model.gamma
        rollouts.cuda()
        passive_error = hypothesis_model.get_prediction_error(rollouts)
        weights, use_weights, total_live, total_dead, ratio_lambda = hypothesis_model.get_weights(passive_error, passive_error_cutoff=args.passive_error_cutoff)     
        # trace = load_from_pickle("data/trace.pkl").cpu().cuda()
        if args.env != "RoboPushing":
            hypothesis_model.compute_interaction_stats(rollouts, passive_error_cutoff=args.passive_error_cutoff)
        # afs = environment_model.construct_action_selector() 
        # controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions],1)]
        hypothesis_model.determine_active_set(rollouts)
        hypothesis_model.collect_samples(rollouts)
        hypothesis_model.cpu()
        hypothesis_model.save(args.save_dir)
        print(hypothesis_model.selection_binary)

    # interaction model statistics:
    # False positive (interaction at non-interaction), false negative (non-interaction at interaction), low value (predicting interaction with low confidence)
    # high error forward modeling not working properly, no variance: forward model not sensitive to changes 9or interaction model

    # print("forward", interaction_model.forward_model)
    # print("state", interaction_model.state_model)
    # print("state counts", interaction_model.difference_counts)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)