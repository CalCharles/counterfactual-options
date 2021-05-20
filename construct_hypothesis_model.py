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
    elif args.env.find("Pushing") != -1:
        args.continuous = False
        environment = Pushing(pushgripper=True)
        if args.env == "StickPushing":
            environment = Pushing(pushgripper=False)
        environment.seed(args.seed)
        environment_model = PushingEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess

    try:
        graph = load_graph(args.graph_dir)
        controllable_feature_selectors = graph.cfs
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, (None, environment_model), "Action")
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        afs = environment_model.construct_action_selector() 
        controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions-1],1)]
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
            model_args = default_model_args(args.predict_dynamics, 10,5) # input and output sizes should not be needed
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
        hypothesis_model.cpu()
        hypothesis_model.cuda()
        delta, gamma = hypothesis_model.delta, hypothesis_model.gamma
        rollouts.cuda()
        passive_error = hypothesis_model.get_prediction_error(rollouts)
        weights, use_weights, total_live, total_dead, ratio_lambda = hypothesis_model.get_weights(passive_error)     
        trace = hypothesis_model.generate_interaction_trace(rollouts, [hypothesis_model.control_feature.object()], [hypothesis_model.name.split('->')[1]])
        ints = hypothesis_model.get_interaction_vals(rollouts)
        bins, fe, pe = hypothesis_model.get_binaries(rollouts)
        pred = hypothesis_model.predict_next_state(rollouts.get_values("state"))[1]
        print(ints.shape, bins.shape, trace.shape, fe.shape, pe.shape)
        pints, ptrace = pytorch_model.wrap(torch.zeros(ints.shape), cuda=hypothesis_model.iscuda), pytorch_model.wrap(torch.zeros(trace.shape), cuda=args.cuda)
        pints[ints > .5] = 1
        ptrace[trace > 0] = 1
        print(weights.shape, pints.shape, ptrace.shape)
        print_weights = (pytorch_model.wrap(weights, cuda=hypothesis_model.iscuda) + pints.squeeze() + ptrace).squeeze()
        print_weights[print_weights > 1] = 1
        next_state = hypothesis_model.gamma(rollouts.get_values("next_state"))[:,[0,1,5,6,7,8]]
        comb = torch.cat([ints, bins, trace.unsqueeze(1), fe, pe, next_state, pred], dim=1)
        np.set_printoptions(precision =3, floatmode='fixed')
        # for i in range(len(comb[print_weights > 0])):
        #     print(pytorch_model.unwrap(comb[print_weights > 0][i]))

        bin_error = bins.squeeze()-trace.squeeze()
        bin_false_positives = bin_error[bin_error > 0].sum()
        bin_false_negatives = bin_error[bin_error < 0].abs().sum()

        int_bin = ints.clone()
        int_bin[int_bin >= .5] = 1
        int_bin[int_bin < .5] = 0
        int_error = int_bin.squeeze() - trace.squeeze()
        int_false_positives = int_error[int_error > 0].sum()
        int_false_negatives = int_error[int_error < 0].abs().sum()

        comb_error = bins.squeeze() + int_bin.squeeze()
        comb_error[comb_error > 1] = 1
        comb_error = comb_error - trace.squeeze()
        comb_false_positives = comb_error[comb_error > 0].sum()
        comb_false_negatives = comb_error[comb_error < 0].abs().sum()

        for i in range(len(comb[comb_error < 0])):
            print(pytorch_model.unwrap(comb[comb_error < 0][i]))

        print("bin fp, fn", bin_false_positives, bin_false_negatives)
        print("int fp, fn", int_false_positives, int_false_negatives)
        print("com fp, fn", comb_false_positives, comb_false_negatives)
        print("total, tp", trace.shape[0], trace.sum())

        afs = environment_model.construct_action_selector() 
        controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions],1)]
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