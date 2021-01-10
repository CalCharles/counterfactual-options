import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from EnvironmentModels.environment_model import ModelRollouts, FeatureSelector, ControllableFeature
from Environments.SelfBreakout.breakout_screen import Screen
from Counterfactual.counterfactual_dataset import CounterfactualStateDataset
from Counterfactual.passive_active_dataset import HackedPassiveActiveDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph, OptionEdge
from Options.option import Option, PrimitiveOption
from DistributionalModels.InteractionModels.interaction_model import default_model_args
from DistributionalModels.InteractionModels.feature_explorer import FeatureExplorer

if __name__ == '__main__':
    args = get_args()
    data = read_obj_dumps(args.dataset_dir, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    environment = Screen(frameskip=args.frameskip)
    environment_model = BreakoutEnvironmentModel(environment)
    try:
        graph, controllable_feature_selectors = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, (None, environment_model), "Action")
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        graph = OptionGraph(nodes, dict())
        afs = FeatureSelector([environment_model.indexes['Action'][1] - 1], {'Action': environment_model.object_sizes['Action'] - 1})
        controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions],1)]
    graph.load_environment_model(environment_model)
    environment_model.shapes_dict["all_state_next"] = [args.num_samples, environment_model.state_size]
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)

    last_state = None
    for cfs in controllable_feature_selectors:
        graph.nodes[cfs.object()].option.set_behavior_epsilon(0)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)
    if args.cuda:
        rollouts.cuda()
    # cf_state = CounterfactualStateDataset(environment_model)
    # rollouts = cf_state.generate_dataset(rollouts, controllable_feature_selectors)
    
    # dataset_model.sample_zero = args.sample_zero
    # hypothesis_model = NeuralInteractionForwardModel(environment_model = environment_model, target_name=args.target, contingent_nodes=[n for n in graph.nodes.values() if n.name != args.target])
    # hypothesis_model.train(rollouts)
    model_args = default_model_args()
    model_args.factor, model_args.num_layers, model_args.interaction_binary, model_args.interaction_prediction = args.factor, args.num_layers, args.interaction_binary, args.interaction_prediction
    model_args['controllable'], model_args['environment_model'] = controllable_feature_selectors, environment_model
    feature_explorer = FeatureExplorer(graph, controllable_feature_selectors, environment_model, model_args) # args should contain the model args, might want subspaces for arguments or something since args is now gigantic
    hypothesis_model, delta, gamma = feature_explorer.search(rollouts, args) # again, args contains the training parameters, but we might want subsets since this is a ton of parameters more 

    # save the cfs
    afs = FeatureSelector([environment_model.indexes['Action'][1] - 1], {'Action': environment_model.object_sizes['Action'] - 1})
    controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions],1)]
    hypothesis_model.determine_active_set(rollouts)
    hypothesis_model.save(args.save_dir)
    print(hypothesis_model.selection_binary)
    # print("forward", interaction_model.forward_model)
    # print("state", interaction_model.state_model)
    # print("state counts", interaction_model.difference_counts)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)