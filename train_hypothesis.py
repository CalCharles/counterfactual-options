import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from EnvironmentModels.environment_model import ModelRollouts, FeatureSelector, ControllableFeature
from Environments.SelfBreakout.breakout_screen import Screen
from Counterfactual.counterfactual_dataset import CounterfactualStateDataset
from Counterfactual.passive_active_dataset import HackedPassiveActiveDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption
from DistributionalModels.InteractionModels.interaction_model import default_model_args, load_hypothesis_model, interaction_models, nf, nfd
from DistributionalModels.InteractionModels.feature_explorer import FeatureExplorer

if __name__ == '__main__':
    args = get_args()
    data = read_obj_dumps(args.dataset_dir, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    environment = Screen(frameskip=args.frameskip)
    environment_model = BreakoutEnvironmentModel(environment)
    hypothesis_model = load_hypothesis_model(args.load_network)

    environment_model.shapes_dict["all_state_next"] = [args.num_samples, environment_model.state_size]
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    last_state = None
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)
    if args.cuda:
        rollouts.cuda()

    active_set = hypothesis_model.determine_active_set(rollouts)
    range_min, range_max = hypothesis_model.determine_range(rollouts, active_set)
    cfs = ControllableFeature(active_set, [range_min, range_max],1)
    # cf_state = CounterfactualStateDataset(environment_model)
    # rollouts = cf_state.generate_dataset(rollouts, controllable_feature_selectors)
    
    # dataset_model.sample_zero = args.sample_zero
    # hypothesis_model = NeuralInteractionForwardModel(environment_model = environment_model, target_name=args.target, contingent_nodes=[n for n in graph.nodes.values() if n.name != args.target])
    # hypothesis_model.train(rollouts)
    model_args = default_model_args(args.predict_dynamics)
    model_args['controllable'], model_args['environment_model'], model_args['cuda'] = [cfs], environment_model, args.cuda
    controllable_entity = cfs.feature_selector.get_entity()[0]
    entity_selection = environment_model.create_entity_selector([controllable_entity, args.target])
    if args.predict_dynamics:
        model_args['output_normalization_function'] = nf5
    else:
        model_args['output_normalization_function'] = nf
    if entity_selection.output_size() == 5:
        model_args['normalization_function'] = nf
    else:
        model_args['normalization_function'] = nfd
    model_args['gamma'] = entity_selection
    model_args['delta'] = environment_model.create_entity_selector([args.target])
    model_args['num_inputs'] = model_args['gamma'].output_size()
    model_args['num_outputs'] = model_args['delta'].output_size()
    model = interaction_models[model_args['model_type']](**model_args)
    train, test = rollouts.split_train_test(args.ratio)
    model.train(train, args, control=cfs, target_name=args.target)
    forward_error, passive_error = model.assess_error(test)
    passed = forward_error > passive_error - args.model_error_significance
    model.save(args.dataset_dir)

    # print("forward", interaction_model.forward_model)
    # print("state", interaction_model.state_model)
    # print("state counts", interaction_model.difference_counts)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)