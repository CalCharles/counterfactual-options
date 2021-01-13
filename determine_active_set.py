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
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model


if __name__ == '__main__':
    args = get_args()
    data = read_obj_dumps(args.record_rollouts, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    environment = Screen(frameskip=args.frameskip)
    environment_model = BreakoutEnvironmentModel(environment)
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    dataset_model = load_hypothesis_model(args.dataset_dir)
    dataset_model.environment_model = environment_model
    environment_model.shapes_dict["all_state_next"] = [args.num_samples, environment_model.state_size]
    last_state = None
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)
    if args.cuda:
        rollouts.cuda()
    dataset_model.determine_active_set(rollouts)
    dataset_model.save(args.dataset_dir)
    print(dataset_model.selection_binary)
