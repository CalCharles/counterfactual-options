import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from EnvironmentModels.environment_model import ModelRollouts
from Environments.SelfBreakout.breakout_screen import Screen
from Counterfactual.counterfactual_dataset import CounterfactualDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel

if __name__ == '__main__':
    args = get_args()
    data = read_obj_dumps(args.dataset_dir, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    environment = Screen()
    environment_model = BreakoutEnvironmentModel(environment)
    cf_data = CounterfactualDataset(environment_model)
    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, None, None, None, environment_model, None, None, "Action", ["Actions"], num_params=environment.num_actions)
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,), num_params=environment.num_actions)}
        graph = OptionGraph(nodes, dict())
    graph.load_environment_model(environment_model)
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    last_state = None
    graph.nodes[args.object].option.set_behavior_epsilon(0)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True)
        rollouts.append(**insert_dict)
    relevant_states, irrelevant_outcomes, outcomes = cf_data.generate_dataset(rollouts, graph.nodes[args.object])
    # dataset_model.sample_zero = args.sample_zero
    dataset_model = FactoredDatasetModel(environment_model = environment_model, option_node=graph.nodes[args.object])
    dataset_model.train(relevant_states, irrelevant_outcomes, outcomes)
    print(dataset_model.observed_differences["Ball"])
    if len(args.target) > 0:
        print("reducing")
        dataset_model.reduce_range([args.target])
    dataset_model.save(args.dataset_dir)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)