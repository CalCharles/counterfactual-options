import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from EnvironmentModels.environment_model import ModelRollouts
from Environments.SelfBreakout.breakout_screen import Screen
from Counterfactual.counterfactual_dataset import CounterfactualDataset
from Options.option_graph import OptionGraph, OptionNode
from Options.option import OptionLayer, Option, PrimativeOption
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel

if __name__ == '__main__':
	args = get_args()
	data = read_obj_dumps(args.dataset_dir, i=-1, rng = args.num_frames, filename='object_dumps.txt')
	environment = Screen()
	environment_model = BreakoutEnvironmentModel(environment)
	cf_data = CounterfactualDataset(environment_model)
	try:
		graph = load_from_pickle(os.path.join(args.graph_dir, "graph.pkl"))
	except OSError as e:
		actions = OptionLayer([PrimativeOption(None, None, None, environment_model, iden=option) for option in range(environment.num_actions)])
		nodes = {'actions': OptionNode('actions', actions)}
		graph = OptionGraph(nodes, dict())
	rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
	last_state = None
	for data_dict in data:
		insert_dict, last_state = environment_model.get_insert_dict(data_dict, last_state, typed=True)
		rollouts.append(**insert_dict)
	relevant_states, irrelevant_outcomes, outcomes = cf_data.generate_dataset(rollouts, graph.nodes[args.object].option_layer)
	dataset_model = FactoredDatasetModel(environment_model = environment_model, option_level = graph.nodes[args.object].option_layer)
	print(outcomes)
	dataset_model.train(relevant_states, irrelevant_outcomes, outcomes)
	print("obs", dataset_model.observed_differences["Paddle"], dataset_model.difference_counts["Paddle"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Paddle"])