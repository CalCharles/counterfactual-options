from arguments import get_args
from file_management import read_obj_dumps
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import screen
from Counterfactual.counterfactual_dataset import CounterfactualDataset
from Options.option_graph import OptionGraph, OptionNode

if __name__ == '__main__':
	args = get_args()
	data = read_obj_dumps(args.dataset_path, i=-1, rng = args.num_frames, filename='object_dumps.txt')
	environment = screen()
	environment_model = BreakoutEnvironmentModel(environment)
	cf_data = CounterfactualDataset(environment_model)
	try:
		graph = load_from_pickle(os.join(args.graph_dir, "graph.pkl"))
	except OSError as e:
		actions = OptionLayer([Option(None, None, None, breakout_environment_model) for option in range(environment.num_actions)])
		nodes = {'actions': OptionNode(actions)}
		graph = OptionGraph(nodes)
	rollouts = ModelRollouts(len(data), environment_model.object_sizes)
	for data_dict in data:
		rollouts.append(data_dict)
	relevant_states, irrelevant_outcomes, outcomes = cf_data.generate_dataset(rollouts, graph.nodes[args.object])
	dataset_model = FactoredDatasetModel(environment_model = environment_model, option_level = graph.nodes[args.object].option_layer)
	dataset_model.train(relevant_states, irrelevant_outcomes, outcomes)
	print(dataset_model.observed_differences)