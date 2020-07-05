import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from ReinforcementLearning.rollouts import RLRollouts
from ReinforcementLearning.train_RL import trainRL, Logger
from ReinforcementLearning.policy import policy_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from Options.termination import terminal_forms
from Counterfactual.counterfactual_dataset import CounterfactualDataset
from Options.option_graph import OptionGraph, OptionNode
from Options.option import Option, PrimitiveOption
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel


if __name__ == '__main__':
	args = get_args()
	environment = Screen()
	environment_model = BreakoutEnvironmentModel(environment)
	dataset_model = load_from_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"))
	try:
		graph = load_from_pickle(os.path.join(args.graph_dir, "graph.pkl"))
	except OSError as e:
		actions = PrimativeOption(None, None, None, environment_model)
		nodes = {'actions': OptionNode('actions', actions)}
		graph = OptionGraph(nodes, dict())
	policy = policy_forms[args.policy_type]() # default args?
	termination = terminal_forms[args.terminal_type](use_diff=args.use_diff==1, use_both=args.use_both==2, name=args.target, min_use=args.min_use, dataset_model=dataset_model)
	option = option_forms[args.option_type](policy, termination, next_level, model, args.object_names, temp_ext=False)
	rollouts = RLRollouts(args.buffer_len, environment_model.shapes_dict)
	logger = Logger(args)
	learning_algorithm = learning_algorithms[args.learning_type](args, option)
	trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm)