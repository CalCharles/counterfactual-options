import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from ReinforcementLearning.rollouts import RLRollouts, get_RL_shapes
from ReinforcementLearning.train_RL import trainRL, Logger
from ReinforcementLearning.policy import policy_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from Options.Termination.termination import terminal_forms
from Options.option_graph import OptionGraph, OptionNode
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel


if __name__ == '__main__':
	args = get_args()
	environment = Screen()
	environment_model = BreakoutEnvironmentModel(environment)
	dataset_model = load_from_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"))
	try:
		graph = load_from_pickle(os.path.join(args.graph_dir, "graph.pkl"))
	except OSError as e:
		actions = PrimitiveOption(None, None, None, environment_model, None, None, num_params=environment.num_actions)
		nodes = {'Action': OptionNode('Action', actions, action_shape = (1,), num_params=environment.num_actions)}
		graph = OptionGraph(nodes, dict())
	termination = terminal_forms[args.terminal_type](use_diff=args.use_both==1, use_both=args.use_both==2, name=args.object, min_use=args.min_use, dataset_model=dataset_model)
	reward = reward_forms[args.reward_type](use_diff=args.use_both==1, use_both=args.use_both==2)
	option = option_forms[args.option_type](None, termination, graph.nodes[dataset_model.option_name].option, dataset_model, reward, args.object, temp_ext=False)
	rl_shape_dict = get_RL_shapes(option, environment_model)
	rollouts = RLRollouts(args.buffer_steps, rl_shape_dict)
	args.num_inputs, args.num_outputs = rl_shape_dict["state"][0], rl_shape_dict["probs"][0]
	policy = policy_forms[args.policy_type](**vars(args)) # default args?
	option.policy = policy
	logger = Logger(args)
	learning_algorithm = learning_algorithms[args.learning_type](args, option)
	names = [option.object_name, dataset_model.option_name] 
	trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names)