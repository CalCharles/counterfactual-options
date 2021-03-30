#TianShou Test

import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel
from ReinforcementLearning.train_RL import trainRL, Logger
import tianshou as ts

if __name__ == '__main__':

	args = get_args()
	torch.cuda.set_device(args.gpu)
	if args.env == "SelfBreakout":
        environment = Screen()
        environment_model = BreakoutEnvironmentModel(environment)
        net = BasicNetwork(input_size=None, output_size=environment.num_actions hidden_sizes = args.hidden_sizes)
    if args.env == "Pendulum-v0":
    	environment = gym.make('Pendulum-v0')
    	environment_model = None
        net = BasicNetwork(input_size=environment.observation_space.shape, output_size=environment.action_space.shape, hidden_sizes = args.hidden_sizes)
    environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    # initialize: policy, 
    class DummyOption():
    	def __init__(self, args): # replace policy with the wrapper that handles param
    		if args.learning_type = "dqn": self.policy = ts.policy.DQNPolicy(net, optim, discount_factor=args.discount_factor, estimation_step=3, target_update_freq=int(args.tau))
    		if args.learning_type = "ddpg": self.policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

    option = DummyOption()
    done_lengths, trained = TSTrainRL(args, rollouts, "Logger replaced by tianshou (eventually)", environment, environment_model, option, "Learning Algorithms replaced by Tianshou", None, None)
