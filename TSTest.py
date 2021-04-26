#TianShou Test

import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel
from ReinforcementLearning.train_RL import TSTrainRL, Logger
import tianshou as ts
import torch, gym
import numpy as np
from Networks.tianshou_networks import BasicNetwork
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import ActorProb, Critic, Actor



if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.set_device(args.gpu)
    if args.env == "SelfBreakout":
        environment = Screen()
        environment_model = BreakoutEnvironmentModel(environment)
        net = BasicNetwork(cuda=args.cuda, num_inputs=None, num_outputs=environment.num_actions, hidden_sizes = args.hidden_sizes)
        environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        environment = gym.make(args.env) # "CartPole-v0, Pendulum-v0"
        environment.seed(args.seed)
        environment_model = None
        action_shape = environment.action_space.shape or environment.action_space.n
        print(environment.observation_space.shape, action_shape)
        if args.learning_type in ['ddpg', 'sac']:
            print(environment.action_space.high[0])
            actor = BasicNetwork(cuda=args.cuda, num_inputs=environment.observation_space.shape, num_outputs=action_shape, hidden_sizes = args.hidden_sizes)
            critic = BasicNetwork(cuda=args.cuda, num_inputs=int(np.prod(environment.observation_space.shape) + np.prod(action_shape)) , num_outputs=1, hidden_sizes = args.hidden_sizes)
            device = 'cpu' if not args.cuda else 'cuda:' + str(args.gpu)
            critic = Critic(critic, device=device).to(device)
            if args.learning_type in ['sac']:
                actor = ActorProb(actor, action_shape, device=device, max_action=environment.action_space.high[0], unbounded=True, conditioned_sigma=True).to(device)
                critic2 = BasicNetwork(cuda=args.cuda, num_inputs=int(np.prod(environment.observation_space.shape) + np.prod(action_shape)) , num_outputs=1, hidden_sizes = args.hidden_sizes)
                critic2 = Critic(critic2, device=device).to(device)
                critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
            else:
                actor = Actor(actor, action_shape, device=device, max_action=environment.action_space.high[0]).to(device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        elif args.learning_type in ['dqn']:
            critic = BasicNetwork(cuda=args.cuda, num_inputs=int(np.prod(environment.observation_space.shape)), num_outputs=action_shape, hidden_sizes = args.hidden_sizes)

        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        net = critic
        optim = critic_optim

    # initialize: policy, 
    class DummyOption():
        def __init__(self, args): # replace policy with the wrapper that handles param
            if args.learning_type == "dqn": 
                self.policy = ts.policy.DQNPolicy(net, optim, discount_factor=args.discount_factor, estimation_step=3, target_update_freq=int(args.tau))
                self.policy.set_eps(args.epsilon)
            if args.learning_type == "ddpg": self.policy = ts.policy.DDPGPolicy(actor, actor_optim, critic, critic_optim,
                                                                                tau=args.tau, gamma=args.gamma,
                                                                                exploration_noise=GaussianNoise(sigma=args.epsilon),
                                                                                estimation_step=args.lookahead, action_space=environment.action_space,
                                                                                action_bound_method='tanh')
            if args.learning_type == "sac": self.policy = ts.policy.SACPolicy(actor, actor_optim, critic, critic_optim, critic2, critic2_optim,
                                                                                tau=args.tau, gamma=args.gamma, alpha=args.alpha,
                                                                                # exploration_noise=GaussianNoise(sigma=args.epsilon),
                                                                                estimation_step=args.lookahead, action_space=environment.action_space,
                                                                                action_bound_method='tanh')
        def get_env_state(self, **kwargs):
            return kwargs["obs"]

    option = DummyOption(args)
    torch.save(option.policy.state_dict(), "data/TSTestPolicy.pt")
    done_lengths, trained = TSTrainRL(args, "rollouts replaced by collector", "Logger replaced by tianshou (eventually)", environment, environment_model, option, "Learning Algorithms replaced by Tianshou", None, None)
