import argparse
import os
import pprint
import imageio

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy, RainbowPolicy, DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger

from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from Baselines.shared_train import make_breakout_env, make_breakout_env_fn, VideoCollector
from Baselines.networks import DQN, Rainbow
from Baselines.env_args import add_env_args

variant_timeout_limits = { 'big_block' : 300,
                           'single_block' : 300,
                           'negative_rand_row' : 2000,
                           'center_large' : 3000,
                           'breakout_priority_large' : 3000,
                           'harden_single' : 3000,
                           'default' : 5000,
                           'proximity' : 200
                           }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['dqn', 'rainbow', 'sac'])
    parser.add_argument('--model-path', type=str)

    # DQN/Rainbow Arguments
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    parser.add_argument('--noisy-std', type=float, default=0.1)
    parser.add_argument('--no-dueling', action='store_true', default=False)
    parser.add_argument('--no-noisy', action='store_true', default=False)
    parser.add_argument('--no-priority', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--beta-anneal-step', type=int, default=5000000)
    parser.add_argument('--no-weight-norm', action='store_true', default=False)
    parser.add_argument('--target-update-freq', type=int, default=500)

    # Shared Arguments
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)

    # SAC Arguments
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)


    add_env_args(parser)

    # Rollout Args
    parser.add_argument('--seed', type=int, default=238)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--output-path', type=str, default='rollout.mp4')

    return parser.parse_args()


def test(args=get_args()):
    env = make_breakout_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    test_envs = DummyVectorEnv([make_breakout_env_fn(args)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    if args.algorithm == 'rainbow':
        observation_info = { 'observation-type' : args.observation_type,
                             'obj-dim' : env.block_dimension,
                             'first-obj-dim' : env.ball_paddle_info_dim
                             }

        net = Rainbow(args.state_shape,
                      args.action_shape,
                      args.num_atoms,
                      args.noisy_std,
                      args.device,
                      is_dueling=not args.no_dueling,
                      is_noisy=not args.no_noisy,
                      observation_info=observation_info
                      ).to(args.device)

        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        # define policy
        policy = RainbowPolicy(
            net,
            optim,
            args.gamma,
            args.num_atoms,
            args.v_min,
            args.v_max,
            target_update_freq=args.target_update_freq
        ).to(args.device)


    elif args.algorithm == 'dqn':
        observation_info = { 'observation-type' : args.observation_type,
                             'obj-dim' : env.block_dimension,
                             'first-obj-dim' : env.ball_paddle_info_dim
                             }

        net = DQN(args.state_shape, args.action_shape, args.device, observation_info).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        # define policy
        policy = DQNPolicy(
            net,
            optim,
            args.gamma,
            target_update_freq=args.target_update_freq
        ).to(args.device)

    elif args.algorithm == 'sac':
        # TODO: Modify SAC networks to deal w different env types
        net = Net(*args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        actor = Actor(net, args.action_shape, softmax_output=False, device=args.device).to(args.device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic1 = Critic(net_c1, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
        net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        critic2 = Critic(net_c2, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

        # better not to use auto alpha in CartPole
        if args.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
            args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim,
                args.tau,
                args.gamma,
                args.alpha,
                estimation_step=args.n_step,
                reward_normalization=args.rew_norm
        )

    else:
        raise("Must specify valid algorithm to offline baseline trainer")

    policy.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("Loaded agent from: ", args.model_path)

    policy.eval()
    if args.algorithm in ['rainbow', 'dqn']:
        policy.set_eps(args.eps_test)

    episode_limit = variant_timeout_limits[args.variant]
    timeout_penalty = env.env.timeout_penalty
    test_collector = VideoCollector(policy, test_envs, exploration_noise=args.algorithm in ['rainbow', 'dqn'], episode_limit=episode_limit, timeout_penalty=timeout_penalty)

    test_envs.seed(args.seed)
    print("Testing agent ...")
    test_collector.reset()
    result = test_collector.collect(
        n_episode=args.num_episodes, render=0.05
    )

    writer = imageio.get_writer(args.output_path)
    for img in result['saved_images']:
        writer.append_data(img)

    writer.close()

    rew = result["rews"].mean()
    print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    assessment = result["assessment"].sum() / result["n/ep"]
    print(f'Mean assessment over each episode: {assessment}')

    drops = result["drops"].sum() / result["n/ep"]
    print(f'Mean number drops over each episode: {drops}')

if __name__ == '__main__':
    test_dqn(get_args())
