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
from tianshou.utils.net.discrete import Actor

from Baselines.shared_train import make_breakout_env, make_breakout_env_fn, VideoCollector
from Baselines.networks import DQN, Rainbow, SACNet
from Baselines.env_args import add_env_args

from Networks.critic import BoundedDiscreteCritic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['dqn', 'rainbow', 'sac'])


    parser.add_argument('--seed', type=int, default=238)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--video-log-period', type=int, default=5)
    parser.add_argument('--save-checkpoint-period', type=int, default=None)


    # DQN/Rainbow Arguments
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
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--encoded-state-dim', type=int, default=128)


    # Shared Arguments
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)


    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--test-steps', type=int, default=2400)

    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)

    add_env_args(parser)

    return parser.parse_args()


def test(args=get_args()):
    env = make_breakout_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = DummyVectorEnv([make_breakout_env_fn(args)])
    test_envs = DummyVectorEnv([make_breakout_env_fn(args)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
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

        if args.no_priority:
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                ignore_obs_next=True,
            )
        else:
            buffer = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                ignore_obs_next=True,
                alpha=args.alpha,
                beta=args.beta,
                weight_norm=not args.no_weight_norm
            )

        log_path = os.path.join(args.logdir, args.variant, args.observation_type, f'rainbow-seed{args.seed}')
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

        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
        )

        log_path = os.path.join(args.logdir, args.variant, args.observation_type, f'dqn-seed{args.seed}')
    elif args.algorithm == 'sac':
        observation_info = { 'observation-type' : args.observation_type,
                             'obj-dim' : env.block_dimension,
                             'first-obj-dim' : env.ball_paddle_info_dim
                             }

        net = SACNet(args.state_shape, args.encoded_state_dim, args.device, observation_info)
        actor = Actor(net, args.action_shape, softmax_output=False, device=args.device).to(args.device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        net_c1 = SACNet(args.state_shape, args.encoded_state_dim, args.device, observation_info)
        critic1 = BoundedDiscreteCritic(net_c1, last_size=args.action_shape,
                        device=args.device).to(args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

        net_c2 = SACNet(args.state_shape, args.encoded_state_dim, args.device, observation_info)
        critic2 = BoundedDiscreteCritic(net_c2, last_size=args.action_shape,
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

        buffer = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
        log_path = os.path.join(args.logdir, args.variant, args.observation_type, f'sac-seed{args.seed}')
    else:
        raise("Must specify valid algorithm to offline baseline trainer")


    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = VideoCollector(policy, train_envs, buffer, exploration_noise=args.algorithm in ['rainbow', 'dqn'])
    test_collector = VideoCollector(policy, test_envs, exploration_noise=args.algorithm in ['rainbow', 'dqn'])
    # log
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        if args.algorithm in ['rainbow', 'dqn']:
            # nature DQN setting, linear decay in the first 1M steps
            if env_step <= 1e6:
                eps = args.eps_train - env_step / 1e6 * \
                    (args.eps_train - args.eps_train_final)
            else:
                eps = args.eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        if args.algorithm in ['rainbow', 'dqn']:
            policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()

        if args.algorithm in ['rainbow', 'dqn']:
            policy.set_eps(args.eps_test)

        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            if args.algorithm == 'rainbow':
                buffer = PrioritizedVectorReplayBuffer(
                    args.buffer_size,
                    buffer_num=len(test_envs),
                    ignore_obs_next=True,
                    alpha=args.alpha,
                    beta=args.beta
                )
            elif args.algorithm == 'dqn':
                buffer = VectorReplayBuffer(
                    args.buffer_size,
                    buffer_num=len(test_envs),
                    ignore_obs_next=True,
                )
            elif args.algorithm == 'sac':
                buffer = VectorReplayBuffer(
                    args.buffer_size,
                    buffer_num=len(test_envs),
                )

            collector = VideoCollector(policy, test_envs, buffer, exploration_noise=args.algorithm in ['rainbow', 'dqn'])
            result = collector.collect(n_step=args.buffer_size, render=0.05)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_step=args.test_steps, render=0.05
            )

        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

        return result


    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, 'checkpoint.pth')
        torch.save({'model': policy.state_dict()}, ckpt_path)

        if args.save_checkpoint_period != None and epoch % args.save_checkpoint_period == 0:
            ckpt_path = os.path.join(log_path, f'checkpoint-epoch{epoch}.pth')
            torch.save({'model' : policy.state_dict()}, ckpt_path)

        if epoch % args.video_log_period == 0:
            result = watch()
            rew = result["rews"].mean()

            if result["n/ep"] == 0:
                assess = result["assessment"].mean()
            else:
                assess = result["assessment"] / result["n/ep"]

            drops = result["drops"].sum()
            images = result['saved_images']

            logger.write("eval/env_step", env_step, {"eval/rew" : rew, "eval/assess" : assess, "eval/drops" : drops, "eval/n_ep" : result["n/ep"]})

            fname = os.path.join(log_path, f'iter-{epoch}.mp4')
            writer = imageio.get_writer(fname, fps=20)

            for img in images:
                writer.append_data(img)

            writer.close()

        return ckpt_path

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
