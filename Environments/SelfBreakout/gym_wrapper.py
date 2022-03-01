import gym
from Environments.SelfBreakout.breakout_screen import Screen
from gym import spaces
import numpy as np
import collections

breakout_action_norm = (np.array([0,0,0,0,1.5]), np.array([1,1,1,1,1.5]))
breakout_paddle_norm = (np.array([72, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_state_norm = (np.array([84 // 2, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_block_norm = (np.array([32, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_relative_norm = (np.array([0,0,0,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_paddle_ball_norm = (np.array([20,0,1.5,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_ball_block_norm = (np.array([20,0,-1.5,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))

class BreakoutGymWrapper():
    def __init__(self, env, args, normalize=True):
        self.env = env

        # Note, normalization not supported on images b/c PixelNet already normalizes
        self.normalize = normalize

        # include observation logic
        self.action_space = spaces.Discrete(env.num_actions)

        if args.observation_type == 'delta':
            self.observation_space = spaces.Box(low=-84, high=84, shape=(5,))
            self.env_wrapper_fn = self.delta_observation_wrapper
        elif args.observation_type == 'image':
            self.num_frames = 4
            self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84))
            self.env_wrapper_fn = self.image_observation_wrapper

            self.frame_buffer = collections.deque(maxlen=self.num_frames)
            for _ in range(self.num_frames):
                self.frame_buffer.append(np.zeros((1, 84, 84)))

        elif args.observation_type in ['multi-block-encoding', 'full-encoding']:
            dim = 15 + env.num_blocks * 5 # paddle + ball + delta + blocks
            if env.variant == 'proximity':
                # Add in param
                dim += 5

            self.observation_space = spaces.Box(low=-84, high=84, shape=(dim,))
            self.env_wrapper_fn = self.multi_block_observation_wrapper
        else:
            raise Exception("invalid observation type")

        self.seed = env.seed

        self.env.spec = None
        self.metadata = None
        self.block_dimension = 5

        if env.variant == 'proximity':
            self.ball_paddle_info_dim = 20
        else:
            self.ball_paddle_info_dim = 15

    def normalize_data(self, data, mean, var):
        return (data - mean) / var

    # Relative position between paddle and ball
    def delta_observation_wrapper(self, obs, resample=False):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])
        if self.normalize:
            delta = self.normalize_data(delta, *breakout_relative_norm)

        return delta

    # Returns uint8_image
    def image_observation_wrapper(self, obs, resample=False):
        # Needed to add single channel for torch image processing (Expects channels, h, w)

        self.frame_buffer.append(np.expand_dims(obs['raw_state'], axis=0))

        ret = np.concatenate(self.frame_buffer, axis=0)
        return ret

    def multi_block_observation_wrapper(self, obs, resample=False):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])
        paddle = factored_state['Paddle']
        ball = factored_state['Ball']
        blocks = []

        if self.env.num_blocks == 1:
            blocks.append(np.array(factored_state['Block']))
        else:
            for curr in range(self.env.num_blocks):
                blocks.append(np.array(factored_state[f'Block{curr}']))

        if self.normalize:
            delta = self.normalize_data(delta, *breakout_relative_norm)
            paddle = self.normalize_data(paddle, *breakout_paddle_norm)
            ball = self.normalize_data(ball, *breakout_state_norm)
            for i in range(len(blocks)):
                blocks[i] = self.normalize_data(blocks[i], *breakout_block_norm)


        if self.env.variant == 'proximity' and resample:
            self.env.sampler.sample(obs)

        if self.env.variant == 'proximity':
            normalized_param = self.normalize_data(self.env.sampler.param, *breakout_block_norm)
            flattened_array = np.concatenate([paddle, ball, delta, normalized_param] + blocks)
        else:
            flattened_array = np.concatenate([paddle, ball, delta] + blocks)

        return flattened_array

    def reset(self):
        orig_obs = self.env.reset()
        return self.env_wrapper_fn(orig_obs)

    def step(self, action):
        orig_obs, reward, done, info = self.env.step(action)
        return self.env_wrapper_fn(orig_obs, resample=reward != 0.0), reward, done, info

    def render(self):
        return self.env.render()
