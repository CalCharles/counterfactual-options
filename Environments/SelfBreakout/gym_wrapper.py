import gym
from Environments.SelfBreakout.breakout_screen import Screen
from gym import spaces
import numpy as np

class BreakoutGymWrapper():
    def __init__(self, env, args):
        self.env = env

        # include observation logic
        self.action_space = spaces.Discrete(env.num_actions)

        if args.observation_type == 'delta':
            self.observation_space = spaces.Box(low=-84, high=84, shape=(5,))
            self.env_wrapper_fn = self.delta_observation_wrapper
        elif args.observation_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84))
            self.env_wrapper_fn = self.image_observation_wrapper
        elif args.observation_type in ['multi-block-encoding', 'full-encoding']:
            dim = 15 + env.num_blocks * 5 # paddle + ball + delta + blocks
            self.observation_space = spaces.Box(low=-84, high=84, shape=(dim,))
            self.env_wrapper_fn = self.multi_block_observation_wrapper
        else:
            raise Exception("invalid observation type")

        self.seed = env.seed

        self.env.spec = None
        self.metadata = None
        self.block_dimension = 5
        self.ball_paddle_info_dim = 15

    # Relative position between paddle and ball
    def delta_observation_wrapper(self, obs):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])

        # Possibly add Ball and Paddle too?
        return delta

    # Returns uint8_image
    def image_observation_wrapper(self, obs):
        # Needed to add single channel for torch image processing (Expects channels, h, w)
        return np.expand_dims(obs['raw_state'], axis=0)

    def multi_block_observation_wrapper(self, obs):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])

        blocks = []

        if self.env.num_blocks == 1:
            blocks.append(np.array(factored_state['Block']))
        else:
            for curr in range(self.env.num_blocks):
                blocks.append(np.array(factored_state[f'Block{curr}']))

        flattened_array = np.concatenate([factored_state['Paddle'], factored_state['Ball'], delta] + blocks)
        return flattened_array


    def reset(self):
        orig_obs = self.env.reset()
        return self.env_wrapper_fn(orig_obs)

    def step(self, action):
        orig_obs, reward, done, info = self.env.step(action)
        return self.env_wrapper_fn(orig_obs), reward, done, info

    def render(self):
        return self.env.render()
