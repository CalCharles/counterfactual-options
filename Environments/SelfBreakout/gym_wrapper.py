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
            # add observation space
            self.observation_space = spaces.Box(low=-84, high=84, shape=(5,))
            self.env_wrapper_fn = self.image_observation_wrapper
        else:
            # add observation space
            self.env_wrapper_fn = self.multi_block_observation_wrapper

        self.seed = env.seed

        self.env.spec = None
        self.metadata = None

    # Relative position between paddle and ball
    def delta_observation_wrapper(self, obs):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])

        # Possibly add Ball and Paddle too?
        return delta

    # Returns uint8_image
    def image_observation_wrapper(self, obs):
        return obs['raw_state']

    def multi_block_observation_wrapper(self, obs):
        return obs


    def reset(self):
        orig_obs = self.env.reset()
        return self.delta_observation_wrapper(orig_obs)

    def step(self, action):
        orig_obs, reward, done, info = self.env.step(action)
        return self.env_wrapper_fn(orig_obs), reward, done, info

    def render(self):
        return self.env.render()
