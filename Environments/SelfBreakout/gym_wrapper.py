import gym
from Environments.SelfBreakout.breakout_screen import Screen
from gym import spaces
import numpy as np

class BreakoutGymWrapper():
    def __init__(self, env):
        self.env = env
        self.action_space = spaces.Discrete(env.num_actions)
        self.observation_space = spaces.Box(low=-84, high=84, shape=(5,))
        self.seed = env.seed

        self.env.spec = None
        self.metadata = None


    def delta_observation_wrapper(self, obs):
        factored_state = obs['factored_state']
        delta = np.array(factored_state['Paddle']) - np.array(factored_state['Ball'])
        return delta

    def reset(self):
        orig_obs = self.env.reset()
        return self.delta_observation_wrapper(orig_obs)

    def step(self, action):
        orig_obs, reward, done, info = self.env.step(action)
        return self.delta_observation_wrapper(orig_obs), reward, done, info
