import torch
import numpy as np
import gym
# from matplotlib.pyplot import imshow
# import matplotlib as plt
from copy import deepcopy as dc
from Environments.environment_specification import RawEnvironment

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action

class Gym(RawEnvironment): # wraps openAI gym environment
    def __init__(self, frameskip=1, gym_name=""):
        super().__init__()
        self.frameskip = frameskip
        self.env = NormalizedActions(gym.make(gym_name))
        self.action_space = self.env.action_space
        self.action_shape = self.action_space.shape
        self.observation_space = self.env.observation_space
        self.observation_shape = self.observation_space.shape
        self.discrete_actions = False
        self.num_actions = 1
        self.done = 0
        self.reward = 0

        self.extracted_state = self.dict_state(self.env.observation_space.sample(), 0, 0, self.env.action_space.sample())
    
    def reset(self):
        self.frame = self.env.reset()
        self.extracted_state = self.dict_state(self.env.observation_space.sample(), 0, 0, self.env.action_space.sample())
        return {"raw_state": self.frame, "factored_state": self.extracted_state}

    def step(self, action):
        observation, self.reward, self.done, info = self.env.step(action)
        # print("true_outputs", observation, self.reward, self.done, action)
        # print(observation, reward, self.done, _)
        # TODO: not rendering
        if len(action.shape) == 0:
            action = np.array([action])
        extracted_state = self.dict_state(observation, self.reward, self.done, action)
        frame = observation
        self.extracted_state, self.frame = extracted_state, frame
        if self.done:
            self.reset()
            # print("FINISHED AN EPISODE")
        return {"raw_state": frame, "factored_state": extracted_state}, self.reward, int(self.done), info

    def extracted_state_dict(self):
        return dc(self.extracted_state)

    def dict_state(self, observation, reward, done, action):
        return {"State": observation, "Frame": observation, "Object": observation, "Reward": np.array([reward]), "Done": np.array([float(done)]), "Action": action}

    def toString(self, extracted_state):
        names = ["Action", "State", "Frame", "Object", "Reward", "Done"]
        es = ""
        for name in names:
            es += name + ":" + " ".join(map(str, extracted_state[name])) + "\t"
        return es

    def get_state(self):
        return {'raw_state': self.frame, 'factored_state': self.extracted_state_dict()}
