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
        gymenv = gym.make(gym_name)
        self.discrete_actions = type(gymenv.action_space) == gym.spaces.Discrete
        self.env = gymenv if self.discrete_actions else NormalizedActions(gymenv)
        self.action_space = self.env.action_space
        self.action_shape = self.action_space.shape
        self.observation_space = self.env.observation_space
        self.state_space = self.env.observation_space
        self.observation_shape = self.observation_space.shape
        self.num_actions = 1
        self.done = 0
        self.reward = 0
        self.action = np.zeros(self.action_shape)
        if len(self.action.shape) == 0:
            self.action = np.array([self.action])

        self.frame = self.env.observation_space.sample()
        self.extracted_state = self.dict_state(self.frame, 0, 0, self.action)


    def seed(self, seed):
        super().seed(seed)
        self.env.seed(seed)

    def reset(self):
        self.frame = self.env.reset()
        self.extracted_state = self.dict_state(self.frame, self.reward, self.done, self.action)
        # print("resetting")
        return {"raw_state": self.frame, "factored_state": self.extracted_state}

    def step(self, action):
        self.action = action
        observation, self.reward, self.done, info = self.env.step(action)
        # print("true_outputs", observation, self.reward, self.done, action)
        # TODO: not rendering
        if len(action.shape) == 0:
            action = np.array([action])
            self.action = action
        extracted_state = self.dict_state(observation, self.reward, self.done, action)
        frame = observation
        self.extracted_state, self.frame = extracted_state, frame
        # if self.done:
        #     self.reset()
        #     self.extracted_state["Action"] = action
            # print("FINISHED AN EPISODE")
        # print(self.extracted_state)
        return {"raw_state": frame, "factored_state": extracted_state}, self.reward, bool(self.done), info

    def extracted_state_dict(self):
        return dc(self.extracted_state)

    def dict_state(self, observation, reward, done, action):
        return {"State": observation, "Frame": observation, "Object": observation, "Reward": np.array([reward]), "Done": np.array([int(done)]), "Action": action}

    def toString(self, extracted_state):
        names = ["Action", "State", "Frame", "Object", "Reward", "Done"]
        es = ""
        for name in names:
            es += name + ":" + " ".join(map(str, extracted_state[name])) + "\t"
        return es

    def get_state(self):
        return {'raw_state': self.frame, 'factored_state': self.extracted_state_dict()}
