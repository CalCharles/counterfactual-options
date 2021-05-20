import numpy as np
import os, cv2, time, torch
from EnvironmentModels.environment_model import EnvironmentModel

class PushingEnvironmentModel(EnvironmentModel):
    def __init__(self, env):
        super().__init__(env)
        self.object_names = ["Action", "Gripper", "Block", 'Target', 'Done', "Reward"] # TODO: Reward missing from the objects
        self.object_sizes = {"Action": 5, "Gripper": 5, "Block": 5, 'Target': 5, 'Done': 1, "Reward": 1}
        self.object_num = {"Action": 1, "Gripper": 1, "Block": 1, 'Target': 1, 'Done': 1, "Reward": 1}
        self.enumeration = {"Action": [0,1], "Gripper": [1,2], "Block": [2,3], 'Target': [3,4], 'Done':[4,5], "Reward":[5,6]}
        if not env.pushgripper: # add the stick in the proper location
            self.object_names = self.object_names[:2] + ["Stick"] + self.object_names[2:]
            self.object_sizes["Stick"] = 5
            self.object_num["Stick"] = 1
            self.enumeration["Stick"] = [2,3]
            self.enumeration["Block"], self.enumeration["Target"], self.enumeration["Done"], self.enumeration["Reward"] = [3,4], [4,5], [5,6], [6,7]
        self.state_size = sum([self.object_sizes[n] * self.object_num[n] for n in self.object_names])
        self.shapes_dict = {"state": [self.state_size], "next_state": [self.state_size], "state_diff": [self.state_size], "action": [1], "done": [1]}
        self.param_size = self.state_size
        self.set_indexes()

    def get_interaction_trace(self, name):
        trace = []
        for i in range(*self.enumeration[name]):
            # print(name, self.environment.objects[i].interaction_trace)
            trace.append(self.environment.objects[i].interaction_trace)
        return trace

    def set_interaction_traces(self, factored_state):
        self.set_from_factored_state(factored_state)
        self.environment.step(factored_state["Action"][-1])
        self.set_from_factored_state(factored_state)


    def get_factored_state(self, instanced = False): # "instanced" indicates if a single type can have multiple instances (true), or if all of the same type is grouped into a single vector
        factored_state = {n: [] for n in self.object_names}
        if not instanced:
            for o in self.environment.objects:
                for n in self.object_names:
                    if o.name.find(n) != -1:
                        factored_state[n] += o.pos.tolist() + o.vel.tolist() + [o.attribute]
                        break
            for n in factored_state.keys():
                factored_state[n] = np.array(factored_state[n])
        else:
            factored_state = {o.name: np.array(o.pos.tolist() + o.vel.tolist() + [o.attribute]) for o in self.environment.objects}
        factored_state["Done"] = np.array([float(self.environment.done)])
        factored_state["Reward"] = np.array([float(self.environment.reward)])
        return factored_state


    def set_from_factored_state(self, factored_state, instanced = False, seed_counter=-1):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.environment.seed_counter = seed_counter
            self.environment.block.reset_seed = seed_counter
        self.environment.block.pos = np.array(self.environment.block.getPos(factored_state["Block"][:2]))
        self.environment.block.vel = np.array(factored_state["Block"][2:4]).astype(int)
        # self.environment.block.losses = 0 # ensures that no weirdness happens since ball losses are not stored, though that might be something to keep in attribute...
        self.environment.gripper.pos = np.array(self.environment.gripper.getPos(factored_state["Gripper"][:2]))
        self.environment.gripper.vel = np.array(factored_state["Gripper"][2:4]).astype(int)
        self.environment.actions.attribute = factored_state["Action"][-1]
        self.environment.target.pos = np.array(self.environment.target.getPos(factored_state["Target"][:2]))
        self.environment.target.vel = np.array(factored_state["Target"][2:4]).astype(int)
        if not self.environment.pushgripper:
            self.environment.stick.pos = np.array(self.environment.stick.getPos(factored_state["Stick"][:2]))
        self.environment.render()