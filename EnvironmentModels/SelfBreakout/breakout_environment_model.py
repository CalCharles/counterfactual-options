import numpy as np
import os, cv2, time
from EnvironmentModels.environment_model import EnvironmentModel

class BreakoutEnvironmentModel(EnvironmentModel):
    def __init__(self, breakout_environment):
        super().__init__(breakout_environment)
        self.object_names = ["Action", "Paddle", "Ball", "Block", 'Done', "Reward"] # TODO: Reward missing from the objects
        self.object_sizes = {"Action": 5, "Paddle": 5, "Ball": 5, "Block": 5, 'Done': 1, "Reward": 1}
        self.object_num = {"Action": 1, "Paddle": 1, "Ball": 1, "Block": 100, 'Done': 1, "Reward": 1}
        self.state_size = sum([self.object_sizes[n] * self.object_num[n] for n in self.object_names])
        self.shapes_dict = {"state": [self.state_size], "next_state": [self.state_size], "state_diff": [self.state_size], "action": [1], "done": [1]}
        self.enumeration = {"Action": [0,1], "Paddle": [1,2], "Ball": [2,3], "Block": [3,103]}
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

    def flatten_factored_state(self, factored_state, instanced=False, names=None):
        if names is None:
            names = self.object_names
        if instanced:
            if type(factored_state) == list:
                flattened_state = list()
                for f in factored_state:
                    flat = list()
                    for n in names:
                        if self.object_num[n] > 1:
                            for i in range(self.object_num[n]):
                                flat += f[n+str(i)]
                    flattened_state += flat
                flattened_state = np.array(flattened_state)
            else:
                flattened_state = list()
                for n in names:
                    if self.object_num[n] > 1:
                        for i in range(self.object_num[n]):
                            flattened_state += factored_state[n+str(i)]
                    else:
                        flattened_state += factored_state[n]

                flattened_state = np.array(flattened_state)
        else:
            if type(factored_state) == list:
                flattened_state = np.array([np.concatenate([factored_state[i][f] for f in names], axis=1) for i in range(factored_state)])
            else:
                flattened_state = np.array(np.concatenate([factored_state[f] for f in names], axis=0))
        return flattened_state

    def unflatten_state(self, flattened_state, vec=False, instanced=False, names=None):
        if names is None:
            names = self.object_names
        def unflatten(flattened):
            at = 0
            factored = dict()
            for name in self.object_names:
                if instanced: #factor each object, even those of the same type 
                    for k in range(self.object_num[name]):
                        usename = name
                        if self.object_num[name] > 1:
                            usename = name+str(k)
                        if vec:
                            factored[name] = flattened[:, at:at+self.object_sizes[name]]
                        else: # a single state at a time
                            factored[name] = flattened[at:at+self.object_sizes[name]]
                        at += self.object_sizes[name]
                else: # factor each object, grouping objects of the same type
                    if vec:
                        factored[name] = flattened[:, at:at+(self.object_sizes[name]*self.object_num[name])]
                    else: # a single state at a time
                        factored[name] = flattened[at:at+(self.object_sizes[name]*self.object_num[name])]
                    at += (self.object_sizes[name]*self.object_num[name])
            return factored
        if len(flattened_state.shape) == 2:
            if vec:
                factored = unflatten(flattened_state)
            else:
                factored = []
                for i in range(flattened_state.shape[0]):
                    factored.append(unflatten(flattened_state[i]))
        else: # assumes state is a vector
            factored = unflatten(flattened_state)
        return factored

    def set_from_factored_state(self, factored_state, instanced = False, seed_counter=-1):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.environment.seed_counter = seed_counter
            self.environment.ball.reset_seed = seed_counter
        self.environment.ball.pos = np.array(self.environment.ball.getPos(factored_state["Ball"][:2]))
        self.environment.ball.vel = np.array(factored_state["Ball"][2:4]).astype(int)
        self.environment.ball.losses = 0 # ensures that no weirdness happens since ball losses are not stored, though that might be something to keep in attribute...
        self.environment.paddle.pos = np.array(self.environment.paddle.getPos(factored_state["Paddle"][:2]))
        self.environment.paddle.vel = np.array(factored_state["Paddle"][2:4]).astype(int)
        self.environment.actions.attribute = factored_state["Action"][-1]
        for i in range(5):
            for j in range(20):
                if instanced:
                    self.environment.blocks[i*20+j].attribute = float(factored_state["Block" + str(i * 20 + j)][-1])
                else:
                    self.environment.blocks[i*20+j].attribute = float(factored_state["Block"][(i*20+j)*5+4])
        self.environment.render_frame()