import numpy as np
import os, cv2, time
from EnvironmentModels.environment_model import EnvironmentModel

class BreakoutEnvironmentModel(EnvironmentModel):
	def __init__(self, breakout_environment):
		super().__init__(breakout_environment)
		self.object_names = ["Action", "Paddle", "Ball", "Block"] # TODO: Done and Reward missing from the objects
		self.object_sizes = {"Action": 5, "Paddle": 5, "Ball": 5, "Block": 5}
		self.object_num = {"Action": 1, "Paddle": 1, "Ball": 1, "Block": 100}

	def get_factored_state(self):
		factored_state = {o.name: o.pos.tolist() + o.vel.tolist() + [o.attribute] for o in self.environment.objects}
		return factored_state

	def flatten_factored_state(self, factored_state):
		if type(factored_state) == list:
			flattened_state = np.array([np.sum([factored_state[i][f] for f in self.object_names], axis=1) for i in range(factored_state)])
		else:
			flattened_state = np.array(sum([factored_state[f] for f in self.object_names]))
		return flattened_state

	def unflatten_state(self, flattened_state, vec=False, typed=False):
		def unflatten(flattened):
			at = 0
			factored = dict()
			for name in range(len(self.object_names)):
				if typed: #factor each object, even those of the same type 
					for k in range(self.object_num[name]):
						if vec:
							factored[name+str(k)] = flattened[:, at:at+self.object_sizes[name]]
						else: # a single state at a time
							factored[name+str(k)] = flattened[at:at+self.object_sizes[name]]
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
			factored = unflatten_single(flattened_state)
		return factored

	def set_from_factored_state(self, factored_state):
		'''
		TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
		'''
        self.breakout_environment.ball.pos = np.array(factored_state["Ball"][:2])
        self.breakout_environment.ball.vel = np.array(factored_state["Ball"][2:4])
        self.breakout_environment.paddle.pos = np.array(factored_state["Paddle"][:2])
        self.breakout_environment.paddle.vel = np.array(factored_state["Paddle"][2:4])
        self.breakout_environment.actions.attribute = factored_state["Action"][-1]
        for i in range(5):
            for j in range(20):
                self.blocks[i*20+j].attribute = factored_state["Block" + str(i * 20 + j)][-1]
        self.render_frame()