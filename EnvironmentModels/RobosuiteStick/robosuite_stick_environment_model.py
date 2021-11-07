from EnvironmentModels.environment_model import EnvironmentModel
import robosuite.utils.macros as macros
macros.SIMULATION_TIMESTEP = 0.02

class RobosuiteStickEnvironmentModel(EnvironmentModel):
    def __init__(self, env):
        super().__init__(env)
        self.enumeration = {"Action": [0,1], "Gripper": [1,2], "Block": [2,3], 'Stick': [3,4], 'Target': [4,5],
                        'Done':[5,6], "Reward":[6,7], 'Grasped': [7,8]}
        self.object_names = ["Action", "Gripper", 'Stick', "Block", 'Target', 'Done', "Reward", 'Grasped']
        self.object_sizes = {"Action": 4, "Gripper": 4, "Block": 3, 'Stick': 3, 'Target': 3, 'Done': 1, "Reward": 1, 'Grasped': 1}
        self.object_num = {"Action": 1, "Gripper": 1, "Block": 1, 'Stick':1, 'Target': 1, 'Done': 1, "Reward": 1, 'Grasped': 1}
        self.state_size = sum([self.object_sizes[n] * self.object_num[n] for n in self.object_names])
        self.shapes_dict = {"state": [self.state_size], "next_state": [self.state_size], "state_diff": [self.state_size], "action": [4], "done": [1], "info": [1]}
        self.param_size = self.state_size
        self.set_indexes()

    # TODO: interaction traces not supported
    # def get_interaction_trace(self, name):
    #     trace = []
    #     for i in range(*self.enumeration[name]):
    #         # print(name, self.environment.objects[i].interaction_trace)
    #         trace.append(self.environment.objects[i].interaction_trace)
    #     return trace

    # def set_interaction_traces(self, factored_state):
    #     self.set_from_factored_state(factored_state)
    #     self.environment.step(factored_state["Action"][-1])
    #     self.set_from_factored_state(factored_state)
    def get_action(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state['Action']

    def get_info(self, full_state):
        factored_state = full_state['factored_state']
        return factored_state["Grasped"] 


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
