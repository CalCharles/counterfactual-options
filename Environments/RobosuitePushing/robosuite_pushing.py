import numpy as np
import gym
import os
from gym import spaces
import robosuite
from robosuite.controllers import load_controller_config
import imageio, tqdm
import copy
import cv2
from Environments.environment_specification import RawEnvironment
from collections import deque
import robosuite.utils.macros as macros
macros.SIMULATION_TIMESTEP = 0.02



# class PushingEnvironment(gym.Env):
#     def __init__(self, horizon, control_freq, renderable=False):
#         self.renderable = renderable
#         self.env = robosuite.make(
#             "Push",
#             robots=["Panda"],
#             controller_configs=load_controller_config(default_controller="OSC_POSE"),
#             has_renderer=False,
#             has_offscreen_renderer=renderable,
#             render_visual_mesh=renderable,
#             render_collision_mesh=False,
#             camera_names=["frontview"] if renderable else None,
#             control_freq=control_freq,
#             horizon=horizon,
#             use_object_obs=True,
#             use_camera_obs=renderable,
#         )

#         low, high = self.env.action_spec
#         self.action_space = spaces.Box(low=low[:2], high=high[:2])

#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[9])
#         self.curr_obs = None
#         self.step_num = None

#     def seed(self, seed=None):
#         if seed is not None:
#             np.random.seed(seed)
#             self.action_space.seed(seed)

#     def _get_flat_obs(self, obs):
#         return np.concatenate([
#             obs["robot0_eef_pos"],
#             obs["gripper_to_cube_pos"],
#             obs["cube_to_goal_pos"],
#         ])

#     def reset(self):
#         self.curr_obs = self.env.reset()
#         self.step_num = 0
#         return self._get_flat_obs(self.curr_obs)

#     def step(self, action):
#         next_obs, reward, done, info = self.env.step(np.concatenate([action, [0, 0, 0, 0]]))
#         return_obs = self._get_flat_obs(next_obs)
#         if self.renderable:
#             info["image"] = self.curr_obs["frontview_image"][::-1]
#             info["step"] = self.step_num
#             if done:
#                 info["final_image"] = next_obs["frontview_image"][::-1]
#         self.curr_obs = next_obs
#         self.step_num += 1
#         return return_obs, reward, done, info

#     def her(self, obs, obs_next):
#         """
#         Takes a list of observations (and next observations) from an entire episode and returns
#         the HER-modified version of the episode in the form of 4 lists: (obs, obs_next, reward, done).
#         """
#         obs = np.array(obs)
#         obs_next = np.array(obs_next)
#         fake_goal = obs_next[-1, :3] - obs_next[-1, 3:6]  # final cube position
#         obs[:, 6:] = (obs[:, :3] - obs[:, 3:6]) - fake_goal
#         obs_next[:, 6:] = (obs_next[:, :3] - obs_next[:, 3:6]) - fake_goal
#         reward = np.array([self.compute_reward(fake_goal, o[:3] - o[3:6], {}) for o in obs_next])
#         done = reward == 0
#         return obs, obs_next, reward, done

#     def compute_reward(self, achieved_goal, desired_goal, info):
#         reward = -1
#         if self.env.check_success(desired_goal, achieved_goal):
#             reward = 0
#         return reward

#     def render(self, mode="human"):
#         assert self.renderable
#         return self.curr_obs["frontview_image"][::-1]

class RoboPushingEnvironment(RawEnvironment):
    def __init__(self, control_freq=2, horizon=30, renderable=False):
        super().__init__()
        self.env = robosuite.make(
                "Push",
                robots=["Panda"],
                controller_configs=load_controller_config(default_controller="OSC_POSE"),
                has_renderer=False,
                has_offscreen_renderer=renderable,
                render_visual_mesh=renderable,
                render_collision_mesh=False,
                camera_names=["frontview"] if renderable else None,
                control_freq=control_freq,
                horizon=horizon,
                use_object_obs=True,
                use_camera_obs=renderable,
                hard_reset = False,
            )

        self.frameskip = control_freq

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low[:3], high=high[:3])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[9])
        self.num_actions = -1 # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "RobosuitePushing" # required for an environment 
        self.itr = 0 # this is used for saving, and is set externally
        self.timer = 0
        self.recycle = 0 # if we don't want to save all of the data
        self.save_path = "" # save dir also is set using set_save
        self.episode_rewards = deque(maxlen=10) # the episode rewards for the last 10 episodes
        self.reshape = (-1) # the shape of an observation
        self.discrete_actions = False
        self.renderable = False

        # should be set in subclass
        self.action_shape = (3,) # should be set in the environment, (1,) is for discrete action environments
        self.action = np.array([0,0,0])
        self.reward = 0
        self.done = False
        self.seed_counter = -1
        self.objects = ["Action", "Gripper", "Block", "Target", "Reward", "Done"]

        self.full_state = self.reset()
        self.frame = self.full_state['raw_state'] # the image generated by the environment

    def set_named_state(self, obs_dict):
        obs_dict['Action'], obs_dict['Gripper'], obs_dict['Block'], obs_dict['Target'] = self.action, obs_dict['robot0_eef_pos'], obs_dict['cube_pos'], obs_dict['goal_pos']# assign the appropriate values
        obs_dict['Reward'], obs_dict['Done'] = [self.reward], [self.done]

    def construct_full_state(self, factored_state, raw_state):
        obs = {'raw_state': raw_state, 'factored_state': factored_state}
        return obs

    def step(self, action):
        self.action = action
        next_obs, reward, done, info = self.env.step(np.concatenate([action, [0, 0, 0]]))
        self.reward, self.done = reward, done
        info["TimeLimit.truncated"] = done
        # print(list(next_obs.keys()))
        # if not self.done: # repeat the last state because the next state will end up belonging to the next episode
        self.set_named_state(next_obs)
        img = next_obs["frontview_image"][::-1] if self.renderable else None
        obs = self.construct_full_state(next_obs, img)
        obs = {'raw_state': img, 'factored_state': next_obs}
        # else:
        #     obs = self.full_state
        self.full_state = obs
        self.frame = self.full_state['raw_state']
        # cv2.imshow('state', next_obs["frontview_image"][::-1].astype(np.uint8))
        # cv2.waitKey(1)
        self.itr += 1
        self.timer += 1
        if len(self.save_path) != 0:
            if self.itr == 0:
                object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                object_dumps.close()
            self.write_objects(obs["factored_state"], next_obs["frontview_image"][::-1].astype(np.uint8) if self.renderable else None)
        if self.done:
            reward = self.reward
            self.reset()
            self.timer = 0
            self.done = True
            self.reward = reward
            # self.full_state['factored_state']['Done'] = [self.done]
            # self.full_state['factored_state']['Reward'] = [self.reward]
            # obs = self.full_state
            # print(self.done, obs)
        return obs, reward, done, info

    def get_state(self):
        return copy.deepcopy(self.full_state)

    def reset(self):
        obs = self.env.reset()
        self.set_named_state(obs)
        # print(list(obs.keys()))
        # print(obs['robot0_eef_pos'], obs['cube_pos'], obs['goal_pos'])
        # 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'frontview_image', 'cube_pos', 'gripper_to_cube_pos', 'goal_pos', 'gripper_to_goal_pos', 'cube_to_goal_pos', 'robot0_proprio-state', 'object-state', 'Action', 'Gripper', 'Block', 'Target', 'Reward', 'Done'
        return self.construct_full_state(obs, obs["frontview_image"][::-1] if self.renderable else None)

    def render(self):
        return self.env.render()

    def toString(self, extracted_state):
        estring = ""
        for i, obj in enumerate(self.objects):
            # print(extracted_state[obj])
            if obj not in ["Reward", "Done"]:
                estring += obj + ":" + " ".join(map(str, extracted_state[obj])) + "\t" # TODO: attributes are limited to single floats
            else:
                estring += obj + ":" + str(int(extracted_state[obj][0])) + "\t"
        # estring += "Reward:" + str(self.reward) + "\t"
        # estring += "Done:" + str(int(self.done)) + "\t"
        return estring

