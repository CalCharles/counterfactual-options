# screen.py
import numpy as np
import cv2
import argparse
import sys
sys.path.insert(0, "/home/calcharles/research/contingency-options/")
from Environments.Pushing.objects import *
import imageio as imio
import os, copy
from Environments.environment_specification import RawEnvironment
from gym import spaces

class Pushing(RawEnvironment):
    def __init__(self, pushgripper=False, frameskip=3, reset_max=300):
        super().__init__()
        self.num_actions = 5
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.done = False
        self.reward = 0
        self.seed_counter = -1
        self.discrete_actions = True
        self.actions = Action(2)
        self.gripper = CartesianGripper(2)
        self.stick = Stick(2)
        # self.block = Sphere(2)
        self.block = Block(2)
        self.target = Target(2)
        self.pushgripper = pushgripper
        if pushgripper:
            self.gripper = CartesianPusher(2)
            self.objects = [self.actions, self.gripper, self.block, self.target]
        else:
            self.objects = [self.actions, self.gripper, self.stick, self.block, self.target]
        self.frameskip = frameskip
        self.render()
        self.reset()
        self.reset_counter = 0
        self.reset_max = reset_max
        self.reward = 0
        self.distance_reward = False
        self.name = "Pusher"

    def use_distance_reward(self):
        self.distance_reward = True

    def reset_object(self, o):
        newy = o.limits[0] + np.round((o.limits[2] - o.limits[0]) * np.random.rand())
        newx = o.limits[1] + np.round((o.limits[3] - o.limits[1]) * np.random.rand())
        # newy = newy - (newy % self.frameskip)
        # newx = newx - (newy % self.frameskip)
        o.updateBounding(np.array([newy, newx]))
        o.touched = False
        o.gripped = None


    def reset(self):
        for o in self.objects:
            self.reset_object(o)
        self.reset_counter = 0
        return self.get_state()

    def get_state(self, render=True):
        extracted_state = {**{obj.name: np.array(obj.getMidpoint().tolist() + obj.getVel().tolist() + [obj.getAttribute()]) for obj in self.objects}, **{'Reward': [self.reward], 'Done': [self.done]}}
        rawframe = None
        if render:
            rawframe = self.render()
        return {"raw_state": rawframe, "factored_state": extracted_state}

    def render(self):
        frame = np.zeros((84,84))
        for i, o in enumerate(self.objects):
            if o.isAABB:
                # frame[int(o.center[0] + np.random.randint(12) - 6), :] = 1.0
                # frame[:, int(o.center[1] + np.random.randint(12) - 6)] = 1.0
                # print(type(o), o.bb)
                by, ey = max(int(np.round(o.bb[0])), 0), min(int(np.round(o.bb[2])), 84)
                bx, ex = max(int(np.round(o.bb[1])), 0), min(int(np.round(o.bb[3])), 84)
                frame[by:ey, bx:ex] = 1.0 / (len(self.objects)-1) * i
                if o.isGripper:
                    # yd = int(np.round(o.bb[2])) - int(np.round(o.bb[0]))
                    # xd = int(np.round(o.bb[3])) - int(np.round(o.bb[1]))
                    yd = int(np.round(o.bb[2])) - int(np.round(o.bb[0]))
                    xd = int(np.round(o.bb[3])) - int(np.round(o.bb[1]))
                    frame[int(np.round(o.bb[0])):int(np.round(o.bb[2]))-int(np.round(yd * .3)), int(np.round(o.bb[1]))+int(np.round(xd * .3)):int(np.round(o.bb[3]))-int(np.round(xd * .3))] = 0.0
                    # print(int(np.round(o.bb[0])),int(np.round(o.bb[2]))-int(np.round(yd * .3)), int(np.round(o.bb[1]))+int(np.round(xd * .3)),int(np.round(o.bb[3]))-int(np.round(xd * .3)))
                if type(o) == Stick:
                    by, ey = max(int(np.round(o.bb[0])), 0), min(int(np.round(o.bb[2])), 84)
                    bx, ex = max(int(np.round(o.bb[1])) + 1, 0), min(int(np.round(o.bb[3])) - 1, 84)
                    # print(bx, ex, o.bb)
                    frame[by:ey, bx:ex] = .7

            elif o.isSphere:
                Y,X = np.ogrid[:84,:84]
                d = np.sqrt((X-o.pos[1])**2 + (Y-o.pos[0])**2)
                mask = d <= o.radius
                frame[mask] = 1.0 / (len(self.objects)-1) * i
            # print(1.0 / (len(self.objects)) * (i + 1))
        self.frame = frame
        return self.frame
    
    def toString(self, extracted_state):
        estring = ""
        for i, obj in enumerate(self.objects):
            estring += obj.name + ":" + " ".join(map(str, extracted_state[obj.name])) + "\t" # TODO: attributes are limited to single floats
        estring += "Reward:" + str(self.reward) + "\t"
        estring += "Done:" + str(int(self.done)) + "\t"
        return estring

    def clear_interactions(self):
        for o in self.objects:
            o.interaction_trace = list()

    def zero_velocities(self):
        for o in self.objects:
            o.vel = np.array([0] * o.dim)

    def step(self, action, render=True, frameskip=2):
        if self.reset_counter == 1:
            self.reward = 0

        self.clear_interactions()
        for i in range(self.frameskip):
            self.zero_velocities()
            # print("before", i, self.get_state()['factored_state'])
            self.actions.attribute = action
            self.gripper.applyAction(self.actions)
            for i in range(len(self.objects)):
                for j in range(len(self.objects)): # objects in order
                    if i != j: # no self-interactions yet
                        o1, o2 = self.objects[i], self.objects[j]
                        contact = o1.getContact(o2)
                        o1.actContact(contact, o2)
            for i in range(len(self.objects)):
                self.objects[i].move()
            # self.zero_velocities()
            if self.block.pos[0] < 0 or self.block.pos[0] > 84 or self.block.pos[1] < 0 or self.block.pos[1] > 84:
                self.reset_object(self.block)  
            # print("after", self.get_state()['factored_state'])
            extracted_state = {**{obj.name: np.array(obj.getMidpoint().tolist() + obj.getVel().tolist() + [obj.getAttribute()]) for obj in self.objects}, **{'Reward': [self.reward], 'Done': [self.done]}}
            rawframe = None
            if render:
                rawframe = self.render()
            self.done = False
            if self.distance_reward:
                # self.reward = -.01 + .005*float(min(self.gripper.center - self.block.center) <= 7) + .
                # self.reward = -.0001 * np.linalg.norm(self.block.center - self.target.center, 1) + -.01 * (1-int(self.block.moved))
                # self.reward = 1/self.reset_max*(.05-(.001 * np.linalg.norm(self.block.center - self.target.center, 1))) + 1/self.reset_max*(.1-(.002 * np.linalg.norm(self.gripper.center - self.block.center, 1)))
                # self.reward = 1/self.reset_max*(.5-(.01 * np.linalg.norm(self.block.center - self.target.center, 1))) + 1/self.reset_max*(2-(.02 * np.linalg.norm(self.gripper.center - self.block.center, 1))) + 1/self.reset_max * int(self.block.moved)
                self.reward = .2 * int(self.block.moved) # -(.0002 * np.linalg.norm(self.gripper.center - self.block.center, 1))
                # print(self.block.center, int(self.block.moved), self.reward)
            if self.target.touched:
                # self.reward = 1
                self.reward = 1
                self.reset()
                self.done = True
                break
            if self.reset_counter % self.reset_max == 0 and self.reset_counter > 0:
                self.reset()
                self.done = True
                break
        full_state = self.get_state()
        frame, extracted_state = full_state['raw_state'], full_state['factored_state']
        if len(self.save_path) != 0:
            if self.itr == 0:
                object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                object_dumps.close()
            self.write_objects(extracted_state, frame.astype(np.uint8))
        # print(self.reward)
        self.reset_counter += 1
        self.itr += 1
        return full_state, self.reward, self.done, {"reset_counter": self.reset_counter}

class RandomPolicy():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, screen):
        return np.random.randint(self.action_space)

class RandomRepeatPolicy():
    def __init__(self, action_space, repeat=3):
        self.action_space = action_space
        self.repeat = repeat
        self.cnt = 0
        self.last_action = 0

    def act(self, screen):
        if self.cnt == 0:
            self.last_action = np.random.randint(self.action_space)
            self.cnt = self.repeat
        self.cnt -= 1
        return self.last_action

def demonstrate(save_dir, num_frames, pushgripper=False):
    pushingDomain = Pushing(pushgripper)
    quit = False
    pushingDomain.set_save(0, save_dir, 0)

    for i in range(num_frames):
        frame = pushingDomain.render()
        cv2.imshow('frame',frame)
        action = 0
        key = cv2.waitKey(500)
        if key == ord('q'):
            quit = True
        elif key == ord('a'):
            action = 3
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 2
        elif key == ord('d'):
            action = 4
        pushingDomain.step(action)

def run(save_dir, num_frames, policy, pushgripper=False):
    # running with a random policy
    pushingDomain = Pushing(pushgripper, frameskip=3)
    pushingDomain.set_save(0, save_dir, -1, save_raw= True)

    for i in range(num_frames):
        action = policy.act(pushingDomain)
        pushingDomain.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('savedir',
                        help='base directory to save results')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='number of training iterations')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion')
    parser.add_argument('--pushgripper', action='store_true', default=False,
                        help='run the pushing gripper domain')
    args = parser.parse_args()
    if args.demonstrate:
        demonstrate(args.savedir, args.num_frames, args.pushgripper)
    else:
        # run(args.savedir, args.num_frames, RandomPolicy(5), args.pushgripper)
        run(args.savedir, args.num_frames, RandomRepeatPolicy(5), args.pushgripper)
