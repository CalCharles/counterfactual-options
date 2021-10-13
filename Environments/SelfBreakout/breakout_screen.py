# Screen
import sys, cv2
import numpy as np
from Environments.SelfBreakout.breakout_objects import *
# from breakout_objects import *
import imageio as imio
import os, copy
from Environments.environment_specification import RawEnvironment
from gym import spaces

def adjacent(i,j):
    return [(i-1,j-1), (i, j-1), (i, j+1), (i-1, j), (i-1,j+1),
            (i, j-2), (i-1, j-2), (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2), (i-1, j+2), (i-2, j+2)]

ball_vels = [np.array([-1.,-1.]).astype(np.int), np.array([-2.,-1.]).astype(np.int), np.array([-2.,1.]).astype(np.int), np.array([-1.,1.]).astype(np.int)]

class Screen(RawEnvironment):
    def __init__(self, frameskip = 1, drop_stopping=True, target_mode = False, angle_mode=False):
        super(Screen, self).__init__()
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

        self.drop_stopping = drop_stopping
        self.target_mode = target_mode
        if self.target_mode:
            self.num_blocks = 1
        else:
            self.num_blocks = 100
        self.angle_mode = angle_mode
        self.used_angle = False
        self.done = False
        self.reward = 0
        self.seed_counter = -1
        self.exposed_blocks = list()
        self.reset()
        self.average_points_per_life = 0
        self.itr = 0
        self.save_path = ""
        self.recycle = -1
        self.frameskip = frameskip
        self.total_score = 0
        self.discrete_actions = True
        self.needs_ball_reset = False

    def ball_reset(self):
        self.ball.pos = [41, np.random.randint(20, 52)]
        self.ball.vel = np.array([np.random.randint(1,2), np.random.choice([-1,1])])

    def reset(self):
        if self.seed_counter > 0:
            self.seed_counter += 1
            np.random.seed(self.seed_counter)
        vel= np.array([np.random.randint(1,2), np.random.choice([-1,1])])
        # self.ball = Ball(np.array([52, np.random.randint(14, 70)]), 1, vel)
        self.ball = Ball(np.array([41, np.random.randint(20, 52)]), 1, vel, top_reset=self.target_mode)
        self.ball.reset_pos(self.target_mode)
        self.ball.losses = 0
        self.paddle = Paddle(np.array([71, 84//2]), 1, np.zeros((2,), dtype = np.int64))
        self.actions = Action(np.zeros((2,), dtype = np.int64), 0)
        self.blocks = []
        self.blocks2D = list()
        if self.target_mode:
            self.blocks = [Block(np.array([17,15 + np.random.randint(4) * 15]), 1, -1, (0,0), size = 6)]
            self.blocks2D = [self.blocks]
        else:
            for i in range(5):
                block2D_row = list()
                for j in range(20):
                    block = Block(np.array([22 + i * 2,12 + j * 3]), 1, i * 20 + j, (i,j))
                    self.blocks.append(block)
                    # self.blocks.append(Block(np.array([32 + i * 2,12 + j * 3]), 1, i * 20 + j))
                    block2D_row.append(block)
                self.blocks2D.append(block2D_row)
            self.blocks2D = np.array(self.blocks2D)
        self.walls = []
        # Topwall
        self.walls.append(Wall(np.array([4, 4]), 1, "Top"))
        self.walls.append(Wall(np.array([80, 4]), 1, "Bottom"))
        self.walls.append(Wall(np.array([0, 8]), 1, "LeftSide"))
        self.walls.append(Wall(np.array([0, 72]), 1, "RightSide"))
        self.animate = [self.paddle, self.ball]
        self.objects = [self.actions, self.paddle, self.ball] + self.blocks + self.walls
        self.obj_rec = [[] for i in range(len(self.objects))]
        self.counter = 0
        self.points = 0
        self.seed_counter += 1
        self.exposed_blocks = {self.blocks[i].index2D: self.blocks[i] for i in range(len(self.blocks)) if self.blocks[i].pos[0] >= 22 + 4 * 2 - 1}
        self.render_frame()
        self.needs_ball_reset = False
        return self.get_state()

    def render(self):
        return self.render_frame()

    def render_frame(self):
        self.frame = np.zeros((84,84), dtype = 'uint8')
        for block in self.blocks:
            if block.attribute == 1:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .5 * 255
        for wall in self.walls:
            self.frame[wall.pos[0]:wall.pos[0]+wall.height, wall.pos[1]:wall.pos[1]+wall.width] = .3 * 255
        ball, paddle = self.ball, self.paddle
        self.frame[ball.pos[0]:ball.pos[0]+ball.height, ball.pos[1]:ball.pos[1]+ball.width] = 1 * 255
        self.frame[paddle.pos[0]:paddle.pos[0]+paddle.height, paddle.pos[1]:paddle.pos[1]+paddle.width] = .75 * 255
        return self.frame

    def get_num_points(self):
        total = 0
        for block in self.blocks:
            if block.attribute == 0:
                total += 1
        # print(total)
        return total

    def extracted_state_dict(self):
        return {obj.name: obj.getMidpoint() for obj in self.objects}

    def get_state(self):
        self.render_frame()
        return {"raw_state": self.frame, "factored_state": {**{obj.name: obj.getMidpoint() + obj.vel.tolist() + [obj.getAttribute()] for obj in self.objects}, **{'Done': [self.done], 'Reward': [self.reward]}}}

    def clear_interactions(self):
        self.ball.clear_hits()

        for o in self.objects:
            o.interaction_trace = list()

    def toString(self, extracted_state):
        estring = "ITR:" + str(self.itr) + "\t"
        for i, obj in enumerate(self.objects):
            estring += obj.name + ":" + " ".join(map(str, extracted_state[obj.name])) + "\t" # TODO: attributes are limited to single floats
        estring += "Reward:" + str(self.reward) + "\t"
        estring += "Done:" + str(int(self.done)) + "\t"
        return estring


    def step(self, action, render=True, angle=-1): # TODO: remove render as an input variable
        self.done = False
        last_loss = self.ball.losses
        self.reward = 0
        hit = False
        self.clear_interactions()
        self.used_angle = False
        needs_reset = False
        if self.needs_ball_reset: self.ball_reset()
        self.needs_ball_reset = False
        for i in range(self.frameskip):
            self.actions.take_action(action)
            for obj1 in self.animate:
                for obj2 in self.objects:
                    if obj1.name == obj2.name:# or (obj1.name == "Ball" and obj2.name == "Paddle"):
                        continue
                    else:
                        preattr = obj2.attribute
                        obj1.interact(obj2)
                        if preattr != obj2.attribute:
                            self.reward += 1
                            self.total_score += 1
                            hit = True
                            if obj2.name.find("Block") != -1 and not self.target_mode:
                                if obj2.index2D in self.exposed_blocks:
                                    self.exposed_blocks.pop(obj2.index2D)
                                for i,j in adjacent(*obj2.index2D):
                                    if 0 <= i < 5 and 0 <= j < 20 and self.blocks2D[i,j].attribute == 1:
                                        self.exposed_blocks[i,j] = self.blocks2D[i,j]
            # self.paddle.move() # ensure the ball moves after the paddle to ease counterfactual
            # self.ball.interact(self.paddle)
            # self.ball.move()
            # print(self.ball.pos[0])
            if (68 <= self.ball.pos[0] <= 69 and self.ball.vel[0] == 1) or (67 <= self.ball.pos[0] <= 69 and self.ball.vel[0] == 2):
                self.used_angle= True
            for ani_obj in self.animate:
                ani_obj.move()
            pre_stop = (self.ball.pos[0] == 77 and self.ball.vel[0] == 2) or (self.ball.pos[0] == 78 and self.ball.vel[0] == 1) or (self.ball.pos[0] == 78 and self.ball.vel[0] == 2)
            if pre_stop and (self.drop_stopping or self.target_mode):
                self.reward += -1 # negative reward for dropping the ball since done is not triggered
                self.done = True
                self.needs_ball_reset = True
                print("dropped", np.array(self.ball.pos.tolist() + self.ball.vel.tolist() + self.paddle.pos.tolist()))
                break
            if (self.ball.losses == 4 and pre_stop) or (self.target_mode and (self.ball.top_wall or self.ball.bottom_wall or self.ball.block)):
                self.average_points_per_life = self.total_score / 5.0
                self.done = True
                self.reward += -1 * int(not self.ball.block)
                self.episode_rewards.append(self.total_score)
                self.total_score = 0
                needs_reset = True
                break
            if hit:
                hit = False
                if self.get_num_points() == len(self.blocks):
                    needs_reset = True
                    break
            if render:
                self.render_frame()
        self.itr += 1
        full_state = self.get_state()
        frame, extracted_state = full_state['raw_state'], full_state['factored_state']
        lives = 5-self.ball.losses
        if len(self.save_path) != 0:
            if self.itr == 0:
                object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                object_dumps.close()
            self.write_objects(extracted_state, frame.astype(np.uint8))
        if needs_reset: self.reset()
        return {"raw_state": frame, "factored_state": extracted_state}, self.reward, self.done, {"lives": lives, "TimeLimit.truncated": False}

    def run(self, policy, iterations = 10000, render=False, save_path = "runs/", save_raw = True, duplicate_actions=1, angle_mode=False):
        self.set_save(0, save_path, -1, save_raw)
        self.angle_mode=angle_mode
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        angle = policy.get_angle(self)
        for self.itr in range(iterations):
            action = policy.act(self, angle=angle)
            if action == -1: # signal to quit
                break
            if self.angle_mode:
                angle = policy.get_angle(self)
                self.step(action, angle=angle)
            else:
                self.step(action)

class Policy():
    def act(self, screen):
        print ("not implemented")

    def get_angle(self, screen):
        return 0

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, screen, angle=0):
        return np.random.randint(self.action_space)

class RandomConsistentPolicy(Policy):
    def __init__(self, action_space, change_prob):
        self.action_space = action_space
        self.change_prob = change_prob
        self.current_action = np.random.randint(self.action_space)

    def act(self, screen, angle=0):
        if np.random.rand() < self.change_prob:
            self.current_action = np.random.randint(self.action_space)
        return self.current_action

class RotatePolicy(Policy):
    def __init__(self, action_space, hold_count):
        self.action_space = action_space
        self.hold_count = hold_count
        self.current_action = 0
        self.current_count = 0

    def act(self, screen, angle=0):
        self.current_count += 1
        if self.current_count >= self.hold_count:
            self.current_action = np.random.randint(self.action_space)
            # self.current_action = (self.current_action+1) % self.action_space
            self.current_count = 0
        return self.current_action

class BouncePolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.internal_screen = Screen(angle_mode = False)
        self.objective_location = 84//2
        self.last_paddlehits = -1

    def act(self, screen, angle=0):
        # print(screen.ball.paddlehits, screen.ball.losses, self.last_paddlehits)
        if screen.ball.paddlehits + screen.ball.losses > self.last_paddlehits or (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
            if (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
                self.last_paddlehits = 0
            self.internal_screen = copy.deepcopy(screen)
            self.internal_screen.angle_mode = False
            self.internal_screen.save_path = ""
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)

            while self.internal_screen.ball.pos[0] < 71 and not self.internal_screen.done:
                # print("running internal")
                self.internal_screen.step(0)
            # print("completed")
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            self.objective_location = self.internal_screen.ball.pos[1] + np.random.choice([-1, 0, 1])
            self.last_paddlehits += 1
        if self.objective_location > screen.paddle.getMidpoint()[1]:
            return 3
        elif self.objective_location < screen.paddle.getMidpoint()[1]:
            return 2
        else:
            return 0

class AnglePolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.internal_screen = Screen(angle_mode = False)
        self.objective_location = 84//2
        self.last_paddlehits = -1
        self.counter = 0

    def reset_screen(self, screen):
        self.internal_screen = copy.deepcopy(screen)
        self.internal_screen.angle_mode = False
        self.internal_screen.save_path = ""

    def pick_action(self, objective_location, xpoint):
        if objective_location > xpoint:
            return 3
        elif objective_location < xpoint:
            return 2
        else:
            return 0

    def get_angle(self, screen):
        # frame = screen.render_frame()
        # cv2.imshow('frame',frame)
        # key = cv2.waitKey(30)
        return self.counter


    def act(self, screen, angle=0):
        # print(screen.ball.paddlehits, screen.ball.losses, self.last_paddlehits)
        if screen.used_angle:
            # print(self.counter)
            self.counter = np.random.randint(4)
        if screen.ball.vel[0] > 0 and 46 <= screen.ball.pos[0] <= 47 or screen.ball.vel[0] < 0 and 67 <= screen.ball.pos[0] <= 68:
            if (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
                self.last_paddlehits = 0
            self.reset_screen(screen)
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)

            while self.internal_screen.ball.pos[0] < 69 and not self.internal_screen.done:
                # print("running internal")
                self.internal_screen.step(0)
            # print("completed")
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            base_location = self.internal_screen.ball.pos[1]
            sv = self.internal_screen.ball.vel[1]
            if angle == 0:
                self.objective_location = base_location + sv * 1
            elif angle == 1:
                self.objective_location = base_location - 2 + sv * 1
            elif angle == 2:
                self.objective_location = base_location - 4 + sv * 1
            elif angle == 3:
                self.objective_location = base_location - 6 + sv * 1
            self.objective_location += self.objective_location % 2
        return self.pick_action(self.objective_location, screen.paddle.pos[1])

def DemonstratorPolicy(Policy):
    def act(self, screen, angle=0):
        action = 0
        frame = screen.render_frame()
        cv2.imshow('frame',frame)
        key = cv2.waitKey(500)
        if key == ord('q'):
            action = -1
        elif key == ord('a'):
            action = 2
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 0
        elif key == ord('d'):
            action = 3
        return action


def demonstrate(save_dir, num_frames):
    domain = Screen()
    quit = False
    domain.set_save(0, save_dir, 0, True)
    for i in range(num_frames):
        frame = domain.render_frame()
        cv2.imshow('frame',frame)
        action = 0
        key = cv2.waitKey(500)
        if key == ord('q'):
            quit = True
        elif key == ord('a'):
            action = 2
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 0
        elif key == ord('d'):
            action = 3
        domain.step(action)
        if quit:
            break


def abbreviate_obj_dump_file(pth, new_path, get_last=-1):
    total_len = 0
    if get_last > 0:
        for line in open(os.path.join(pth,  'object_dumps.txt'), 'r'):
            total_len += 1
    current_len = 0
    new_file = open(os.path.join(new_path, 'object_dumps.txt'), 'w')
    for line in open(os.path.join(pth,  'object_dumps.txt'), 'r'):
        current_len += 1
        if current_len< total_len-get_last:
            continue
        new_file.write(line)
    new_file.close()

def get_action_from_dump(obj_dumps):
    return int(obj_dumps["Action"][1])

def get_individual_data(name, obj_dumps, pos_val_hash=3):
    '''
    gets data for a particular object, gets everything in obj_dumps
    pos_val_hash gets either position (1), value (2), full position and value (3)
    '''
    data = []
    for time_dict in obj_dumps:
        if pos_val_hash == 1:
            data.append(time_dict[name][0])
        elif pos_val_hash == 2:
            data.append(time_dict[name][1])
        elif pos_val_hash == 3:
            data.append(list(time_dict[name][0]) + list(time_dict[name][1]))
        else:
            data.append(time_dict[name])
    return data

def hot_actions(action_data):
    for i in range(len(action_data)):
        hot = np.zeros(4)
        hot[int(action_data[i])] = 1
        action_data[i] = hot.tolist()
    return action_data


if __name__ == '__main__':
    screen = Screen()
    # policy = RandomPolicy(4)
    policy = RotatePolicy(4, 9)
    # policy = BouncePolicy(4)
    screen.run(policy, render=True, iterations = 1000, duplicate_actions=1, save_path=sys.argv[1])
    # demonstrate(sys.argv[1], 1000)
