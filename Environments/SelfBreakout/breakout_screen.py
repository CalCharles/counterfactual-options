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

# default settings for normal variants, args in order: 
# target_mode (1)/edges(2)/center(3), scatter (4), num rows, num_columns, no_breakout (value for hit_reset), negative mode, bounce_count
breakout_variants = {"default": (0,5, 20, -1, "", 0,0, 0,-10),
                     "row":  (0,1,10,-1,"", 0,0, 0,-10),
                     "small": (0,2,10,-1,"", 0,0, 0,-10), 
                    "row_nobreak": (0,1,10,10,"", 0,0, 0,-1), 
                    "small_nobreak": (0,2,10,15,"", 0,0, 0,-1),
                    "full_nobreak": (0,5,20,115,"", 0,0, 0,-1),
                    "big_block": (1,1,1,-1,"",0,0, 0,-10),
                    "single_block": (1,1,1,-1,"",-1,0, 0,-10),
                    "negative_split_full": (0,5,20,75,"side",0,0, 0,-20),
                    "negative_split_small": (0,2,10,15,"side",0,0, 0,-20),
                    "negative_split_row": (0,1,10,5,"side",0,0, 0,-20),
                    "negative_center_full": (0,5,20,75,"center",0,0, 0,-20),
                    "negative_center_small": (0,2,10,15,"center",0,0, 0,-10),
                    "negative_center_row": (0,1,10,10,"center",0,0, 0,-10),
                    "negative_edge_full": (0,5,20,75,"edge",0,0, 0,-10),
                    "negative_edge_small": (0,2,10,15,"edge",0,0, 0,-10),
                    "negative_edge_row": (0,1,10,10,"edge",0,0, 0,-10),
                    "negative_checker_row": (0,1,10,10,"checker",0,0, 0,-10),
                    "negative_rand_row": (0,1,10,5,"rand",0,0, 0, -10),
                    "negative_double": (1,1,1,-1,"rand",-1,0, 0, -10),
                    "negative_multi": (1,1,1,-1,"rand",-1,0, 0, -10),
                    "negative_top_full": (0,5,20,40,"top",0,0,0, -120),
                    "negative_top_small": (0,2,10,7,"top",0,0,0, -30),
                    "breakout_priority_small": (0,2,10,-1,"",-2,0, 1, -30),
                    "breakout_priority_medium": (0,3,10,-1,"",-2,0, 5, -75),
                    "breakout_priority_large": (0,4,15,-1,"",-1,0, 20, -100),
                    "breakout_priority_full": (0,5,20,-1,"",-2,0, 20, -120),
                    "edges_full": (2,5,20,-1,"",-1,0, 20, -120),
                    "edges_small": (2,2,10,-1,"",-1,0, 1, -30),
                    "center_small": (3,2,10,-1,"",-1,0, 1,-30),
                    "center_medium": (3,3,15,-1,"",-1,0, 5,-75),
                    "center_large": (3,4,15,-1,"",-1, 0,20,-100),
                    "center_full": (3,5,20,-1,"",-2, 0,20,-120),
                    "harden_single": (4,3,10,-1,"",-1,10,0,-10),
                    "proximity": (0,4,15,60,"", 0,0, 0,-10)}

class Screen(RawEnvironment):
    def __init__(self, frameskip = 1, drop_stopping=True, target_mode = False, angle_mode=False,
                num_rows = 5, num_columns = 20, max_block_height=4, no_breakout=False,
                negative_mode="", random_exist=-1, hit_reset=-1, breakout_variant="", bounce_cost=0, bounce_reset=0, sampler=None):
        super(Screen, self).__init__()
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.name = "Breakout"
        self.variant = breakout_variant

        bounce_reset = -1
        bounce_cost = 0
        bounce_reset = 0
        self.completion_reward = 0
        self.timeout_penalty = -1
        if len(breakout_variant) > 0: # overrides any existing arguments
            var_form, num_rows, num_columns, hit_reset, negative_mode, bounce_cost, bounce_reset, completion_reward, timeout_penalty = breakout_variants[breakout_variant]
            target_mode = (var_form == 1)
            if var_form == 2:
                negative_mode = "hardedge"
            if var_form == 3:
                negative_mode = "hardcenter"
            if var_form == 4:
                negative_mode = "hardscatter"
            no_breakout = hit_reset > 0
            self.timeout_penalty = timeout_penalty
            # self.completion_reward = completion_reward

        self.sampler = sampler
        self.full_stopping = bounce_cost < 0 or hit_reset > 0 or bounce_reset > 0
        self.assessment_stat = 0 # a measure of performance specific to the variant
        self.drop_stopping = drop_stopping
        self.target_mode = target_mode
        self.bounce_reset = bounce_reset
        self.bounce_cost = bounce_cost
        self.default_reward = 1 if self.bounce_reset == 0 else 1
        self.num_rows = num_rows # must be a factor of 10
        self.num_columns = num_columns # must be a factor of 60
        self.max_block_height = max_block_height
        if self.target_mode:
            self.num_blocks = 1
        elif negative_mode == "hardscatter":
            self.num_blocks = 6 # hardcoded for now
        else:
            self.num_blocks = num_rows * num_columns
        self.angle_mode = angle_mode
        self.used_angle = False
        self.done = False
        self.reward = 0
        self.total_episode_reward = 0
        self.seed_counter = -1
        self.exposed_blocks = list()
        self.average_points_per_life = 0
        self.itr = 0
        self.save_path = ""
        self.recycle = -1
        self.frameskip = frameskip
        self.total_score = 0
        self.discrete_actions = True
        self.needs_ball_reset = False
        self.block_height = min(self.max_block_height, 10 // self.num_rows)
        self.block_width = 60 // self.num_columns
        self.no_breakout = no_breakout
        self.hit_reset = hit_reset # number of block hits before resetting
        self.hit_counter = 0
        self.bounce_reset = bounce_reset # number of paddle bounces before resetting
        self.bounce_counter = 0
        self.negative_mode = negative_mode
        self.hard_mode = self.negative_mode[:4] == "hard"
        self.choices = list()
        self.random_exist = random_exist
        self.use_2D = not self.target_mode and not self.random_exist
        self.resetted = True
        self.since_last_bounce = 0
        self.reset()
        self.num_remove = self.get_num_remove()

    def assign_assessment_stat(self):
        if self.dropped and self.variant != "proximity":
            self.assessment_stat += -1000
        elif self.dropped:
            self.assessment_stat = -1000            
        elif self.variant == "big_block":
            if self.ball.block: self.assessment_stat = 1
        elif self.variant == "default":
            if self.ball.block: self.assessment_stat += 1
        elif self.variant == "negative_rand_row":
            if self.reward > 0: self.assessment_stat += 1
        elif self.variant == "center_large":
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "breakout_priority_large":
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "harden_single": 
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "single_block": 
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "proximity":
            if self.ball.block:
                # print(self.assessment_stat, self.sampler.param, self.ball.block_id.getMidpoint(), np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1))
                if type(self.assessment_stat) == tuple: self.assessment_stat = (self.assessment_stat[0] + 1, self.assessment_stat[1] + np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1))
                print("hit at l1", np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1), self.ball.block_id.getMidpoint(), self.sampler.param[:2])
                # self.assessment_stat = np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1)
            if self.done:
                if type(self.assessment_stat) == tuple:
                    self.assessment_stat = self.assessment_stat[1] / self.assessment_stat[0]

    def ball_reset(self):
        self.ball.pos = [41, np.random.randint(20, 52)]
        # self.ball.pos = [np.random.randint(38, 45), np.random.randint(14, 70)]
        self.ball.vel = np.array([np.random.randint(1,2), np.random.choice([-1,1])])

    def assign_attribute(self, nmode, block, atrv):
        if nmode == "side":
            if block.pos[1] < 42:
                block.attribute = atrv
        elif nmode == "top":
            if block.pos[0] < 22 + self.block_height * self.num_rows / 2:
                block.attribute = atrv
        elif nmode == "edge":
            if block.pos[1] < 28 or block.pos[1] > 56:
                block.attribute = atrv
        elif nmode == "center":
            if 28 < block.pos[1] < 56:
                block.attribute = atrv
        elif nmode == "checker":
            block.attribute = -1 + (int(block.name[5:]) % 2) * 2


    def assign_attributes(self):
        atrv = -1
        nmode = self.negative_mode
        if self.negative_mode[:4] == "zero":
            atrv = 0
            nmode = self.negative_mode[4:]
        if self.negative_mode[:4] == "hard":
            atrv = -1
            nmode = self.negative_mode[4:]
        if nmode == "scatter":
            newblocks = list()
            pos = list(range(len(self.blocks)))
            self.target = np.random.choice(pos, size=1, replace=False)[0]
            pos.pop(self.target)
            self.choices = np.random.choice(pos, size=5, replace=False) # Hardcoded at the moment
            for choice in self.choices:
                self.blocks[choice].attribute = atrv
                newblocks.append(self.blocks[choice])
            self.blocks[self.target].attribute = 1
            newblocks.append(self.blocks[self.target])
            for i, block in enumerate(newblocks):
                block.name = "Block" + str(i)
            self.blocks = newblocks
        elif nmode == "rand":
            self.choices = np.random.choice(list(range(len(self.blocks))), size=len(self.blocks) // 2, replace=False)
            for choice in self.choices:
                self.blocks[choice].attribute = atrv
        else:
            for block in self.blocks:
                self.assign_attribute(nmode, block, atrv)


    def reset(self):
        self.assessment_stat = 0 if self.variant != "proximity" else (0,0)
        if self.seed_counter > 0:
            self.seed_counter += 1
            np.random.seed(self.seed_counter)
        vel= np.array([np.random.randint(1,2), np.random.choice([-1,1])])
        # self.ball = Ball(np.array([52, np.random.randint(14, 70)]), 1, vel)
        self.ball = Ball(np.array([np.random.randint(38, 45), np.random.randint(14, 70)]), 1, vel, top_reset=self.target_mode and self.bounce_cost == 0)
        self.ball.hard_mode = self.hard_mode
        self.ball.reset_pos(self.target_mode)
        self.ball.losses = 0
        self.paddle = Paddle(np.array([71, 84//2]), 1, np.zeros((2,), dtype = np.int64))
        self.actions = Action(np.zeros((2,), dtype = np.int64), 0)
        self.blocks = []
        self.blocks2D = list()
        if self.target_mode:
            if self.bounce_cost != 0: # target mode with small blocks
                pos_block = Block(np.array([int(17 + np.random.rand() * 20),int(15 + np.random.rand() * 51)]), 1, -1, (0,0), size = 2)
                self.block_width = pos_block.width
                self.block_height = pos_block.height
                self.blocks = [pos_block]
                if len(self.negative_mode) > 0:
                    while True:
                        neg_pos = np.array([int(17 + np.random.rand() * 20),int(15 + np.random.rand() * 51)])
                        if abs(neg_pos[0] - pos_block.pos[0]) > 4 or abs(neg_pos[1] - pos_block.pos[1]) > 6:
                            break
                    neg_block = Block(neg_pos, -1, -1, (0,0), size = 2)
                    self.blocks.append(neg_block)
                self.blocks2D = [self.blocks]
            else:
                self.blocks = [Block(np.array([17,15 + np.random.randint(4) * 15]), 1, -1, (0,0), size = 6)]
                self.blocks2D = [self.blocks]
        else:
            blockheight = min(self.max_block_height, 10 // self.num_rows)
            blockwidth = 60 // self.num_columns
            for i in range(self.num_rows):
                block2D_row = list()
                for j in range(self.num_columns):
                    block = Block(np.array([22 + i * blockheight,12 + j * blockwidth]), 1, i * self.num_columns + j, (i,j), width=blockwidth, height=blockheight)
                    self.blocks.append(block)
                    # self.blocks.append(Block(np.array([32 + i * 2,12 + j * 3]), 1, i * 20 + j))
                    block2D_row.append(block)
                self.blocks2D.append(block2D_row)
            self.blocks2D = np.array(self.blocks2D)
            if self.random_exist > 0:
                choices = np.random.choice(list(range(len(self.blocks))), replace=False, size=self.random_exist)
                new_blocks = list()
                for i, choice in enumerate(choices):
                    newb = self.blocks[choice]
                    newb.name = "Block" + str(i)
                    new_blocks.append(newb)
                self.blocks = new_blocks
        self.assign_attributes()
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
        self.hit_counter = 0
        self.bounce_counter = 0
        self.resetted = True
        self.since_last_bounce = 0
        self.total_episode_reward = 0
        return self.get_state()

    def render(self):
        return self.render_frame()

    def render_frame(self):
        self.frame = np.zeros((84,84), dtype = 'uint8')
        for block in self.blocks:
            if block.attribute != 0:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .5 * 255
            if block.attribute == -1:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .2 * 255
            if block.attribute == 2:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .8 * 255
        for wall in self.walls:
            self.frame[wall.pos[0]:wall.pos[0]+wall.height, wall.pos[1]:wall.pos[1]+wall.width] = .3 * 255
        ball, paddle = self.ball, self.paddle
        self.frame[ball.pos[0]:ball.pos[0]+ball.height, ball.pos[1]:ball.pos[1]+ball.width] = 1 * 255
        self.frame[paddle.pos[0]:paddle.pos[0]+paddle.height, paddle.pos[1]:paddle.pos[1]+paddle.width] = .75 * 255
        return self.frame

    def get_num_remove(self):
        total = 0
        for block in self.blocks:
            if block.attribute == 1:
                total += 1
        # print(total)
        return total

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

    def reset_blocks(self):
        nmode = self.negative_mode
        atrv = -1
        if self.negative_mode[:4] == "zero":
            atrv = 0
            nmode = self.negative_mode[4:]
        if self.negative_mode[:4] == "hard":
            atrv = 2 
            nmode = self.negative_mode[4:]
        if nmode == "rand":
            for block in self.blocks:
                if block.pos[0] == 22:
                    block.attribute = 1
            for choice in self.choices:
                if self.blocks[choice].pos[0] == 22:
                    self.blocks[choice].attribute = atrv
        else:
            for block in self.blocks:
                if block.pos[0] == 22:
                    block.attribute = 1
                    self.assign_attribute(nmode, block, atrv)


    def step(self, action, render=True, angle=-1): # TODO: remove render as an input variable
        self.done = False
        last_loss = self.ball.losses
        self.reward = 0
        hit = False
        self.clear_interactions()
        self.used_angle = False
        needs_reset = False
        if self.needs_ball_reset: self.ball_reset()
        if self.no_breakout: self.reset_blocks()
        self.needs_ball_reset = False
        self.resetted = False
        self.truncate = False
        self.dropped = False
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
                            if self.variant == "proximity":
                                dist = np.linalg.norm(np.array(obj1.getMidpoint()) - np.array(obj2.getMidpoint()), ord=1)
                                self.reward += (np.exp(-dist/10) - .2) * 2
                            else:
                                self.reward += preattr * self.default_reward
                                self.total_score += preattr * self.default_reward
                            hit = True
                            if obj2.name.find("Block") != -1 and not self.target_mode and self.use_2D:
                                if obj2.index2D in self.exposed_blocks:
                                    self.exposed_blocks.pop(obj2.index2D)
                                for i,j in adjacent(*obj2.index2D):
                                    if 0 <= i < self.num_rows and 0 <= j < self.num_columns and self.blocks2D[i,j].attribute == 1:
                                        self.exposed_blocks[i,j] = self.blocks2D[i,j]
            # self.paddle.move() # ensure the ball moves after the paddle to ease counterfactual
            # self.ball.interact(self.paddle)
            # self.ball.move()
            # print(self.ball.pos[0])
            if self.ball.paddle and self.bounce_cost < 0:
                self.reward += self.bounce_cost # the bounce count is a penalty for paddle bounces
            if (68 <= self.ball.pos[0] <= 69 and self.ball.vel[0] == 1) or (67 <= self.ball.pos[0] <= 69 and self.ball.vel[0] == 2):
                self.used_angle= True
            for ani_obj in self.animate:
                ani_obj.move()
            pre_stop = (self.ball.pos[0] == 77 and self.ball.vel[0] == 2) or (self.ball.pos[0] == 78 and self.ball.vel[0] == 1) or (self.ball.pos[0] == 78 and self.ball.vel[0] == 2)
            if pre_stop:
                self.reward += -10 * self.default_reward # negative reward for dropping the ball since done is not triggered
                self.total_score += -10 * self.default_reward * int(not self.ball.block)
                self.needs_ball_reset = True
                self.since_last_bounce = 0
                self.dropped = True
                self.ball.losses += 1
                print("dropped", np.array(self.ball.pos.tolist() + self.ball.vel.tolist() + self.paddle.pos.tolist()), self.ball.losses)
                if self.drop_stopping:
                    self.truncate = True
                    self.done = True
                    if self.ball.losses == 4 and pre_stop:
                        self.episode_rewards.append(self.total_score)
                        self.total_score = 0
                        self.since_last_bounce = 0
                        needs_reset = True
                    break

            if (self.ball.losses == 4 and pre_stop) or (self.target_mode and ((self.ball.top_wall and self.bounce_cost == 0) or self.ball.bottom_wall or self.ball.block)):
                self.average_points_per_life = self.total_score / 5.0
                self.done = True
                self.reward += -10 * self.default_reward * int(not self.ball.block)
                self.total_score += -10 * self.default_reward * int(not self.ball.block)
                self.episode_rewards.append(self.total_score)
                self.total_score = 0
                self.since_last_bounce = 0
                needs_reset = True
                print("eoe", self.total_score)
                if not self.ball.block: print("top drp", np.array(self.ball.pos.tolist() + self.ball.vel.tolist() + self.paddle.pos.tolist()))
                self.truncate = not self.ball.block and not (self.target_mode and ((self.ball.top_wall and self.bounce_cost == 0)))
                break

            if hit:
                hit = False
                self.hit_counter += 1
                if (self.get_num_points() == self.num_remove
                    or self.get_num_points() == len(self.blocks) 
                    or (self.no_breakout and self.hit_reset <= 0 and self.get_num_points() == self.num_columns + 1 and self.num_rows > 1)
                    or (self.no_breakout and self.hit_reset > 0 and self.hit_counter == self.hit_reset)
                    or self.negative_mode == "hardscatter" and self.get_num_points() == 1):
                    needs_reset = True
                    self.reward += self.completion_reward * self.default_reward
                    print("block_reset", self.get_num_points(), self.num_remove, len(self.blocks), self.hit_counter, self.hit_reset, self.no_breakout and self.hit_reset <= 0 and self.get_num_points() == self.num_columns + 1 and self.num_rows > 1, self.no_breakout and self.hit_reset > 0 and self.hit_counter == self.hit_reset)
                    if self.full_stopping:
                        self.done = True
                    break
            self.since_last_bounce += 1
            if self.since_last_bounce > 1000:
                needs_reset = True
                print("stuck reset")
            if self.ball.paddle:
                self.since_last_bounce = 0
            if self.bounce_reset > 0 and self.ball.paddle:
                self.bounce_counter += 1
                if self.bounce_counter == self.bounce_reset:
                    needs_reset = True
                    self.truncate=True
                    print("bounce_reset", self.get_num_points(), self.num_remove, len(self.blocks), self.hit_counter, self.hit_reset, self.no_breakout and self.hit_reset <= 0 and self.get_num_points() == self.num_columns + 1 and self.num_rows > 1, self.no_breakout and self.hit_reset > 0 and self.hit_counter == self.hit_reset)
                    if self.full_stopping:
                        self.done = True
                    break
            if render:
                self.render_frame()
            if pre_stop:
                break
        self.itr += 1
        full_state = self.get_state()
        frame, extracted_state = full_state['raw_state'], full_state['factored_state']
        lives = 5-self.ball.losses
        if len(self.save_path) != 0:
            if self.itr == 0:
                object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                object_dumps.close()
            self.write_objects(extracted_state, frame.astype(np.uint8))
        self.assign_assessment_stat() # bugs occur if using frame skipping
        assessment_stat = self.assessment_stat
        self.total_episode_reward += self.reward
        if needs_reset: 
            print("total episode reward: ", self.total_episode_reward)
            self.reset()
        return {"raw_state": frame, "factored_state": extracted_state}, self.reward, self.done, {"lives": lives, "TimeLimit.truncated": self.truncate, "assessment": assessment_stat}

    def run(self, policy, iterations = 10000, render=False, save_path = "runs/", save_raw = True, duplicate_actions=1, angle_mode=False, visualize=False):
        self.set_save(0, save_path, -1, save_raw)
        self.angle_mode=angle_mode
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        angle = policy.get_angle(self)
        for self.itr in range(iterations):
            if visualize:
                frame = self.render_frame()
                cv2.imshow('frame',frame)
                if cv2.waitKey(10) & 0xFF == ord(' ') & 0xFF == ord('c'):
                    continue
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


    def act(self, screen, angle=0, force=False):
        # print(screen.ball.paddlehits, screen.ball.losses, self.last_paddlehits)
        if screen.used_angle:
        #     # print(self.counter)
            self.counter = np.random.randint(4)
        if screen.ball.vel[0] > 0 and 46 <= screen.ball.pos[0] <= 47 or screen.ball.vel[0] < 0 and 67 <= screen.ball.pos[0] <= 68 or force:
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
