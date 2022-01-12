import numpy as np
from file_management import ObjDict, save_to_pickle, load_from_pickle, read_obj_dumps, read_action_dumps, get_start
from Networks.network import PairNetwork, pytorch_model, BasicMLPNetwork
from Environments.SelfBreakout.breakout_screen import Screen, AnglePolicy
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import copy
from tianshou.policy import BasePolicy


print("value")

def row_counter(blocks, row, flipped=False, no_reset=False):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += 1
    return count

def col_counter(blocks, col, flipped=False, no_reset=False):
    count = 0
    for b in blocks:
        if b.pos[1] <= 12 + col * b.width < b.pos[1] + b.width:
            count += 1
    return count

def attribute_sum(blocks, row, flipped=False, no_reset=False):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += b.attribute
    return count


def attribute_bin(blocks, row, flipped=False, no_reset=False):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += b.attribute
    return float(count > 0)

class BreakoutTargetDeterminer():
    def __init__(self, env):
        self.current_environment = env 
        self.angle_policy = AnglePolicy(4)

    def find_block(self, blocks, block):
        for i, b in enumerate(blocks):
            if np.sum(np.abs(b.pos - block.pos)) < .1:
                return i
        return -1

    def hit_determine(self, blocks, act, flipped = False, no_reset = False, action_choice=False, keep_both=False, freeze_model=False):
        # is only able to return current samples, should be ok though
        current_model = self.current_environment
        if action_choice or freeze_model:
            alternate_model = copy.deepcopy(self.current_environment)
        block_bin = np.zeros(len(blocks))
        def run_action(act, model, render=False):
            block_hit = False
            action = self.angle_policy.act(model, angle=act, force=True)
            max_count = 0 
            hitstart = model.ball.paddlehits
            last_hit_block = model.ball.pos[0] < 50
            while not block_hit and max_count < 150 and model.ball.paddlehits - hitstart < 2 - int(not last_hit_block):
                action = self.angle_policy.act(model, angle=act)
                model.step(action)
                block_hit = model.ball.block
                max_count += 1
                # if render:
                #     im = model.render()
                #     # print(action, act)
                #     cv2.imshow('frame',im)
                #     if cv2.waitKey(10) & 0xFF == ord(' ') & 0xFF == ord('c'):
                #         continue

            # print(max_count, act, int(model.ball.block_id.name[5:]))
        # print(current_model.hit_counter)
        if freeze_model:
            run_action(act, alternate_model, render=True)
            block_id = alternate_model.ball.block_id
        else:
            run_action(act, current_model, render=True)
            block_id = current_model.ball.block_id
        if action_choice or keep_both:
            if current_model.ball.block_id is None:
                true_block = np.array([0,0,0,0,0])
            else:
                true_block = np.array(current_model.ball.block_id.getMidpoint() + [0,0,0])
            alt_act = (act + np.random.randint(3)+1) % 4
            run_action(alt_act, alternate_model, render=False)
            if alternate_model.ball.block_id is None:
                alt_block = np.array([0,0,0,0,0])
            else:
                alt_block = np.array(alternate_model.ball.block_id.getMidpoint() + [0,0,0])
            block_choice = np.random.rand() > .6
            print(true_block, alt_block, np.sum(np.abs(true_block - alt_block)) < .1, block_choice)
            if keep_both:
                act = (act, alt_act)
                true_block = (true_block, alt_block) 
            elif np.sum(true_block) == 0:
                return -1, true_block, act
            elif np.sum(np.abs(true_block - alt_block)) < .1 or block_choice: # if the block would be the same regardless of action, return the true block and 10 reward
                # print(np.sum(np.abs(true_block - alt_block)), block_choice)
                return 1, true_block, act
            else: # still return the true block as parameter, but return -1 reward for hitting the wrong block
                return -1, true_block, alt_act


        if flipped:
            if block_id is None:
                return np.array([0,0,0,0,0])
            return np.array(block_id.getMidpoint() + [0,0,0])
        else:
            block_bin[int(block_id.name[5:])] = 1
            return block_bin

def determine_bounce(factored_state, next_factored_state):
    vertical_before = factored_state["Ball"][2]
    vertical_loc = factored_state["Ball"][0]
    vertical_after = next_factored_state["Ball"][2]
    vel_after = next_factored_state["Ball"][2:4]
    if vertical_before > 0 and vertical_after < 0 and vertical_loc > 60:
        return True, vel_after
    return False, vel_after

def determine_hit(blocks_state, next_blocks_state):
    blocks_hit = blocks_state[:,-1]
    next_blocks_hit = next_blocks_state[:,-1]
    blocks_diff = blocks_hit - next_blocks_hit
    if np.sum(blocks_diff) > 0:
        return True, np.nonzero(blocks_diff)
    return False, [-1]

def get_blocks_state(factored_state):
    i=0
    name = "Block" + str(i)
    blocks = list()
    while name in factored_state:
        blocks.append(factored_state[name])
        i += 1
        name = "Block" + str(i)
    blocks = copy.deepcopy(np.array(blocks))
    return blocks

class TargetingPolicy():
    def __init__(self, env, network, construct_state, num_samples):
        self.current_environment = env 
        self.angle_policy = AnglePolicy(4)
        self.network = network
        self.sampler = BreakoutTargetDeterminer(env)
        self.construct_state = construct_state
        self.dist_fn = torch.distributions.Categorical
        self.num_samples = num_samples
        self.act = 0
        self.target = None
        self.hit = 0
        self.miss = 0

    def forward(self, state):
        block_hit = self.current_environment.ball.block
        reset = self.current_environment.resetted
        if block_hit or self.target is None:
            if block_hit: 
                print("hit block", self.current_environment.ball.block_id.getMidpoint(), self.target)
                vals = np.sum(np.abs(self.current_environment.ball.block_id.getMidpoint() - self.target[:2]))
                self.hit += int(vals == 0)
                self.miss += int(vals != 0)
            target_act = np.random.randint(4)
            self.target = self.sampler.hit_determine(self.current_environment.blocks,
             target_act, flipped = True, freeze_model = True)
            state = self.construct_state(state, self.target, num_samples=self.num_samples, flipped = True)
            print(state)
            state = pytorch_model.wrap(state, cuda=True)
            logits = self.network(state)
            dist = self.dist_fn(logits)
            self.act = dist.sample()
        action = self.angle_policy.act(self.current_environment, angle=self.act)
        return action

class TSNetworkWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.last = network
        self.network = network

    def forward(
        self,
        obs,
        state=None,
        info=None
    ):
        print(obs)
        obs = pytorch_model.wrap(obs, cuda= True)
        return self.network(obs), None


def hot_act(act, act_num):
    z = np.zeros(act_num)
    z[int(act)] = 1
    return z

def hot_dataset(dataset, test_dataset):
    if args.flipped:
        v_list = list()
        for a in dataset:
            v = hot_act(a, 4)
            v_list.append(v)
        dataset = np.array(v_list)
        v_list = list()
        for a in test_dataset:
            v = hot_act(a, 4)
            v_list.append(v)
        test_dataset = np.array(v_list)
    return dataset, test_dataset


def construct_state(state, act, act_num=5, num_samples=50, flipped = False):
    fac_state = state["factored_state"]
    ballstate = (np.array(fac_state["Ball"].copy()) - np.array([42, 42, 0, 0, 1])) / np.array([42, 42, 2, 1, 1])
    mean, var = np.array([32, 42, 0, 0, 0]), np.array([10, 42, 1, 1, 1])
    if flipped:
        state_comp = [(act-mean)/var, ballstate]
    else:
        state_comp = [hot_act(act, act_num), ballstate]
    
    for i in range(num_samples):
        state_comp = state_comp + [(np.array(fac_state["Block" + str(i)].copy()) - mean) / var]
    return np.concatenate(state_comp, axis=0)

def run_optimizer(network, optimizer, loss):
    optimizer.zero_grad()
    (loss.mean()).backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
    optimizer.step()



print(__name__)
if __name__ == "__main__":
    '''
    Testing cases for point network:
        count of blocks in a row
        count of blocks in a column
        sum of block attributes in a row > 0
        given random 10x2 arrangement of blocks, a target block and an angle, indicator on whether block will be hit
        given random 10 row of blocks, target block and angle, indicator on whether block will be hit
    '''
    args = ObjDict()
    args.num_frames = 50000
    args.object_dim = 5
    args.hidden_sizes = [128,128,128,256,1024]
    args.post_dim = 0
    args.include_last = True
    args.reduce = True
    args.init_form = "xnorm"
    args.activation = "relu"
    args.batchsize = 64
    args.use_layer_norm = False
    args.activation_final = "softmax" # row counter mode
    args.num_iters = 30000
    args.num_rows = 2 # row counter mode
    args.num_samples = 20 # row counter mode
    args.num_outputs = 4 # row counter mode
    # args.num_outputs = num_samples
    args.num_columns = 10
    args.negative_mode = "" 
    # args.negative_mode = "zerorand" # attribute counter mode
    # args.num_actions = num_rows
    args.num_actions = 4
    # args.num_actions = 5
    # args.first_obj_dim = 5
    # args.first_obj_dim = 5 + args.num_actions
    args.first_obj_dim = 10
    # args.num_inputs = 5 + 5 * args.num_samples + args.num_actions
    # args.num_inputs = 5 + 5 * args.num_samples
    args.num_inputs = 10 + 5 * args.num_samples
    args.num_outputs = args.num_outputs # row counter mode
    args.no_breakout = True
    args.hit_reset = 15
    args.action_choice = False
    args.flipped = True # row counter mode
    args.no_reset = True
    args.keep_both = False
    args.late_state = False
    args.filename = "ball_test_policy_small"
    args.generate_data = False
    args.generate_from_obj_dict = False
    args.generate_from_collector_buffer = "/hdd/datasets/counterfactual_data/breakout/pretrain_block2.10.4.1.15/pretrain_collector.pkl"
    args.choice_labels = False
    args.register_adjacent = 0
    np.set_printoptions(threshold = 1000000, linewidth = 150, precision=3)


    screen = Screen(num_rows = args.num_rows, num_columns = args.num_columns, max_block_height=4, random_exist=args.num_samples, negative_mode = args.negative_mode, no_breakout = args.no_breakout, hit_reset=args.hit_reset)
    determiner = BreakoutTargetDeterminer(screen)
    blocks = screen.blocks

    # change values for different tests:
    # label function
    # label_maker = row_counter # row counter mode
    label_maker = determiner.hit_determine

    # network values
    args.output_dim = 1 # row counter mode
    args.aggregate_final = True # row counter mode

    # args.output_dim = 1
    # args.aggregate_final = False

    args.output_dim = args.num_outputs
    args.aggregate_final = True

    # loss function
    loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss() # row counter mode

    test_network = PairNetwork(**args)
    # test_network = BasicMLPNetwork(**args)
    test_network.cuda()

    # load mode
    print(args)

    def construct_states_values(num_frames=-1, state_dict=None, param_dict=None):

        states = list()
        values = list()
        choice_actions = list()
        for i in range(num_frames):
            if not args.no_reset: state = screen.reset()
            else: state = screen.get_state()
            if args.action_choice:
                rand_act = np.random.randint(4)
                print(rand_act)
                label, act, choice_action = label_maker(screen.blocks, rand_act, flipped=args.flipped, no_reset=args.no_reset, action_choice=args.action_choice)
                # Act is the parameter
                choice_actions.append(choice_action)
                print(i, act, rand_act, choice_action)
            elif args.flipped:
                label = np.random.randint(4)
                act = label_maker(screen.blocks, label, flipped=args.flipped, no_reset=args.no_reset)
            else:
                act = np.random.randint(4)
                label = label_maker(screen.blocks, act, flipped=args.flipped, no_reset=args.no_reset)
            state = construct_state(state, act, args.num_actions, args.num_samples, flipped=args.flipped or args.action_choice)
            states.append(state)
            # print(state, label)
            # print(construct_state(state, act, num_actions, num_samples))
            # im = screen.render()
            # cv2.imshow('frame',im)
            # if cv2.waitKey(100) & 0xFF == ord(' ') & 0xFF == ord('c'):
            #     continue
            values.append(label)
        states = np.array(states)
        values = np.array(values)
        if args.action_choice: choice_actions = np.array(choice_actions)
        return states, values, choice_actions


    if not args.generate_data:
        if len(args.generate_from_collector_buffer) > 0:
            buffer = load_from_pickle(args.generate_from_collector_buffer).hindsight_buffer
            buffer = buffer.sample(0)[0]
            labels = np.array([hot_act(a, 4) for a in buffer.act])
            print(labels)
            states = buffer.obs[:int(len(buffer.obs) * .9)]
            print(states[:10])
            values = labels[:int(len(buffer.obs) * .9)]
            test_states = buffer.obs[int(len(buffer.obs) * .9):]
            test_values = labels[int(len(buffer.obs) * .9):]
            error
        else:
            values = load_from_pickle("data/breakout/network_test/values" + args.filename + ".pkl")
            states = load_from_pickle("data/breakout/network_test/states" + args.filename + ".pkl")
            if args.action_choice: choice_actions = load_from_pickle("data/breakout/network_test/actions" + args.filename + ".pkl")
            test_values = load_from_pickle("data/breakout/network_test/test_values" + args.filename + ".pkl")
            test_states = load_from_pickle("data/breakout/network_test/test_states" + args.filename + ".pkl")
            if args.action_choice: test_choice_actions = load_from_pickle("data/breakout/network_test/test_actions" + args.filename + ".pkl")
    else:
        if args.generate_from_obj_dict:
            print("../hdd_data/breakout/" + args.filename + "/")
            true_options = list()
            corresponding_option = list()

            states = list()
            values = list()
            true_action = None
            param = None
            option_action = None
            angle_reference = {(-1,-1): 0, (-2,-1):1, (-2, 1):2, (-1,1):3}
            miss = 0
            rate = 0
            i, total_len = get_start("../hdd_data/breakout/" + args.filename + "/", "object_dumps.txt", 0, -1)
            if total_len > 10000:
                num_segments = total_len // 10000 + int(total_len % 10000 > 0)
            else:
                num_segments = 1
            for k in range(num_segments):
                full_data = read_obj_dumps("../hdd_data/breakout/" + args.filename + "/", filename="object_dumps.txt", i = k * 10000, rng=10000)
                option_actions, idxes, additional = read_action_dumps("../hdd_data/breakout/" + args.filename + "/", filename="param_dumps.txt", indexed=True, i = k*10000, rng=10000)
                # print(full_data[0], len(full_data))
                # print(option_actions[0], len(option_actions))
                # print(idxes[0])
                # print(additional[0])
                if k == 0:
                    option_actions = option_actions[1:]
                    idxes = idxes[1:]
                    factored_state = full_data[0]
                    state = {'factored_state': factored_state} 
                    blocks_state = get_blocks_state(factored_state)
                    full_data = full_data[1:]
                    last_option_action = None
                # else:
                    # print(last_option_action)
                    # option_actions = [last_option_action] + option_actions

                for opt_a, next_factored_state in zip(option_actions, full_data):
                    bounce, angle = determine_bounce(factored_state, next_factored_state)
                    if bounce:
                        true_action = angle_reference[tuple(angle)]
                        option_action = angle_reference[tuple(opt_a[2:4])]
                        if option_action != true_action:
                            miss += 1
                        rate += 1
                        print("bounce", factored_state["Ball"], next_factored_state["Ball"])
                    next_blocks_state = get_blocks_state(next_factored_state)
                    hit, block_ids = determine_hit(blocks_state, next_blocks_state)
                    if hit:
                        param = next_blocks_state[block_ids[0][0]]
                        # param = next_blocks_state[np.random.randint(20)]
                        param[-1] = 0
                        completed_state = construct_state(state, param, args.num_actions, args.num_samples, flipped=args.flipped or args.action_choice)
                        print("ballstate", state["factored_state"]["Ball"], next_factored_state["Ball"])
                        if option_action is not None:
                            states.append(completed_state)
                            true_options.append(true_action)
                            corresponding_option.append(option_action)
                        print(miss/rate, next_blocks_state[:,-1], completed_state, param, factored_state["Ball"], next_factored_state["Ball"], true_action, option_action)
                        state = {'factored_state': next_factored_state}
                    if next_factored_state["Done"][0]:
                        state = {'factored_state': next_factored_state}
                    factored_state = next_factored_state
                    blocks_state = next_blocks_state
                last_option_action = option_actions[-1]
            values = true_options
            choice_actions = corresponding_option
            test_states = np.array(states[-int(len(states) * .9):])
            test_values = np.array(values[-int(len(values) * .9):])
            test_choice_actions = np.array(choice_actions[-int(len(choice_actions) * .9):])
            states = np.array(states[:int(len(states) * .9)])
            values = np.array(values[:int(len(values) * .9)])
            choice_actions = np.array(choice_actions[:int(len(choice_actions) * .9)])
            values, test_values = hot_dataset(values, test_values)
            choice_actions, test_choice_actions = hot_dataset(choice_actions, test_choice_actions)
        else:
            states, values, choice_actions = construct_states_values(args.num_frames)
            test_states, test_values, test_choice_actions = construct_states_values(int(args.num_frames * .1))
            values, test_values = hot_dataset(values, test_values)
            print("values", np.sum(values))

        # save_to_pickle("data/breakout/network_test/values" + args.filename + ".pkl", values)
        # save_to_pickle("data/breakout/network_test/states" + args.filename + ".pkl", states)
        # if args.action_choice or args.generate_from_obj_dict: save_to_pickle("data/breakout/network_test/actions" + args.filename + ".pkl", choice_actions)
        # save_to_pickle("data/breakout/network_test/test_values" + args.filename + ".pkl", test_values)
        # save_to_pickle("data/breakout/network_test/test_states" + args.filename + ".pkl", test_states)
        # if args.action_choice or args.generate_from_obj_dict: save_to_pickle("data/breakout/network_test/test_actions" + args.filename + ".pkl", test_choice_actions)

    if args.choice_labels:
        values = choice_actions
        test_values = test_choice_actions
    torch.set_printoptions(precision=3)

    optimizer = optim.Adam(test_network.parameters(), 1e-4, eps=1e-5, betas=(.9,.999), weight_decay=0)
    print(type(states), type(values))
    print( states.shape, values.shape)
    mean_loss = 0
    for i in range(args.num_iters):
        idxes = np.random.randint(len(states), size=(args.batchsize,))
        batch = pytorch_model.wrap(states[idxes], cuda=True)
        labels = pytorch_model.wrap(values[idxes], cuda=True)
        outputs = test_network(batch)
        if args.action_choice: 
            actions = choice_actions[idxes]
            # print(labels.shape, batch.shape, outputs.shape, actions, len(idxes))
            # print(outputs[:20], outputs[list(range(len(idxes))), actions], actions, choice_actions[:100], choice_actions[100:200], choice_actions[200:300], choice_actions[300:400])
            # print(choice_actions.tolist())

            outputs = outputs[list(range(len(idxes))), actions]
        if len(labels.shape) < len(outputs.shape):
            labels = labels.unsqueeze(-1)
        # print(labels, outputs.shape)
        loss = loss_fn(outputs, labels)
        run_optimizer(test_network, optimizer, loss)
        mean_loss += loss

        if i % 1000 == 0:
            idxes = np.random.randint(len(test_states), size=(args.batchsize * 2,))
            batch = pytorch_model.wrap(test_states[idxes], cuda=True)
            toutputs = test_network(batch)
            tlabels = pytorch_model.wrap(test_values[idxes], cuda=True)
            if args.action_choice:
                clabels = test_values[idxes]
                coutputs = pytorch_model.unwrap(toutputs)
                tactions = test_choice_actions[idxes]
                toutputs = toutputs[list(range(len(idxes))), tactions]
                ctoutputs = coutputs[list(range(len(idxes))), tactions]
                match_positive = len(np.abs((tactions - np.argmax(coutputs, axis=1))[clabels > 0]).nonzero()[0]) / np.sum((clabels > 0).astype(int))
                match_negative = np.sum((np.abs((tactions - np.argmax(tactions))[clabels < 0]) == 0).astype(int))
                print("positive negative", np.sum((clabels > 0).astype(int)), np.abs((tactions - np.argmax(coutputs, axis=1))[clabels > 0]), match_positive, match_negative)
            if len(tlabels.shape) < len(toutputs.shape):
                tlabels = tlabels.unsqueeze(-1)
            # print(toutputs[:20], tlabels[:20])
            toutputs.detach()
            test_loss = loss_fn(toutputs, tlabels)
            # if not args.action_choice:
            #     toutputs[toutputs < .5] = 0
            match_count = np.sum((np.sum(np.abs(pytorch_model.unwrap(toutputs - tlabels)), axis = 1) < .3).astype(int))
            print(toutputs[0], outputs[0])
            # print("Iters: ", i, ": ", loss, test_loss, torch.cat([toutputs, tlabels], dim=1)[:20])
            # print("Iters: ", i, ": ", loss, test_loss, toutputs.nonzero(), tlabels.nonzero(), match_count)            
            print("Iters: ", i, ": ", loss, test_loss, batch[0], toutputs[:10], tlabels[:10])
            # print("Iters: ", i, ": ", loss, test_loss, coutputs[:20], toutputs[:20], tlabels[:20], tactions[:20])

            mean_loss = 0
            torch.save(test_network, "data/breakout/network_test/net_"+ args.filename + ".pt")
    torch.save(test_network, "data/breakout/network_test/net_"+ args.filename + ".pt")
