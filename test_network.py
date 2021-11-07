import numpy as np
from file_management import ObjDict
from Networks.network import PairNetwork, pytorch_model
from Environments.SelfBreakout.breakout_screen import Screen
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

def row_counter(blocks, row):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += 1
    return count

def col_counter(blocks, col):
    count = 0
    for b in blocks:
        if b.pos[1] <= 12 + col * b.width < b.pos[1] + b.width:
            count += 1
    return count

def attribute_sum(blocks, row):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += b.attribute
    return count


def attribute_bin(blocks, row):
    count = 0
    for b in blocks:
        if b.pos[0] <= 22 + row * b.height < b.pos[0] + b.height:
            count += b.attribute
    return float(count > 0)

def hot_act(act, act_num):
    z = np.zeros(act_num)
    z[int(act)] = 1
    return z

def construct_state(state, act, act_num=5, num_samples=50):
    fac_state = state["factored_state"]
    ballstate = (np.array(fac_state["Ball"].copy()) - np.array([42, 42, 0, 0, 1])) / np.array([42, 42, 2, 1, 1])
    state_comp = [hot_act(act, act_num), ballstate]
    mean, var = np.array([22, 42, 0, 0, 0]), np.array([10, 30, 1, 1, 1])
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
    args.num_frames = 10000
    args.object_dim = 5
    args.hidden_sizes = [128,128,1024]
    args.output_dim = 1
    args.post_dim = 0
    args.include_last = True
    args.reduce = True
    args.init_form = "xnorm"
    args.num_outputs = 1
    args.activation = "tanh"
    args.batchsize = 64
    args.aggregate_final = True
    args.use_layer_norm = False
    num_iters = 100000
    num_rows = 5
    num_samples = 50
    num_columns = 20
    args.first_obj_dim = 5 + num_rows
    args.num_inputs = 5 + 5 * num_samples

    test_network = PairNetwork(**args)
    test_network.cuda()
    screen = Screen(num_rows = num_rows, num_columns = num_columns, max_block_height=4, random_exist=num_samples, negative_mode = "zerorand")
    blocks = screen.blocks
    states = list()
    values = list()
    for i in range(args.num_frames):
        state = screen.reset()
        act = np.random.randint(num_rows)
        num = row_counter(screen.blocks, act)
        states.append(construct_state(state, act, num_rows, num_samples))
        # print(construct_state(state, act, num_rows, num_samples))
        # im = screen.render()
        # cv2.imshow('frame',im)
        # if cv2.waitKey(100) & 0xFF == ord(' ') & 0xFF == ord('c'):
        #     continue
        values.append(num)
    states = np.array(states)
    values = np.array(values)
    test_states = list()
    test_values = list()
    for i in range(int(args.num_frames * .1)):
        state = screen.reset()
        act = np.random.randint(num_rows)
        num = row_counter(screen.blocks, act)
        test_states.append(construct_state(state, act, num_rows, num_samples))
        test_values.append(num)
    test_states = np.array(test_states)
    test_values = np.array(test_values)

    # loss = nn.BCELoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(test_network.parameters(), 1e-4, eps=1e-5, betas=(.9,.999), weight_decay=0)

    mean_loss = 0
    for i in range(num_iters):
        idxes = np.random.randint(len(states), size=(args.batchsize,))
        batch = pytorch_model.wrap(states[idxes], cuda=True)
        labels = pytorch_model.wrap(values[idxes], cuda=True).unsqueeze(1)
        outputs = test_network(batch)
        loss = loss_fn(outputs, labels)
        run_optimizer(test_network, optimizer, loss)
        mean_loss += loss

        if i % 1000 == 0:
            idxes = np.random.randint(len(test_states), size=(args.batchsize * 2,))
            batch = pytorch_model.wrap(test_states[idxes], cuda=True)
            toutputs = test_network(batch)
            tlabels = pytorch_model.wrap(test_values[idxes], cuda=True).unsqueeze(1)
            test_loss = (toutputs - tlabels).pow(2).sum(dim=1).pow(.5).mean()
            print("Iters: ", i, ": ", loss, test_loss, torch.cat([toutputs, tlabels], dim=1)[:20])

    
            mean_loss = 0