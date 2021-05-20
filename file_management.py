import pickle, os
import numpy as np
import imageio as imio
import sys, cv2 

def load_from_pickle(pth):
    fid = open(pth, 'rb')
    save_dict = pickle.load(fid)
    fid.close()
    return save_dict

def save_to_pickle(pth, val):
    try:
        os.makedirs(os.path.join(*pth.split("/")[:-1]))
    except OSError:
        pass
    fid = open(pth, 'wb')
    pickle.dump(val, fid)
    fid.close()

def get_edge(train_edge):
    '''
    edges in format tail1_tail2...->head
    '''
    splt = train_edge.split("->")
    tail = splt[0].split("_")
    head = splt[1]
    return head,tail

def get_cp_models_from_dict(cp_dict):
    keys = [k for k in cp_dict.keys()]
    keys.sort()
    keys.pop(0)
    return keys, [cp_dict[k] for k in keys]

def dump_from_line(line, time_dict):
    for obj in line.split('\t'):
        if obj == "\n":
            continue
        else:
            split = obj.split(":")
            name = split[0]
            vals = split[1].split(" ")
            state = [float(i) for i in vals]
            # BB = (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]))
            # pos = (int(vals[1]),)
            time_dict[name] = state
    return time_dict

def get_start(pth, filename, i, rng):
    total_len = 0
    if i < 0:
        for line in open(os.path.join(pth, filename), 'r'):
            total_len += 1
        print("length", total_len)
        if rng == -1:
            i = 0
        else:
            i = max(total_len - rng, 0)
    return i, total_len



def read_obj_dumps(pth, i= 0, rng=-1, filename='object_dumps.txt'):
    '''
    TODO: move this out of this file to support reading object dumps from other sources
    i = -1 means count rng from the back
    rng = -1 means take all after i
    i is start position, rng is number of values
    '''
    obj_dumps = []
    i, total_len = get_start(pth, filename, i, rng)
    current_len = 0
    for line in open(os.path.join(pth, filename), 'r'):
        current_len += 1
        if current_len< i:
            continue
        if rng != -1 and current_len > i + rng:
            break
        time_dict = dump_from_line(line, dict())
        obj_dumps.append(time_dict)
    return obj_dumps

def visualize_frame_dumps(pth, i= 0, rng=-1, filename='focus_dumps.txt'):
    '''
    TODO: move this out of this file to support reading object dumps from other sources
    i = -1 means count rng from the back
    rng = -1 means take all after i
    i is start position, rng is number of values
    '''
    obj_dumps = []
    i, total_len = get_start(pth, filename, i, rng)
    current_len = 0
    for line in open(os.path.join(pth, filename), 'r'):
        current_len += 1
        if current_len < i:
            continue
        if rng != -1 and current_len > i + rng:
            break
        time_dict = dump_from_line(line, dict())
        print(current_len)
        d = str((current_len-1) // 2000)
        j = (current_len-1) % 2000
        p = os.path.join(pth, d, "state" + str(j) + ".png")
        raw_state = imio.imread(p)
        pval = ""
        for k in time_dict.keys():
            if k != 'Action' and k != 'Reward':
                raw_state[int(time_dict[k][0][0]), :] = 255
                raw_state[:, int(time_dict[k][0][1])] = 255
            if k == 'Action' or k == 'Reward':
                pval += k + ": " + str(time_dict[k][1]) + ", "
            else:
                pval += k + ": " + str(time_dict[k][0]) + ", "
        print(pval[:-2])
        raw_state = cv2.resize(raw_state, (336, 336))
        cv2.imshow('frame',raw_state)
        if cv2.waitKey(10000) & 0xFF == ord(' ') & 0xFF == ord('c'):
            continue
    return obj_dumps

def get_raw_data(pth, i=0, rng=-1):
    '''
    loads raw frames, i denotes starting position, rng denotes range of values. If 
    '''
    frames = []
    if rng == -1:
        try:
            f = i
            while True:
                frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
                f += 1
        except OSError as e:
            return frames
    else:
        for f in range(i, i + rng[1]):
            frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
    return frames


def get_individual_data(name, obj_dumps, pos_val_hash=3):
    '''
    gets data for a particular object, gets everything in obj_dumps
    pos_val_hash gets either position (1), value (2), full position and value (3)
    '''
    data = []
    if len(obj_dumps) > 0:
        names = list(obj_dumps[0].keys())
        relevant_names = [n for n in names if n.find(name) != -1]
        relevant_names.sort() # sorting procedure should be fixed between this and state getting
    for time_dict in obj_dumps:
        # print("td1", time_dict[name][1])
        if pos_val_hash == 1:
            data.append(sum([list(time_dict[name][0]) for name in relevant_names], []))
        elif pos_val_hash == 2:
            # print("td2", list(time_dict[name][1]))
            data.append(sum([list(time_dict[name][1]) for name in relevant_names], []))
        elif pos_val_hash == 3:
            data.append(sum([list(time_dict[name][0]) + list(time_dict[name][1]) for name in relevant_names], []))
        else:
            data.append(sum([list(time_dict[name]) for name in relevant_names], []))
    return np.array(data)

def default_value_arg(kwargs, key, value):
    if key in kwargs:
        return kwargs[key]
    else:
        return value

def render_dump(obj_dumps):
    frame = np.zeros((84,84), dtype = 'uint8')
    for bn in [bn for bn in obj_dumps.keys() if bn.find('Block') != -1]:
        block = obj_dumps[bn]
        pos = (int(block[0][0]), int(block[0][1]))
        # print(pos, block[1])
        if block[1][0] == 1:
            frame[pos[0]-1:pos[0]+1, pos[1]-1:pos[1]+2] = .5 * 255
    walln = "TopWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    # print(pos)
    width = 84
    height = 4
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "RightSideWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 4
    height = 84
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "LeftSideWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 4
    height = 84
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    walln = "BottomWall"
    wall = obj_dumps[walln]
    pos = (int(wall[0][0]), int(wall[0][1]))
    width = 84
    height = 4
    frame[pos[0]-height//2:pos[0]+height//2, pos[1]-width//2:pos[1]+width//2] = .3 * 255

    pos = (int(obj_dumps["Paddle"][0][0]), int(obj_dumps["Paddle"][0][1]))
    width = 7
    height = 2
    frame[pos[0]:pos[0]+height, pos[1]-3:pos[1]+4] = .75 * 255

    pos = (int(obj_dumps["Ball"][0][0]), int(obj_dumps["Ball"][0][1]))
    width = 2
    height = 2
    frame[pos[0]-1:pos[0]+1, pos[1]-1:pos[1]+1] = 1.0 * 255
    return frame

def printframe(state, name='frame', waittime=100):
    cv2.imshow(name, state)
    if cv2.waitKey(waittime) & 0xFF == ord('q'):
        pass

def saveframe(state, pth='data/', count=-1, name='frame'):
    try:
        os.makedirs(pth)
    except OSError:
        pass
    imio.imsave(os.path.join(pth), state)



if __name__ == '__main__':
    visualize_frame_dumps(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
