from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]
target = sys.argv[2]
episode_reader = True if len(sys.argv) > 3

iters = "Iters:  "
tmr = "Test mean returns: "
stl = "Steps:  "
epl = "Episodes:  "
rt = "Test mean returns: "
ase = "Assess Train:  "
erl = "total episode reward:  "


steps = list()
episodes = list()
current_episode_rewards = list()
returns = list()
assess = list()
episode_reward = list()
for filename in os.listdir(pth):
    for line in open(os.path.join(pth, filename), 'r'):
        if line.find(iters) != -1:
            steps.append(int(line[line.find(stl) + len(stl):].split(" ")[0]))
            steps.append(int(line[line.find(epl) + len(epl:-1)]))
            if len(current_episode_rewards) > 0:
                cer = np.mean(current_episode_rewards)
                episode_reward.append(cer)
            elif len(episode_reward) > 0:
                episode_reward.append(episode_reward[-1])
            current_episode_rewards = list()
        if line.find("Test mean returns: ") != -1:
            returns.append(float(line[line.find(rt) + len(rt)].split(" ")[0]))
            assess.append(float(line[line.find(ase) + len(ase)].split(" ")[0]))
        if line.find(erl) != -1:
            current_episode_rewards.append(float(line[line.find(erl) + len(erl):-1]))
