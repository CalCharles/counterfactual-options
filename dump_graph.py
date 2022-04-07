from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]
target = sys.argv[2]

iters = "Iters:  "
tmr = "Test mean returns: "
stl = "Steps:  "
epl = "Episodes:  "
rt = "Test mean returns: "
ase = "Assess Train: "
erl = "total episode reward:  "

HAC_episode = "Episode: "
HAC_time = "Time: "
HAC_reward = "Reward: "


full_dict = dict()
for filename in os.listdir(pth):
    steps = list()
    episodes = list()
    current_episode_rewards = list()
    returns = list()
    assess = list()
    episode_reward = list()
    for line in open(os.path.join(pth, filename), 'r'):
        if line.find(iters) != -1:
            steps.append(int(line[line.find(stl) + len(stl):].split(" ")[0]))
            episodes.append(int(line[line.find(epl) + len(epl):-1]))
            if len(current_episode_rewards) > 0:
                cer = np.mean(current_episode_rewards)
                print("appending")
                if cer < -30:
                    cer = cer - 10
                episode_reward.append(cer)
            elif len(episode_reward) > 0:
                episode_reward.append(episode_reward[-1])
            current_episode_rewards = list()
        if line.find(HAC_episode) != -1:
            print(line)
            episodes.append(int(line[len(HAC_episode):].split("\t")[0]))
            steps.append(float(line[line.find(HAC_time) + len(HAC_time):].split("\t")[0]))
            if float(line[line.find(HAC_reward) + len(HAC_reward):].split("\t")[0][:-1]) < -50:
                returns.append(-50)
            else:
                returns.append(float(line[line.find(HAC_reward) + len(HAC_reward):].split("\t")[0][:-1]))
        if line.find("Test mean returns: ") != -1:
            print(line)
            print(line.find(rt), line[line.find(rt) + len(rt)], line[line.find(rt) + len(rt):].split(" ")[0])
            print(line.find(ase), line[line.find(ase) + len(ase):].split(" ")[0])
            returns.append(float(line[line.find(rt) + len(rt):].split(" ")[0]))
            assess.append(float(line[line.find(ase) + len(ase):].split(" ")[0]))
        if line.find(erl) != -1:
            current_episode_rewards.append(float(line[line.find(erl) + len(erl):-1]))
    result_dict = {"steps": steps, "episodes": episodes, "returns": returns, "assess": assess, "episode_rewards": episode_reward}
    full_dict[filename] = result_dict
save_to_pickle(target, full_dict)
# print(full_dict)
# print([(k, result_dict[k], len(result_dict[k]) )for k in result_dict.keys()])