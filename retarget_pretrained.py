from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]
target = sys.argv[2]
final = sys.argv[3]

HAC_episode = "Episode: "
HAC_time = "Time: "
HAC_reward = "Reward: "

steps = list()
episodes = list()
returns = list()

last_episode = 0
full_dict = dict()
for line in open(target, 'r'):
    if line.find(HAC_episode) != -1:
        episodes.append(int(line[len(HAC_episode):].split("\t")[0]))
        steps.append(int(line[line.find(HAC_time) + len(HAC_time):].split("\t")[0]))
        returns.append(float(line[line.find(HAC_reward) + len(HAC_reward):].split("\t")[0][:-1]))
        if steps[-1] > 1e6:
            last_episode = episodes[-1]
            break
first_hundred = 0
for line in open(pth, 'r'):
    if line.find(HAC_episode) != -1:
        if first_hundred < 100:
            first_hundred += 1
            continue
        print(line)
        episodes.append(int(line[len(HAC_episode):].split("\t")[0]) + last_episode)
        steps.append(int(line[line.find(HAC_time) + len(HAC_time):].split("\t")[0]) + 1e6)
        returns.append(float(line[line.find(HAC_reward) + len(HAC_reward):].split("\t")[0][:-1]))
final = open(final, 'w')
for i in range(len(episodes)):
    fstr = "\t".join([HAC_episode + str(episodes[i]), HAC_time + str(steps[i]), HAC_reward + str(returns[i])]) + "\n"
    final.write(fstr)
# print(episodes, steps, returns)

# result_dict = {"steps": steps, "episodes": episodes, "returns": returns, "assess": assess, "episode_rewards": episode_reward}
# full_dict[filename] = result_dict
# save_to_pickle(target, full_dict)
# print(full_dict)
# print([(k, result_dict[k], len(result_dict[k]) )for k in result_dict.keys()])