# read prox data
import os
from collections import deque
import numpy as np

pth = "logs/breakout/variant_trials/prox/"
filename = "train_full_block_no_breakout_dqn_prox2.txt\r"
last100 = deque(maxlen=500)
for line in open(os.path.join(pth, filename), 'r'): # there should only be one line since actions are tab separated
	if line.find("hit at l1") != -1:
		# print(line[len("hit at l1"):].split(" ")[1])
		last100.append(float(line[len("hit at l1"):].split(" ")[1]))
	elif line.find("Iters") != -1:
		print(line[len("Iters:  "):].split(" ")[0])
		print(np.mean(last100))