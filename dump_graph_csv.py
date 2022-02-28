from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]


steps = list()
returns = list()
for filename in os.listdir(pth):
    for line in open(os.path.join(pth, filename), 'r'):
        try:
            steps.append(int(line.split(",")[0]))
            returns.append(float(line.split(",")[1]))
        except ValueError as e:
            continue
save_to_pickle(sys.argv[2], {"name": pth, "steps": steps, "returns": returns})