from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]

steps = list()
returns = list()
for line in open(pth, 'r'):
    print(line.split(","))
    try:
        steps.append(int(line.split(",")[1]))
        returns.append(float(line.split(",")[2][:-1]) * 3)
    except ValueError as e:
        pass
print(steps, returns)
save_to_pickle(sys.argv[2], {"base": {"dummy": {"steps": steps, "returns": returns}}, "pretrain": {}})
