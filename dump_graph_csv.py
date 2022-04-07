from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]

full_data_pretrain = dict()
full_data_base = dict()
for filename in os.listdir(pth):
    steps = list()
    returns = list()
    for line in open(os.path.join(pth, filename), 'r'):
        try:
            steps.append(int(line.split(",")[0]))
            returns.append(float(line.split(",")[1]))
        except ValueError as e:
            continue
    if filename.find("pretrain") != -1:
        full_data_pretrain[filename] = {"steps": steps, "returns": returns}
    else:
        full_data_base[filename] = {"steps": steps, "returns": returns}
save_to_pickle(sys.argv[2], {"pretrain": full_data_pretrain, "base": full_data_base})
print(full_data_base, full_data_pretrain)
