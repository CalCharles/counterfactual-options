from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]

tmr = "Test mean returns: "
ase = "Assess: "
aset = "Assess Train: "
second_to_last = False
print(pth, pth.find("center"))
if pth.find("center") != -1 or pth.find("prio") != -1:
    ase = aset
    second_to_last = True

ignore = ["train_negative_rand_row8.txt", "train_negative_rand_row10.txt"]

assesses = list()
for filename in os.listdir(pth):
    assess = 0
    last_assess = 0
    if filename in ignore:
        continue
    for line in open(os.path.join(pth, filename), 'r'):
        if line.find(tmr) != -1:
            last_assess = assess
            print(line.find(ase))
            print(line[line.find(ase) + len(ase):].split(" "))
            assess = float(line[line.find(ase) + len(ase):].split(" ")[0])
            print(last_assess, assess)
    if second_to_last:
        assesses.append(last_assess)
    else:
        asseses.append(assess)
print(assesses)
print("main", np.mean(assesses), np.std(assesses))
