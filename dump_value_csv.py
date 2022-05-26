from file_management import save_to_pickle
import os
import sys
import numpy as np

pth = sys.argv[1]

if pth.find("csv_data") != -1:
    res_dict = dict()
    res_dict["pretrain"] = list()
    res_dict["base"] = list()
    for filename in os.listdir(pth):

        assess = 0
        for line in open(os.path.join(pth, filename), 'r'):
            try:
                print(line.split(",")[3])
                assess = float(line.split(",")[3])
                print(assess)
            except ValueError as e:
                continue
        if filename.find("pretrain") != -1:
            res_dict["pretrain"].append(assess)
        else:
            res_dict["base"].append(assess)
        for n in ["pretrain", "base"]:
            print(n, np.mean(res_dict[n]), np.std(res_dict[n]))
else:
    ase = "Mean assessment over each episode: "
    dirnames = list()
    for dirname in os.listdir(pth):
        assesses = list()
        for filename in os.listdir(os.path.join(pth, dirname)):
            assess = 0
            for line in open(os.path.join(pth, dirname, filename), 'r'):
                if line.find(ase) != -1:
                    # print(line.find(ase))
                    # print(line[line.find(ase) + len(ase):].split(" "))
                    assess = float(line[line.find(ase) + len(ase):-1])
            assesses.append(assess)
        print(dirname, np.mean(assesses), np.std(assesses))

