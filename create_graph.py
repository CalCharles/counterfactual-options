# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os
import math
import argparse
from file_management import load_from_pickle
import copy
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

'''
USAGE:
python visualize.py trained_models/pongW2pongH/ --plotall
OR
python visualize.py trained_models/pong_vanilla/ trained_models/pongH2pong
'''

def compute_error(results, default_std=1, maxrange=1e6, use_episode=False, increment=0, scaled=-1, decre=0, mean_window=True):
    meanvals = list()
    stdvs = list()
    steps = list()
    keyword = "returns"
    if use_episode:
        keyword = "episode_rewards"
    window = deque(maxlen=5)

    for i in [j*50000 for j in range(int(maxrange // 50000 + 1))]:
        compiled_returns = list()
        for fn in results.keys():
            ri = 0
            lastv = results[fn][keyword][-1]
            print(results[fn])
            while ri < len(results[fn]["steps"]):
                if abs(results[fn]["steps"][ri] - i) <= 25000:
                    val = results[fn][keyword][ri]
                    if scaled > 0:
                        if val > 0:
                            val = val /scaled
                    val = val - decre
                    # if mean_window:
                    #     window.append(val)
                    #     val = np.mean(window)
                    compiled_returns.append(val)
                    print(results[fn][keyword][ri])
                ri += 1
        if len(compiled_returns) > 0:
            meanvals.append(np.mean(compiled_returns))
            print(i, compiled_returns)
            if len(compiled_returns) > 0:
                stdvs.append(np.std(compiled_returns)/2)
            else:
                stdvs.append(default_std)
            steps.append(i + increment)
    return steps, meanvals, stdvs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--log-dir', metavar='log_dir', nargs='+')
    parser.add_argument('--model', default='a2c')
    parser.add_argument('--title', default='Graph')
    parser.add_argument('--plotall', action = 'store_true', default=False)
    parser.add_argument('--labels', metavar='labels', nargs='+')
    parser.add_argument('--xlim', type=float, default=10e5)
    parser.add_argument('--yrng', type=float, default=[-1,1], nargs='+')
    parser.add_argument('--default-std', type=float, default=1)
    parser.add_argument('--target', default='plot.png')
    args = parser.parse_args()

    use_episode = False
    no_transfer = False
    no_shift = False
    increment = 0
    scaled = -1
    decre=0
    if args.log_dir[0].find("reward") != -1:
        use_episode = True
        increment = 200000
    if args.log_dir[0].find("robo") != -1:
        increment = 600000
        no_shift = True
        no_transfer = True
    
    if args.log_dir[0].find("prox") != -1:
        no_transfer = True
        scaled = 20
    # if args.log_dir[0].find("neg") != -1:
    #     scaled = 5
    HOOD_results = load_from_pickle(args.log_dir[0])
    baseline_results = load_from_pickle(args.log_dir[1])
    baseline_results, pretrain_results = baseline_results["base"], baseline_results["pretrain"]
    HAC_results = load_from_pickle(args.log_dir[2]) if len(args.log_dir) > 2 else None

    # single python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/singles.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/singles.pkl --yrng -5 0 --xlim 500000 --title "Single Block Minimum Bounce" --target graphs/singles.png
    # center fine python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/centers.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/centers.pkl --yrng -50 2 --xlim 2000000 --title "Center Blocks Indestructible" --target graphs/centers.png
    # prox python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/proxs.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/proxs.pkl --yrng -1 .6 --xlim 1000000 --title "Block Targeting Proximity Reward" --target graphs/proxs.png
    # neg python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/negatives.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/negatives.pkl --yrng -3 1.5 --xlim 1000000 --title "Randomly Assigned Negative Reward Blocks" --target graphs/negatives.png
    # hard python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/hardens.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/harden.pkl --yrng -10 0 --xlim 1000000 --title "Single Block Minimum Bounce with 10 Obstacle Blocks" --target graphs/harden.png
    # big fine python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/bigs.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/bigs.pkl --yrng -20 0 --xlim 500000 --title "Big Block Domain" --target graphs/big.png
    # default fine python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/reward.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/reward.pkl /hdd/datasets/counterfactual_data/graph_pkls/HAC/rewards.pkl --yrng -50 100 --xlim 2000000 --title "Breakout" --target graphs/breakout.png
    # robo python create_graph.py --log-dir /hdd/datasets/counterfactual_data/graph_pkls/robopushing.pkl /hdd/datasets/counterfactual_data/graph_pkls/baselines/robos.pkl /hdd/datasets/counterfactual_data/graph_pkls/HAC/robopushing.pkl --yrng -50 0 --xlim 2000000 --title "Robotic Pushing with Negative Reward Regions" --target graphs/robopush.png

    def plot(results, name, ci, use_episode=False, increment=0, scaled=-1, decre=0, mean_window = False):
        steps, meanvals, stdvs = compute_error(results, maxrange=args.xlim, use_episode=use_episode, increment=increment, scaled=scaled, decre=decre, mean_window=mean_window)
        steps = np.array(steps)
        returns = np.array(meanvals)
        error = np.array(stdvs)
        print(len(steps), len(returns))
        plt.plot(steps, returns, label=name, color=color_defaults[ci])
        plt.fill_between(steps, returns+error, returns-error, alpha=0.1, color=color_defaults[ci])
        print(returns)
        if len(returns.shape) > 0:
            return np.min(returns), np.max(returns)
        return None, None
    minrtHO, maxrtHO = plot(HOOD_results, "HOOD", 0, use_episode=use_episode, increment=increment, )
    minrtbase, maxrtbase = plot(baseline_results, "Base", 1, scaled = scaled, decre=decre, mean_window=True)
    baseline_shifted_results = copy.deepcopy(baseline_results)
    for f in baseline_shifted_results.keys():
        baseline_shifted_results[f]["steps"] = [s-1e6 for s in baseline_shifted_results[f]["steps"]]
    if no_transfer and not no_shift: minrtsht, maxrtsht = plot(baseline_shifted_results, "Shift", 2, scaled=scaled, decre=decre, mean_window=True)
    if not use_episode and not no_transfer: minrtft, maxrtft = plot(pretrain_results, "FT", 2, scaled=scaled, decre=decre, mean_window=True)
    if HAC_results is not None: plot(HAC_results, "HAC", 3)
    # miny = np.min([minrtHO, minrtbase, minrtft])
    # miny = np.max([args.ymin, miny])
    # maxy = np.max([maxrtHO, maxrtbase, maxrtft])

    # plt.fill_between(tx, y_mean+y_err, y_mean-y_err, alpha=0.1, color=color_defaults[m_idx])
    # plt.plot([0, xlim], [244, 244], linewidth =2, color = color_defaults[7], label="HyPE Test Performance at 55k frames ")
    # plt.plot([0, xlim], [17.5, 17.5], linewidth =2, color = color_defaults[7], label="HyPE Test Performance at 55k frames ")
    # plt.plot([2000, 2000], [-100, 100], linewidth =1, color = color_defaults[4])
    if use_episode: plt.plot([200000, 200000], [-100.0, 100], linewidth =1, color = color_defaults[5])
    if no_shift: plt.plot([600000, 600000], [-100.0, 100], linewidth =1, color = color_defaults[6])
    # plt.plot([1500000, 1500000], [-0.0, 20], linewidth =1, color = color_defaults[7])
    # plt.xticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 10e5], ["100k", "200k", "300k", "400k", "500k", "600k", "700k", "800k", "900k", "1m"])
    plt.xticks([1e5, 4e5, 7e5, 10e5, 13e5, 15e5, 18e5, 20e5], ["100k", "400k", "700k", "1M", "1.3M", "1.5M", "1.7M", "2M"])
    # plt.xticks([0, 10e5, 20e5, 30e5, 40e5, 50e5, 60e5], ["0", "1m", "2m", "3m", '4m', '5m', '6m'])
    # plt.xticks([1e3, 1e4, 1e5, 5e5, 1e6, 5e6], ["1k", "10k", "100k", "500k", "1m", "5m"])
    # xlim = min(xlim, args.xlim)
    # plt.xlim(0, math.ceil(xlim/1e6)*1e6)
    plt.xlim(0, args.xlim)
    plt.ylim(args.yrng[0], args.yrng[1])
    # plt.ylim(0, 270)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Rewards per Episode')
    plt.title(args.title)
    # plt.legend(loc=2)
    # plt.figure(figsize = (600, 200))
    plt.savefig(args.target)