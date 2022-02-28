import torch
import gym
import sys
import os
import numpy as np
import pickle
import copy

from Baselines.HAC.HAC_args import get_args
from Baselines.HAC.HAC_agent import HAC, BreakoutHAC, Breakout2HAC, RobopushingHAC
from Baselines.HAC.HAC_collector import run_HAC
from Environments.environment_initializer import initialize_environment
from DistributionalModels.InteractionModels.samplers import BreakoutRandomSampler
from Rollouts.rollouts import ObjDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #################### Hyperparameters ####################
    # python train_HAC.py --specialized --reduced-state --num-episodes 30000 --pretrain-episodes 100 --env SelfBreakout --max-critic 200 --ctrl-type paddle --epsilon-close .5 --log-interval 1 --grad-epoch 10 --lr 3e-5 --epsilon .1 --policy-type pair
    # python train_HAC.py --specialized --reduced-state --num-episodes 30000 --pretrain-episodes 20 --env RoboPushing --max-critic 200 --epsilon-close .01 --log-interval 1 --grad-epoch 30 --lr 3e-5 --epsilon .1 --policy-type pair --drop-stopping --H 10 --ctrl-type block2 --time-cutoff 300 --learning-type sac
    print("pid", os.getpid())
    print(" ".join(sys.argv))
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    environment, environment_model, args = initialize_environment(args, set_save=len(args.record_rollouts) != 0, render="")

    # ACCOUNTED FOR
    # env_name = "MountainCarContinuous-h-v1"
    # save_episode = 10               # keep saving every n episodes
    # max_episodes = 1000             # max num of training episodes
    # random_seed = 0
    # render = False
    # env = gym.make(env_name)

    
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0] if not environment.discrete_actions else environment.action_space.n
    
    ### TODO: MOVE TO ARGS
    # """
    #  Actions (both primitive and subgoal) are implemented as follows:
    #    action = ( network output (Tanh) * bounds ) + offset
    #    clip_high and clip_low bound the exploration noise
    # """
    
    # # primitive action bounds and offset
    # action_bounds = env.action_space.high[0]
    # action_offset = np.array([0.0])
    # action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    # action_clip_low = np.array([-1.0 * action_bounds])
    # action_clip_high = np.array([action_bounds])
    
    # # state bounds and offset
    # state_bounds_np = np.array([0.9, 0.07])
    # state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    # state_offset =  np.array([-0.3, 0.0])
    # state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    # state_clip_low = np.array([-1.2, -0.07])
    # state_clip_high = np.array([0.6, 0.07])
    
    # # exploration noise std for primitive action and subgoals
    # exploration_action_noise = np.array([0.1])        
    # exploration_state_noise = np.array([0.02, 0.01]) 
    
    # goal_state = np.array([0.48, 0.04])        # final goal state to be achived
    # threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved
    
    # # HAC parameters:
    # k_level = 2                 # num of levels in hierarchy
    # H = 20                      # time horizon to achieve subgoal
    # lamda = 0.3                 # subgoal testing parameter
    
    # # DDPG parameters:
    # gamma = 0.95                # discount factor for future rewards
    # n_iter = 100                # update policy n_iter times in one DDPG update
    # batch_size = 100            # num of transitions sampled from replay buffer
    # lr = 0.001
    ### MOVE TO ARGS
    
    # save trained models
    directory = "/hdd/datasets/counterfactual_data/Baselines/HAC/preTrained/{}/{}level/".format(args.env, args.k_level) 
    filename = "HAC_{}".format(args.save_name) + "_" + str(args.ctrl_type)

    #########################################################
    
    max_steps = 200
    print(args.env)
    if args.env == "SelfBreakout":
        goal_based = False
        max_steps = 2000
        args.final_instanced = True
    elif args.env == "RoboPushing":
        args.final_instanced = True
        goal_based = True
    else:
        goal_based = True

    sampler = None
    if args.breakout_variant == "proximity":
        dataset_model = ObjDict({'delta': 0, 'cfselectors': []})
        sampler = BreakoutRandomSampler(environment_model=environment_model, dataset_model=dataset_model, no_combine_param_mask = True, init_state=None)
        environment.sampler = sampler
        # NOT goal based because hindsight block targeting is not supported

    if args.seed:
        print("Random Seed: {}".format(args.seed))
        # env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # creating HAC agent and setting parameters
    flat_state = environment_model.get_HAC_flattened_state(environment_model.get_state(), instanced=True, use_instanced=True)
    reduced_flat_state = environment_model.get_HAC_flattened_state(environment_model.get_state(), instanced=True, use_instanced=False)
    print(reduced_flat_state.shape)
    if args.env == "SelfBreakout": object_dim = environment_model.object_sizes["Block"]
    elif args.env == "RoboPushing": object_dim = environment_model.object_sizes["Obstacle"]
    else: object_dim = environment_model.object_sizes["State"]
    print(environment.name)
    if args.env == "SelfBreakout": goal_final = None if args.breakout_variant != "proximity" else sampler.sample(None)[0]
    elif args.env == "RoboPushing": goal_final = environment_model.get_state()['factored_state']["Target"]
    else: goal_final = np.array([0.48, 0.04]) # the goal state for mountaincar
    args.augmented_goal = False


    if args.env == "SelfBreakout" and args.specialized:
        if args.ctrl_type in ["paddle", "brel", "bvel", "bpos"]:
            agent = Breakout2HAC(args, args.k_level, args.H, args.epsilon_close, args.epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state,
                    environment_model.flat_rel_space, environment_model.reduced_flat_rel_space, object_dim=object_dim, sampler=sampler)
        elif args.ctrl_type in ["", "bvel3", "bpos3", "limit", "lrel"]:
            agent = BreakoutHAC(args, args.k_level, args.H, args.epsilon_close, args.epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state,
                    environment_model.flat_rel_space, environment_model.reduced_flat_rel_space, object_dim=object_dim, sampler=sampler)
        agent.final_threshold = agent.threshold
    elif args.env == "RoboPushing":
        agent = RobopushingHAC(args, args.k_level, args.H, args.epsilon_close, args.epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state,
                environment_model.flat_rel_space, environment_model.reduced_flat_rel_space, object_dim=object_dim, sampler=sampler)
        args.augmented_goal = True
        agent.final_threshold = np.array([.05, .05, .1])
    else:
        agent = HAC(args, args.k_level, args.H, args.epsilon_close, args.epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state,
                    environment_model.flat_rel_space, environment_model.reduced_flat_rel_space, object_dim=object_dim, sampler=sampler)
        agent.final_threshold = agent.threshold
    agent.set_parameters(args.lamda, args.gamma)
    if len(args.load_name) > 0:
        new_obs_space = copy.deepcopy(agent.HAC[-1].obs_space)
        loadname = "HAC_{}".format(args.load_name) + "_" + str(args.ctrl_type)
        agent.load(directory, loadname)
        agent.HAC[-1].obs_space = new_obs_space
        agent.HAC[-1].compute_mean_var()
        agent.cpu()
    agent.cuda(device="cuda:" + str(args.gpu), actor_lr = args.actor_lr, critic_lr = args.critic_lr)


    
    # logging file:
    log_f = open("log.txt","w+")

    # force_actions = dict()
    # for i in range(args.k_level):
    #     force_actions[i] = list()
    #     for j in range(1000):
    #         force_actions[i].append(np.random.uniform(agent.HAC[i].paction_space.low, agent.HAC[i].paction_space.high))
    # print(force_actions)
    # fid = open("../forceHAC.pkl", 'wb')
    # pickle.dump(force_actions, fid)
    # fid.close()
    # agent.set_epsilon_below(agent.k_level, 0)

    # training procedure
    agent.save(directory, filename)
    total_time = 0
    total_reward = 0
    pre_epsilon = agent.epsilon
    agent.epsilon = 1
    episode_scaling = 5 if args.env=="SelfBreakout" and args.drop_stopping else 1
    print(args.pretrain_episodes, episode_scaling)
    if args.top_level_random > 0:
        agent.top_level_random == True
    args.pretrain_episodes *= episode_scaling
    for i_episode in range(1, args.pretrain_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        full_state = environment.reset()
        if args.env == "RoboPushing":
            goal_final = full_state['factored_state']['Target']
        if sampler is not None:
            goal_final = sampler.sample(None)[0]
        next_state, reward, done, info, ep_time,reached = run_HAC(agent, environment_model, agent.k_level-1, full_state, goal_final, False,
                                 goal_based=goal_based, max_steps=max_steps, render=False, printout=args.printout, 
                                 reached=dict(), augmented_goal=args.augmented_goal, sampler=sampler)
        total_time += ep_time
        total_reward += reward
        if i_episode % episode_scaling == 0:
            print("Episode: {}\t Time: {}\t Reward: {}".format(i_episode // episode_scaling, total_time, total_reward))
            total_reward = 0
            if args.log_interval > 0 and i_episode % args.log_interval == 0 and args.printout:
                for k in range(agent.k_level):
                    print("buffer filled to ", agent.buffer_at[k][0], args.buffer_len)
                    printout_num=10
                    for j in range(printout_num):
                        idx = (agent.buffer_at[k][0] + (j - printout_num)) % args.buffer_len
                        print(k, idx, len(agent.replay_buffer[k]), agent.buffer_at[k][0])
                        print(k, j, agent.replay_buffer[k][idx])
            if args.log_interval > 0 and i_episode % args.log_interval == 0:
                print("reached", reached)


    agent.epsilon = pre_epsilon
    args.num_episodes *= episode_scaling
    for i_episode in range(args.pretrain_episodes+1, args.num_episodes+args.pretrain_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        if i_episode < args.top_level_random:
            agent.top_level_random = True
        else:
            agent.top_level_random = False
        
        full_state = environment.reset()
        if args.env == "RoboPushing":
            goal_final = full_state['factored_state']['Target']
        # collecting experience in environment
        if sampler is not None:
            goal_final = sampler.sample(None)[0]
        next_state, reward, done, info, ep_time,reached = run_HAC(agent, environment_model, agent.k_level-1, full_state, goal_final, 
            False, goal_based=goal_based, max_steps=max_steps, render=False, printout=args.printout, 
            reached=dict(), augmented_goal=args.augmented_goal, sampler=sampler)
        total_time += ep_time
        total_reward += reward
        # update all levels
        losses_dict = agent.update(args.batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, reward))
        log_f.flush()
        
        if args.save_interval > 0 and i_episode % args.save_interval == 0:
            agent.save(directory, filename)
        
        if i_episode % episode_scaling == 0:
            print("Episode: {}\t Time: {}\t Reward: {}".format(i_episode // episode_scaling, total_time, total_reward))
            total_reward = 0
            if args.log_interval > 0 and i_episode % args.log_interval == 0 and args.printout:
                for k in range(agent.k_level):
                    print("buffer filled to ", agent.buffer_at[k][0], args.buffer_len)
                    printout_num=10
                    for j in range(printout_num):
                        idx = (agent.buffer_at[k][0] + (j - printout_num)) % args.buffer_len
                        print(k, idx, len(agent.replay_buffer[k]), agent.buffer_at[k][0])
                        print(k, j, agent.replay_buffer[k][idx])
            if args.log_interval > 0 and i_episode % args.log_interval == 0:
                print("losses", losses_dict, "reached", reached)
    
if __name__ == '__main__':
    train()