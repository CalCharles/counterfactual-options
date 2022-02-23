import torch
import gym
import sys
import os
import numpy as np
from Baselines.HAC.HAC_args import get_args
from Baselines.HAC.HAC_agent import HAC
from Baselines.HAC.HAC_collector import run_HAC
from Environments.environment_initializer import initialize_environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #################### Hyperparameters ####################
    print("pid", os.getpid())
    print(" ".join(sys.argv))
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    environment, environment_model, args = initialize_environment(args, set_save=len(args.record_rollouts) != 0, render=False)

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
    filename = "HAC_{}".format(args.env)
    #########################################################
    
    max_steps = 200
    if args.env == "SelfBreakout":
        goal_based = False
        max_steps = 3000
    else:
        goal_based = True
    
    if args.seed:
        print("Random Seed: {}".format(args.seed))
        # env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # creating HAC agent and setting parameters
    flat_state = environment_model.get_HAC_flattened_state(environment_model.get_state(), use_instanced=True)
    reduced_flat_state = environment_model.get_HAC_flattened_state(environment_model.get_state(), use_instanced=False)
    if args.env == "SelfBreakout": object_dim = environment_model.object_sizes["Block"]
    elif args.env == "RoboPushing": object_dim = environment_model.object_sizes["Obstacle"]
    else: object_dim = environment_model.object_sizes["State"]
    print(environment.name)
    if args.env == "SelfBreakout": goal_final = None 
    elif args.env == "RoboPushing": environment_model.goal_function
    else: goal_final = np.array([0.48, 0.04]) # the goal state for mountaincar
    agent = HAC(args, args.k_level, args.H, args.epsilon_close, args.epsilon, environment, environment_model, goal_final, flat_state, reduced_flat_state,
    environment_model.flat_rel_space, environment_model.reduced_flat_rel_space, object_dim=object_dim)
    agent.set_parameters(args.lamda, args.gamma)
    
    # logging file:
    log_f = open("log.txt","w+")
    
    # training procedure 
    for i_episode in range(1, args.num_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        
        full_state = environment.reset()
        # collecting experience in environment
        next_state, reward, done, info, total_time = run_HAC(agent, environment_model, agent.k_level-1, full_state, goal_final, False, goal_based=goal_based, max_steps=max_steps, render=False, printout=args.printout)
        
        # update all levels
        losses_dict = agent.update(args.batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, reward))
        log_f.flush()
        
        if args.save_interval > 0 and i_episode % args.save_interval == 0:
            agent.save(directory, filename)
        
        print("Episode: {}\t Reward: {}".format(i_episode, reward))
        if args.log_interval > 0 and i_episode % args.log_interval == 0 and args.printout:
            for k in range(agent.k_level):
                for j in range(40):
                    idx = (agent.buffer_at[k][0] + (j - 100)) % args.buffer_len
                    print(k, idx, len(agent.replay_buffer[k]), agent.buffer_at[k][0])
                    print(k, j, agent.replay_buffer[k][idx])

            print("losses", losses_dict)
    
if __name__ == '__main__':
    train()