import numpy as np
from tianshou.data import Batch
from Networks.network import pytorch_model
import copy
import pickle
import cv2

def update_full_data(data, obs, act, mapped_act,
    obs_next, param, next_target, rew, gamma, done, info):
    ed = lambda x: np.expand_dims(x, 0)
    data.update(obs=ed(obs), act=act, mapped_act=ed(mapped_act), obs_next=ed(obs_next), param=ed(param), 
        next_target=ed(next_target), rew=rew, gamma=[gamma], done=done, info=info)

def print_data(data):
    return " ".join([str(data.obs), str(data.act), str(data.rew), str(data.obs_next), str(data.param), str(data.next_target), str(data.gamma), str(data.done), str(data.mapped_act)]) 

def run_HAC(agent, env_model, i_level, full_state, goal, is_subgoal_test, goal_based=True, max_steps=0, render=False, printout=False, reached={}, augmented_goal = False, sampler=None):
    next_state = None
    done = None
    goal_transitions = []

    subgoal_test_transitions = list()
    data = Batch(
        obs={}, param={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}, gamma={}
    )
    reward = 0.0
    total_time = 0
    if goal is None: # the goal is not used at this level
        goal = np.array(0) # use a dummy goal
    # logging updates

    fid = open("../forceHAC.pkl", 'rb') # FAX
    force_actions = pickle.load(fid)
    fid.close()

    # H attempts
    data.update(full_state=full_state)
    if i_level == agent.k_level-1: # we are in the top level
        rng = max_steps
    else: rng = agent.H
    for i in range(rng): # TODO: the top level should run until episode end/large number of steps
        # if this is a subgoal test, then next/lower level goal has to be a subgoal test
        is_next_subgoal_test = is_subgoal_test
        
        # print(goal, i_level)
        obs = agent.get_obs(i_level, full_state, goal, env_model)
        target_lower = agent.get_target(i_level-1, full_state, env_model) if i_level > 0 else None
        # print(i_level, obs.shape, goal_based)
        # print(obs.shape, target_lower.shape)
        act, action = agent.HAC[i_level].select_action(obs, target_lower)
        sa, sent_goal = act, action
        # if i_level > 0:
        #     print(i_level, act, action)
        # action = force_actions[i_level][i] # FAX
        # if i_level == 1:
        #     # print(i_level, action, act, agent.HAC[i_level].relative_action)
        #     # act = agent.HAC[i_level].reverse_map_action(action, target_lower)
        #     # print(act)
        #     print(sampler.param, obs[:5])

        if printout:
            print("retargeted   ", i, i_level, act, action)
        # if np.random.rand() < .01:
        # act = pytorch_model.unwrap(act[0])
        # action = pytorch_model.unwrap(tensor_action[0])
        
        #   <================ high level policy ================>
        if i_level > 0:
            # add noise or take random action if not subgoal testing
            # if is_subgoal_test:
            #     agent.set_epsilon_below(i_level, 0)
            # else:
            #     agent.set_epsilon_below(i_level, agent.epsilon)

            if not is_subgoal_test:
                if np.random.random_sample() < agent.epsilon or (agent.top_level_random and i_level == agent.k_level -1):
                  action = np.random.uniform(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                  act = agent.HAC[i_level].reverse_map_action(action, target_lower)
                else:
                  noise = np.random.normal(0, agent.epsilon, size=agent.HAC[i_level].action_space.shape)
                  act = act + noise
                  action = agent.HAC[i_level].map_action(act, target_lower)
                  action = action.clip(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                  # print(noise, act, action)

            # print("retargeted   ", i, i_level, act, action)
            # Determine whether to test subgoal (action)
            if np.random.random_sample() < agent.lamda:
                is_next_subgoal_test = True
            
            # Pass subgoal to lower level 
            next_full_state, rew, done, info, tim, reached = run_HAC(agent, env_model, i_level-1, full_state, action, is_next_subgoal_test, render=render, printout=printout, reached=reached)
            reward += rew
            total_time += tim
            data.update(next_full_state=next_full_state)
            next_target_lower = agent.get_target(i_level-1, next_full_state, env_model)

            if sampler is not None:
                print(action, rew)
                if rew != 0:
                    sampler.sample(None)
                    done = True
            # if subgoal was tested but not achieved, add subgoal testing transition
            if is_next_subgoal_test and not agent.check_goal(action, next_target_lower, agent.threshold):
                # states, actions, rewards, next_states, goals, gamma, dones
                obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
                subgoal_data = copy.deepcopy(data)
                next_target = agent.get_target(i_level, next_full_state, env_model)
                info["TimeLimit.truncated"] = False
                update_full_data(subgoal_data, obs=obs, act=act, mapped_act=action, rew=-agent.H, obs_next=obs_next,
                 param=goal, next_target=next_target, gamma=0.0, done=True, info=info)
                if printout: print("subgoal transition", i, i_level, print_data(subgoal_data))
                subgoal_test_transitions.append(subgoal_data)
                # agent.set_epsilon_below(i_level, agent.epsilon)
            
            # for hindsight action transition
            action = next_target_lower
            act = agent.HAC[i_level].reverse_map_action(action, target_lower)
            # print(i_level, action, act)

        #   <================ low level policy ================>
        else:
            # add noise or take random action if not subgoal testing
            # if is_subgoal_test:
            #     agent.set_epsilon_below(i_level, 0)
            # else:
            #     agent.set_epsilon_below(i_level, agent.epsilon)
            if not is_subgoal_test:
                if agent.primitive_action_discrete:
                    if np.random.random_sample() < agent.epsilon: # epsilon rate
                      action = np.random.randint(agent.max_action)
                      act = agent.HAC[i_level].reverse_map_action(action, None)
                else:
                    if np.random.random_sample() > 0.2:
                      action = np.random.uniform(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                      act = agent.HAC[i_level].reverse_map_action(action, None)
                    else:
                      action = action + np.random.normal(0, agent.epsilon, size=agent.HAC[i_level].paction_space.shape)
                      action = action.clip(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                      act = agent.HAC[i_level].reverse_map_action(action, None)
            # take primitive action
            next_full_state, rew, done, info = env_model.step(action)
                    # flush goal transitions
            # cv2.imshow('frame',next_full_state['raw_state'])
            # if cv2.waitKey(10):
            #     pass

            if env_model.environment.name == "RobosuitePushing" and rew > -1.9:
                rew = -0.1

            # print("env done", done)
            data.update(next_full_state=next_full_state)
            
            if render:
                
                env.render() ##########
                
            # this is for logging
            agent.reward += rew
            reward += rew
            total_time += 1
        # if np.random.rand()<.002 and i_level==0 or np.random.rand()<.004 and i_level==1 or np.random.rand()<.2 and i_level==2:
        #     print("retargeted   ", i, i_level, act, action, full_state['factored_state']['Paddle'][1], full_state['factored_state']['Ball'][2:4], goal, goal_based, sent_goal, sa)
        #     print(agent.HAC[i_level].compute_Q(Batch(obs=np.expand_dims(obs, 0), info=dict(), act=np.expand_dims(act, 0)), False))
        #   <================ finish one step/transition ================>
        goal_achieved = False
        if goal_based:
            # check if goal is achieved
            next_target = agent.get_target(i_level, next_full_state, env_model)
            goal_achieved = agent.check_goal(next_target, goal, agent.threshold)
            # print("comparison reached", i_level, goal, next_target, goal_achieved)
            # info["TimeLimit.truncated"] = False

            # add values
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
            # print(i_level, full_state['raw_state'], next_full_state['raw_state'], obs, obs_next)

            # hindsight action transition

            update_full_data(data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew =-1.0, gamma=agent.gamma, done=done, info=info)
            if goal_achieved:
                if i_level in reached:
                    reached[i_level] += 1
                else:
                    reached[i_level] = 1
                data.update(rew=0.0, done=True)
                # print("reached transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
            else:
                data.update(rew=-1.0)
                # if np.random.rand() < .001: print("base transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
                
            # copy for goal transition
            goal_data = copy.deepcopy(data)
            update_full_data(goal_data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew=-1.0, gamma=agent.gamma, done=done, info=info)
            goal_transitions.append(goal_data)


        ##### AUGMENTED GOALS ARE FOR ROBOPUSHING #### 
        elif augmented_goal and i_level == agent.k_level - 1:
            # check if goal is achieved
            next_target = agent.get_target(i_level, next_full_state, env_model)
            goal_achieved = agent.check_goal(next_target, goal, agent.final_threshold)
            # print("comparison reached", i_level, goal, next_target, goal_achieved)
            
            # add values
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
            # print(i_level, full_state['raw_state'], next_full_state['raw_state'], obs, obs_next)

            # hindsight action transition
            update_full_data(data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew =-1.0, gamma=agent.gamma, done=done, info=info)
            if goal_achieved:
                if i_level in reached:
                    reached[i_level] += 1
                else:
                    reached[i_level] = 1
                data.update(rew=0.0, done=True)
                # print("reached transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
            else:
                data.update(rew=rew)
                # if np.random.rand() < .001: print("base transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
                
            # copy for goal transition
            goal_data = copy.deepcopy(data)
            update_full_data(goal_data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew=rew, gamma=agent.gamma, done=done, info=info)
            goal_transitions.append(goal_data)
        #### NO GOALS IS FOR MOST BREAKOUT ENVIRONMENTS ####
        else: # add a transition with the environment reward
            # print("reward", rew)
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
            next_target = agent.get_target(i_level, next_full_state, env_model)
            # info["TimeLimit.truncated"] = False
            update_full_data(data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew =rew, gamma=agent.gamma, done=done, info=info)
            if printout: print("rew transition", i, i_level, print_data(data))

            agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
        # if printout:
        #     target = agent.get_target(i_level, full_state, env_model)
        #     if printout: print("retargeted", i, i_level, act, action, target, next_target, goal)
        
        full_state = next_full_state
        data.update(full_state=full_state)
        
        if done or goal_achieved:
            # print("reached", done, goal_achieved)
            break

    for transition in subgoal_test_transitions:
        agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(transition)
    
    
    #   <================ finish H attempts ================>
    
    target_rew = 1
    if goal_based:
        # hindsight goal transition
        # last transition reward and discount is 0
        hit_rew = 0.0
        if sampler is not None: # special logic for transitions
            if -1 <= goal_transitions[-1].rew <= 1:
                hit_rew = 1.0
            else:
                return next_full_state, reward, done, info, total_time, reached
        goal_transitions[-1].update(rew=hit_rew)
        goal_transitions[-1].update(done= True)
        param = agent.get_target(i_level, goal_transitions[-1].next_full_state, env_model)
        for g, transition in enumerate(goal_transitions):
            # last state is goal for all transitions
            obs = agent.get_obs(i_level, transition.full_state, param, env_model)
            obs_next = agent.get_obs(i_level, transition.next_full_state, param, env_model)
            if sampler is not None:
                rew = 0.0
                gamma = [0.0]
                hindsight_done = False
            else:
                goal_check = agent.check_goal(transition.next_target[0], param, agent.threshold)
                rew = 0.0 if goal_check else transition.rew
                gamma = [0.0] if goal_check else transition.gamma
                hindsight_done = True if goal_check else transition.done
            transition.update(obs=np.expand_dims(obs, 0), obs_next=np.expand_dims(obs_next, 0), param=np.expand_dims(param, 0), rew=rew, gamma=gamma, done=hindsight_done)
            if printout: print("goal transition", g, i_level, print_data(transition))
            agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(transition)
    return next_full_state, reward, done, info, total_time, reached
