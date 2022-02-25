import numpy as np
from tianshou.data import Batch
from Networks.network import pytorch_model
import copy
import pickle

def update_full_data(data, obs, act, mapped_act,
    obs_next, param, next_target, rew, gamma, done, info):
    ed = lambda x: np.expand_dims(x, 0)
    data.update(obs=ed(obs), act=act, mapped_act=ed(mapped_act), obs_next=ed(obs_next), param=ed(param), 
        next_target=ed(next_target), rew=rew, gamma=[gamma], done=done, info=info)

def print_data(data):
    return " ".join([str(data.obs), str(data.act), str(data.rew), str(data.obs_next), str(data.param), str(data.gamma), str(data.done), str(data.mapped_act)]) 

def run_HAC(agent, env_model, i_level, full_state, goal, is_subgoal_test, goal_based=True, max_steps=0, render=False, printout=False):
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
        
        obs = agent.get_obs(i_level, full_state, goal, env_model)
        # print(i_level, obs.shape, goal_based)
        act, action = agent.HAC[i_level].select_action(obs)
        
        # action = force_actions[i_level][i] # FAX
        act = agent.HAC[i_level].reverse_map_action(action)
        
        if printout:
            print("retargeted   ", i, i_level, act, action, agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
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
                if np.random.random_sample() < agent.epsilon:
                  action = np.random.uniform(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                  act = agent.HAC[i_level].reverse_map_action(action)
                else:
                  action = action + np.random.normal(0, agent.epsilon, size=agent.HAC[i_level].paction_space.shape)
                  action = action.clip(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)

            # Determine whether to test subgoal (action)
            if np.random.random_sample() < agent.lamda:
                is_next_subgoal_test = True
            
            # Pass subgoal to lower level 
            next_full_state, rew, done, info, tim = run_HAC(agent, env_model, i_level-1, full_state, action, is_next_subgoal_test, render=render, printout=printout)
            reward += rew
            total_time += tim
            data.update(next_full_state=next_full_state)
            next_target = agent.get_target(i_level, next_full_state, env_model)

            # if subgoal was tested but not achieved, add subgoal testing transition
            if is_next_subgoal_test and not agent.check_goal(action, next_target, agent.threshold):
                # states, actions, rewards, next_states, goals, gamma, dones
                obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
                subgoal_data = copy.deepcopy(data)
                update_full_data(subgoal_data, obs=obs, act=act, mapped_act=action, rew=-agent.H, obs_next=obs_next, param=goal, next_target=next_target, gamma=0.0, done=True, info=info)
                if printout: print("subgoal transition", i, i_level, print_data(subgoal_data))
                subgoal_test_transitions.append(subgoal_data)
                # agent.set_epsilon_below(i_level, agent.epsilon)
            
            # for hindsight action transition
            action = next_target
            act = agent.HAC[i_level].reverse_map_action(action)
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
                      act = agent.HAC[i_level].reverse_map_action(action)
                else:
                    if np.random.random_sample() > 0.2:
                      action = np.random.uniform(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
                      act = agent.HAC[i_level].reverse_map_action(action)
                    else:
                      action = action + np.random.normal(0, agent.epsilon, size=agent.HAC[i_level].paction_space.shape)
                      action = action.clip(agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)


            # take primitive action
            next_full_state, rew, done, info = env_model.step(action)
            # print("env done", done)
            data.update(next_full_state=next_full_state)
            
            if render:
                
                env.render() ##########                    
                
            # this is for logging
            agent.reward += rew
            reward += rew
            total_time += 1
        #   <================ finish one step/transition ================>
        goal_achieved = False
        if goal_based:
            # check if goal is achieved
            next_target = agent.get_target(i_level, next_full_state, env_model)
            goal_achieved = agent.check_goal(next_target, goal, agent.threshold)
            # print("comparison reached", i_level, goal, next_target, goal_achieved)
            
            # add values
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
            # print(i_level, full_state['raw_state'], next_full_state['raw_state'], obs, obs_next)

            # hindsight action transition
            update_full_data(data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew =-1.0, gamma=agent.gamma, done=done, info=info)
            if goal_achieved:
                data.update(rew=0.0, done=True)
                if printout: print("base transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
            else:
                data.update(rew=-1.0)
                if printout: print("base transition", i, i_level, print_data(data))
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
                
            # copy for goal transition
            goal_data = copy.deepcopy(data)
            update_full_data(goal_data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew=-1.0, gamma=agent.gamma, done=done, info=info)
            goal_transitions.append(goal_data)
        else: # add a transition with the environment reward
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
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
    
    if goal_based:
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1].update(rew=0.0)
        goal_transitions[-1].update(done= True)
        param = agent.get_target(i_level, goal_transitions[-1].next_full_state, env_model)
        for transition in goal_transitions:
            # last state is goal for all transitions
            obs = agent.get_obs(i_level, transition.full_state, param, env_model)
            obs_next = agent.get_obs(i_level, transition.next_full_state, param, env_model)
            goal_check = agent.check_goal(transition.next_target[0], param, agent.threshold)
            rew = 0.0 if goal_check else transition.rew
            gamma = [0.0] if goal_check else transition.gamma
            hindsight_done = True if goal_check else transition.done
            transition.update(obs=np.expand_dims(obs, 0), obs_next=np.expand_dims(obs_next, 0), param=np.expand_dims(param, 0), rew=rew, gamma=gamma, done=hindsight_done)
            if printout: print("goal transition", i, i_level, print_data(transition))
            agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(transition)
    return next_full_state, reward, done, info, total_time
