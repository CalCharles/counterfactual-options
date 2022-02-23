import numpy as np
from tianshou.data import Batch
from Networks.network import pytorch_model
import copy

def update_full_data(data, obs, act, mapped_act,
    obs_next, param, next_target, rew, gamma, done, info):
    ed = lambda x: np.expand_dims(x, 0)
    data.update(obs=ed(obs), act=act, mapped_act=ed(mapped_act), obs_next=ed(obs_next), param=ed(param), 
        next_target=ed(next_target), rew=rew, gamma=[gamma], done=done, info=info)



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
    
    # H attempts
    data.update(full_state=full_state)
    if i_level == agent.k_level-1: # we are in the top level
        rng = max_steps
    else: rng = agent.H
    for i in range(rng): # TODO: the top level should run until episode end/large number of steps
        # if this is a subgoal test, then next/lower level goal has to be a subgoal test
        is_next_subgoal_test = is_subgoal_test
        
        obs = agent.get_obs(i_level, full_state, goal, env_model)
        act, action = agent.HAC[i_level].select_action(obs)
        if i_level != 0 and printout:
            print(i_level, act, action, agent.HAC[i_level].paction_space.low, agent.HAC[i_level].paction_space.high)
        # act = pytorch_model.unwrap(act[0])
        # action = pytorch_model.unwrap(tensor_action[0])
        
        #   <================ high level policy ================>
        if i_level > 0:
            # add noise or take random action if not subgoal testing
            if is_subgoal_test:
                agent.set_epsilon_below(i_level, 0)
            
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
                subgoal_test_transitions.append(subgoal_data)
                agent.set_epsilon_below(i_level, agent.epsilon)
            
            # for hindsight action transition
            action = next_target
            act = agent.HAC[i_level].reverse_map_action(action)
            
        #   <================ low level policy ================>
        else:
            # add noise or take random action if not subgoal testing
            if is_subgoal_test:
                agent.set_epsilon_below(i_level, 0) # sets levels below EXCEPT with the bottom
            else:
                agent.set_epsilon_below(i_level, agent.epsilon) # sets levels below EXCEPT with the bottom

            # take primitive action
            next_full_state, rew, done, info = env_model.step(action)
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
            
            # add values
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
            # print(i_level, full_state['raw_state'], next_full_state['raw_state'], obs, obs_next)

            # hindsight action transition
            update_full_data(data, obs=obs, act=act, mapped_act=action, obs_next=obs_next, 
                param=goal, next_target=next_target, rew =-1.0, gamma=agent.gamma, done=done, info=info)
            if goal_achieved:
                data.update(rew=0.0, done=True)
                agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
            else:
                data.update(rew=-1.0)
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
            agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(data)
        if printout:
            target = agent.get_target(i_level, full_state, env_model)
            print(i, rng, i_level, act, action, target, next_target, goal)
        
        full_state = next_full_state
        data.update(full_state=full_state)
        
        if done or goal_achieved:
            # print("done", done, goal_achieved)
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
            done = True if goal_check else transition.done
            transition.update(obs=np.expand_dims(obs, 0), obs_next=np.expand_dims(obs_next, 0), param=np.expand_dims(param, 0), rew=rew, done=done)
            agent.buffer_at[i_level], ep_rew, ep_len, ep_idx = agent.replay_buffer[i_level].add(transition)
    return next_full_state, reward, done, info, total_time
