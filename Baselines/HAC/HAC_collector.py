import numpy as np    


def run_HAC(agent, env_model, i_level, full_state, goal, is_subgoal_test, goal_based=True, max_steps=0, render=False):
    next_state = None
    done = None
    goal_transitions = []
    subgoal_test_transitions = list()
    data = Batch(
        obs={}, param={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}, gamma={}
    )
    reward = 0.0
    total_time = 0
    # logging updates
    
    # H attempts
    data.update(full_state=full_state)
    if i_level == agent.k_level-1: # we are in the top level
        rng = max_steps
    else: rng = agent.H

    for _ in range(rng): # TODO: the top level should run until episode end/large number of steps
        # if this is a subgoal test, then next/lower level goal has to be a subgoal test
        is_next_subgoal_test = is_subgoal_test
        
        action = agent.HAC[i_level].select_action(state, goal)
        
        #   <================ high level policy ================>
        if i_level > 0:
            # add noise or take random action if not subgoal testing
            if not is_subgoal_test:
                if np.random.random_sample() > 0.2:
                  action = action + np.random.normal(0, agent.exploration_state_noise)
                  action = action.clip(agent.state_clip_low, agent.state_clip_high)
                else:
                  action = np.random.uniform(agent.state_clip_low, agent.state_clip_high)
            
            # Determine whether to test subgoal (action)
            if np.random.random_sample() < agent.lamda:
                is_next_subgoal_test = True
            
            # Pass subgoal to lower level 
            next_full_state, rew, done, info, tim = run_HAC(agent, env_model, i_level-1, state, action, is_next_subgoal_test, render=render)
            reward += rew
            total_time += tim
            data.update(next_full_state=next_full_state)
            next_target = agent.get_target(i_level, next_full_state, environment_model)

            # if subgoal was tested but not achieved, add subgoal testing transition
            if is_next_subgoal_test and not agent.check_goal(action, next_target, agent.threshold):
                # states, actions, rewards, next_states, goals, gamma, dones
                obs = agent.get_obs(i_level, full_state, goal, env_model)
                obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)
                data.update(obs=state, act=action, rew=-agent.H, obs_next=obs_next, param=goal, gamma=0.0, done=True, info=info)
                subgoal_test_transitions.append(copy.deepcopy(data))
            
            # for hindsight action transition
            action = next_target
            
        #   <================ low level policy ================>
        else:
            # add noise or take random action if not subgoal testing
            if not is_subgoal_test:
                if np.random.random_sample() > 0.2:
                  action = action + np.random.normal(0, agent.exploration_action_noise)
                  action = action.clip(agent.action_clip_low, agent.action_clip_high)
                else:
                  action = np.random.uniform(agent.action_clip_low, agent.action_clip_high)
            
            # take primitive action
            next_state, rew, done, info = env.step(action)
            
            if render:
                
                env.render() ##########                    
                
            # this is for logging
            reward += rew
            total_time += 1
        
        #   <================ finish one step/transition ================>
        goal_achieved = False
        if goal_based:
            # check if goal is achieved
            goal_achieved = agent.check_goal(next_state, goal, agent.threshold)
            
            # add values
            next_target = agent.get_target(i_level, next_full_state, environment_model)
            obs = agent.get_obs(i_level, full_state, goal, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, goal, env_model)

            # hindsight action transition
            data.update(obs=state, act=action, obs_next=obs_next, param=goal, gamma=0.0, done=float(done), info=info)
            if goal_achieved:
                data.update(rew=0.0, done=True)
                agent.replay_buffer[i_level].add(data)
            else:
                data.update(rew=-1.0)
                agent.replay_buffer[i_level].add(data)
                
            # copy for goal transition
            goal_data = copy.deepcopy(data)
            goal_data.update(obs=state, act=action, rew=-1.0, obs_next=obs_next, param=goal, gamma=0.0, done=float(done), info=info)
            goal_transitions.append(goal_data)
        
        full_state = next_full_state
        data.update(full_state=full_state)
        
        if done or goal_achieved:
            break

    for transition in subgoal_test_transitions:
        agent.replay_buffer[i_level].add(transition)
    
    
    #   <================ finish H attempts ================>
    
    if goal_based:
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1].update(rew=0.0)
        goal_transitions[-1].update(done= True)
        for transition in goal_transitions:
            # last state is goal for all transitions
            param = agent.get_target(i_level, goal_transitions[-1].next_full_state, environment_model)
            obs = agent.get_obs(i_level, full_state, param, env_model)
            obs_next = agent.get_obs(i_level, next_full_state, param, env_model)
            transition.update(obs=obs, obs_next=obs_next, param=param)
            agent.replay_buffer[i_level].add(transition)
        
    return next_state, reward, done, info, total_time
