from collections import deque
import numpy as np
from file_management import save_to_pickle
import os
import copy

def _collect_test_trials(args, test_collector, i, total_steps, test_perf, suc, hit_miss, hit_miss_train, random=False, option=None):
    '''
    collect trials with the test collector
    the environment is reset before starting these trials
    most of the inputs are used for printing out the results of these trials 
    '''
    test_collector.reset()
    trials = args.test_trials
    # print(test_collector.data)
    if random:
        trials = args.pretest_trials
    #     trials = args.test_trials * 10
    if args.object == "Block" and args.env == "SelfBreakout":
        orig_env_model = test_collector.option.sampler.current_environment_model
        test_collector.option.sampler.current_environment_model = test_collector.environment_model
    for j in range(trials):
        result = test_collector.collect(n_episode=1, n_term=1, n_step=args.max_steps, random=random, visualize_param=args.visualize_param)
        test_perf.append(result["rews"].mean())
        suc.append(float(result["terminate"]))
        hit_miss.append(result['n/h'])
    if random:
        print("Initial trials: ", trials)
    else:
        print("Iters: ", i, "Steps: ", total_steps)
    mean_perf, mean_suc, mean_hit = np.array(test_perf).mean(), np.array(suc).mean(), sum(hit_miss)/ max(1, len(hit_miss))
    hmt = 0.0
    if len(list(hit_miss_train)) > 0:
        hmt = np.sum(np.array(list(hit_miss_train)), axis=0)
        hmt = hmt[0] / (hmt[0] + hmt[1])
    print(f'Test mean returns: {mean_perf}', f"Success: {mean_suc}", f"Hit Miss: {mean_hit}", f"Hit Miss train: {hmt}")
    if args.object == "Block" and args.env == "SelfBreakout":
        test_collector.option.sampler.current_environment_model = orig_env_model
    return mean_perf, mean_suc, mean_hit 

def full_save(args, option, graph):
    option.save(args.save_dir)
    graph.save_graph(args.save_graph, [args.object], args.environment_model, cuda=args.cuda)


def trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph):
    '''
    Run the RL train loop
    '''
    test_perf, suc = deque(maxlen=2000), deque(maxlen=2000)

    
    # collect initial random actions 
    train_collector.collect(n_step=args.pretrain_iters, random=True, visualize_param=args.visualize_param) # param doesn't matter with random actions
    if args.input_norm: option.policy.compute_input_norm(train_collector.buffer)

    # collect initial test trials
    initial_perf, initial_suc, initial_hit = _collect_test_trials(args, test_collector, 0, 0, list(), list(), list(), list(), random=True, option=option)

    total_steps = 0
    hit_miss_queue_test = deque(maxlen=2000)
    hit_miss_queue_train = deque(maxlen=args.log_interval)
    cumul_losses = deque(maxlen=args.log_interval)


    for i in range(args.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.num_steps, visualize_param=args.visualize_param) # TODO: make n-episode a usable parameter for collect
        total_steps = collect_result['n/st'] + total_steps
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        hit_miss_queue_train.append([collect_result['n/h'], collect_result['n/m']])

        if i % args.log_interval == 0:
            print("testing collection")
            _collect_test_trials(args, test_collector, i, total_steps, test_perf, suc, hit_miss_queue_test, hit_miss_queue_train, option=option)
        # train option.policy with a sampled batch data from buffer
        losses = option.policy.update(args.batch_size, train_collector.buffer)
        cumul_losses.append(losses)

        if args.input_norm:
            option.policy.compute_input_norm(train_collector.buffer)
        if i % args.log_interval == 0:
            # compute the average loss
            total_losses = copy.deepcopy(cumul_losses[0])
            for j in range(len(cumul_losses) - 1):
                l = cumul_losses[j]
                for k in l.keys():
                    total_losses[k] += l[k]
            for k in total_losses.keys():
                total_losses[k] = total_losses[k] / len(cumul_losses)
            print("losses", total_losses)
            option.print_epsilons()
            # print("epsilons", epsilon, interaction, epsilon_close)

        if args.save_interval > 0 and (i+1) % args.save_interval == 0:
            full_save(args, option, graph)

        # Buffer debugging printouts
        if i % args.log_interval == 0:
            buf = train_collector.buffer
            print("main buffer", len(buf), train_collector.get_buffer_idx())
            rv = lambda x: ""
            if args.env == "RoboPushing" and args.object == 'Block':
                rv_mean = [-.105,-.05,.8725, -.105,-.05,.824, -.105,-.05,.824, 0,0,0, 0,0,0.03]
                rv_variance = [.2,.26,.0425, .2,.26,.001, .2,.26,.001, .2,.26,.0425, .2,.26,.0425]

                rv = lambda x: (x * rv_variance) + rv_mean

            for j in range(50):
                idx = (train_collector.get_buffer_idx() + (j - 100)) % args.buffer_len
                d, info,r, bi, a, ma, ti, p, t, nt, itr, obs = buf[idx].done, buf[idx].info, buf[idx].rew, buf[idx].inter, buf[idx].act, buf[idx].mapped_act, buf[idx].time, buf[idx].param, buf[idx].target, buf[idx].next_target, buf[idx].inter_state, buf[idx].obs_next
                print(j, idx, d, info["TimeLimit.truncated"], r, bi, a, ma, ti, p, t, nt, itr, obs, rv(obs))

            if option.policy.is_her:
                hrb = option.policy.learning_algorithm.replay_buffer
                if len(hrb) > 100:
                    print("hindsight buffer", len(hrb), option.policy.learning_algorithm.get_buffer_idx())
                    for j in range(50):
                        idx = (option.policy.learning_algorithm.get_buffer_idx() + (j - 100)) % args.buffer_len
                        dh, infoh, rh, ih, ah, mah, tih, ph, th, nth, itrh, obsh = hrb[idx].done, hrb[idx].info, hrb[idx].rew, hrb[idx].inter, hrb[idx].act, hrb[idx].mapped_act, hrb[idx].time, hrb[idx].param, hrb[idx].target, hrb[idx].next_target, hrb[idx].inter_state, hrb[idx].obs_next
                        print(j, idx, dh, infoh["TimeLimit.truncated"], rh, ih, ah, mah, tih, ph, th, nth, itrh, obsh, rv(obsh))
        # # END PRINTOUTS

    if args.save_interval > 0:
        full_save(args, option, graph)
    final_perf, final_suc = list(), list()
    final_perf, final_suc, final_hit = _collect_test_trials(args, test_collector, 0, 0, final_perf, final_suc, list(), list())

    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
