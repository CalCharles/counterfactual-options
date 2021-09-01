from collections import deque
import numpy as np
from file_management import save_to_pickle
import os

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
    for j in range(trials):
        result = test_collector.collect(n_episode=1, n_term=1, n_step=args.max_steps, random=random)
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
    train_collector.collect(n_step=args.pretrain_iters, random=True) # param doesn't matter with random actions
    if args.input_norm: option.policy.compute_input_norm(train_collector.buffer)

    # collect initial test trials
    initial_perf, initial_suc, initial_hit = _collect_test_trials(args, test_collector, 0, 0, list(), list(), list(), list(), random=True, option=option)

    total_steps = 0
    hit_miss_queue_test = deque(maxlen=2000)
    hit_miss_queue_train = deque(maxlen=20)

    for i in range(args.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.num_steps) # TODO: make n-episode a usable parameter for collect
        total_steps = collect_result['n/st'] + total_steps
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        hit_miss_queue_train.append([collect_result['n/h'], collect_result['n/m']])

        if i % args.log_interval == 0:
            print("testing collection")
            _collect_test_trials(args, test_collector, i, total_steps, test_perf, suc, hit_miss_queue_test, hit_miss_queue_train, option=option)
        # train option.policy with a sampled batch data from buffer
        losses = option.policy.update(args.batch_size, train_collector.buffer)
        if args.input_norm:
            option.policy.compute_input_norm(train_collector.buffer)
        if i % args.log_interval == 0:
            print("losses", losses)
            # print("epsilons", epsilon, interaction, epsilon_close)

        if args.save_interval > 0 and (i+1) % args.save_interval == 0:
            full_save(args, option, graph)

        # Buffer debugging printouts
        if i % args.log_interval == 0:
            buf, hrb = train_collector.buffer, option.policy.learning_algorithm.replay_buffer
            print("main buffer", len(buf), train_collector.get_buffer_idx()) 
            rv_mean = [-.105,-.05,.8725, -.105,-.05,.824, -.105,-.05,.824, 0,0,0, 0,0,0.03]
            rv_variance = [.2,.26,.0425, .2,.26,.001, .2,.26,.001, .2,.26,.0425, .2,.26,.0425]

            rv = lambda x: (x * rv_variance) + rv_mean 

            for j in range(100):
                idx = (train_collector.get_buffer_idx() + (j - 100)) % args.buffer_len
                d, info,r, bi, p, itr, t, obs = buf[idx].done, buf[idx].info, buf[idx].rew, buf[idx].inter, buf[idx].param, buf[idx].inter_state, buf[idx].next_target, buf[idx].obs_next
                print(j, idx, d, info["TimeLimit.truncated"], r, bi, p, itr, t, obs, rv(obs))

            if len(hrb) > 100:
                print("hindsight buffer", len(hrb), option.policy.learning_algorithm.get_buffer_idx())
                for j in range(100):
                    idx = (option.policy.learning_algorithm.get_buffer_idx() + (j - 100)) % args.buffer_len
                    dh, infoh, rh, ih, ph, itrh, th, obsh = hrb[idx].done, hrb[idx].info, hrb[idx].rew, hrb[idx].inter, hrb[idx].param, hrb[idx].inter_state, hrb[idx].next_target, hrb[idx].obs_next
                    print(j, idx, dh, infoh["TimeLimit.truncated"], rh, ih, ph, itrh, th, obsh, rv(obsh))
        # # END PRINTOUTS

    if args.save_interval > 0:
        full_save(args, option, graph)
    final_perf, final_suc = list(), list()
    final_perf, final_suc, final_hit = _collect_test_trials(args, test_collector, 0, 0, final_perf, final_suc, list(), list())

    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
