from collections import deque
import numpy as np

def _collect_test_trials(args, test_collector, i, total_steps, test_perf, suc, trials, random=False, option=None):
    '''
    '''
    test_collector.reset()
    trials = args.test_trials
    # print(test_collector.data)
    if random:
        trials = args.pretest_trials
    #     trials = args.test_trials * 10
    for j in range(trials):
        result = test_collector.collect(n_episode=1, n_step=args.max_steps, random=random)
        test_perf.append(result["rews"].mean())
        suc.append(float(result["terminate"]))
    if random:
        print("Initial trials: ", trials)
    else:
        print("Iters: ", i, "Steps: ", total_steps)
    mean_perf, mean_suc, mean_hit = np.array(test_perf).mean(), np.array(suc).mean(), sum(test_collector.hit_miss_queue)/ max(1, len(test_collector.hit_miss_queue))
    print(f'Test mean returns: {mean_perf}', f"Success: {mean_suc}", f"Hit Miss: {mean_hit}")
    return mean_perf, mean_suc, mean_hit 


def trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph):
    '''
    Run the RL train loop
    '''
    test_perf, suc = deque(maxlen=2000), deque(maxlen=2000)

    
    # collect initial random actions 
    train_collector.collect(n_step=args.pretrain_iters, random=True) # param doesn't matter with random actions
    if args.input_norm: option.policy.compute_input_norm(train_collector.buffer)

    # collect initial test trials
    initial_perf, initial_suc, initial_hit = _collect_test_trials(args, test_collector, 0, 0, list(), list(), args.pretest_trials,  random=True, option=option)

    total_steps = 0

    for i in range(args.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.num_steps) # TODO: make n-episode a usable parameter for collect
        total_steps = collect_result['n/st'] + total_steps
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        if i % args.log_interval == 0:
            print("testing collection")
            # option.policy.set_eps(0.05)
            test_collector.reset()
            needs_new_param = False
            for j in range(args.test_trials):
                if args.max_steps > 0: result = test_collector.collect(n_term=1, n_step=args.max_steps, visualize_param=args.visualize_param)
                else: result = test_collector.collect(n_term=1, visualize_param=args.visualize_param)
                test_perf.append(result["rews"].mean())
                # print("num steps, sucess, rew",result["n/st"], float(result["n/st"] != args.max_steps and (result["true_done"] < 1 or args.true_environment)), result["rews"].mean())
                suc.append(result['terminate'])
                needs_new_param = True
            print("Iters: ", i, "Steps: ", total_steps)
            print(f'Test mean returns: {np.array(test_perf).mean()}', f"Success: {np.array(suc).mean()}", f"Hit Miss: {sum(test_collector.hit_miss_queue)/ max(1, len(test_collector.hit_miss_queue))}", f"Hit Miss train: {sum(train_collector.hit_miss_queue)/ max(1, len(train_collector.hit_miss_queue))}")
            train_collector.reset_env() # because test collector and train collector share the same environment
            
            # intialize temporal extension
            # train_collector.reset_temporal_extension()

        # train option.policy with a sampled batch data from buffer
        losses = option.policy.update(args.batch_size, train_collector.buffer)
        if args.input_norm:
            option.policy.compute_input_norm(train_collector.buffer)
        if i % args.log_interval == 0:
            print(losses)
            # print("epsilons", epsilon, interaction, epsilon_close)
        if args.save_interval > 0 and (i+1) % args.save_interval == 0:
            option.save(args.save_dir)
            graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
    if args.save_interval > 0:
        option.save(args.save_dir)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
        save_to_pickle(os.path.join(args.record_rollouts, "action_record.pkl"), action_record)
    final_perf, final_suc = list(), list()
    final_perf, final_suc, final_hit = collect_test_trials(args, test_collector, 0, 0, final_perf, final_suc)

    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
