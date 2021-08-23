from Environments.environment_initializer import initialize_environment
from Rollouts.rollouts import ObjDict
from Rollouts.param_buffer import ParamReplayBuffer
from Rollouts.collector import OptionCollector
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Tests.test_util import compare_factored
from Tests.DummyClasses.collector_dummy_classes_robopush import *
from Options.Termination.termination import terminal_forms
from Options.Reward.reward import reward_forms
from Options.done_model import DoneModel
from Options.terminate_reward import TerminateReward
import cv2
import numpy as np

def test_robo_terminate_reward():
    args = ObjDict()
    args.seed = 0
    args.env = "RoboPushing"
    args.record_rollouts = ""
    args.save_recycle = -1
    args.max_steps = 30
    args.time_cutoff = 30
    args.save_raw = False
    args.drop_stopping = False
    args.true_environment = False
    environment, environment_model, args = initialize_environment(args)
    environment.reset()

    dataset_model = TestDatasetModel()

    # initialize parameters for termination and reward
    args.epsilon_close = .01
    args.dataset_model = dataset_model
    args.name = "Gripper"
    args.object = "Gripper"
    args.interaction_probability = 0
    args.param_interaction = True
    args.true_reward_lambda = 0
    args.parameterized_lambda = 10
    args.interaction_lambda = 0
    args.reward_constant = -1
    args.use_termination = True
    args.true_interaction = False
    args.terminal_type = "tcomb"
    args.reward_type = "tcomb"
    args.dataset_model = dataset_model
    args.not_true_done_stopping = False

    # initialize termination function, reward function, done model
    tt = args.terminal_type[:4] if args.terminal_type.find('inst') != -1 else args.terminal_type
    rt = args.reward_type[:4] if args.reward_type.find('inst') != -1 else args.reward_type
    termination = terminal_forms[tt](**args)
    reward = reward_forms[rt](**args) # using the same epsilon for now, but that could change
    done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)
    
    # initialize terminate-reward
    args.reward = reward
    args.termination = termination
    args.epsilon_min = 1
    args.epsilon_close_schedule = 0
    args.interaction_prediction = .3
    args.state_extractor = TestExtractor()
    terminate_reward = TerminateReward(args)

    models = ObjDict()
    models.sampler = TestSampler()
    models.state_extractor = args.state_extractor
    models.terminate_reward = terminate_reward
    models.action_map = None
    models.dataset_model = args.dataset_model
    models.temporal_extension_manager = TestTemporalExtensionManager()
    models.done_model = done_model
    models.policy = TestPolicy()


    option = TestOption(models)

    # Initialize HER
    args.prioritized_replay = list()
    args.select_positive = .5
    args.resample_timer = 10
    args.use_interact = True
    args.resample_interact = False
    args.max_hindsight = 500
    args.early_stopping = False    
    args.buffer_len = 1000
    option.policy.init_HER(args, option)


    trainbuffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

    args.param_recycle = 0.1
    train_collector = OptionCollector(option.policy, environment, trainbuffer, exploration_noise=True, test=False,
                    option=option, args = args) # for now, no preprocess function
    
    train_collector.reset()
    batches = list()
    for i in range(3):
        collect_result = train_collector.collect(n_step=25)
        batches += collect_result['saved_fulls']
    for i, b in enumerate(batches):
        print(i, b.rew, b.inter, b.terminate, b.obs, b.obs_next)
    print("buffer values")
    for i in range(len(trainbuffer)):
        print(i, trainbuffer[i].time, trainbuffer[i].rew, trainbuffer[i].terminate, trainbuffer[i].obs, trainbuffer[i].obs_next)
        print(trainbuffer[i].info)
    replay_buffer = option.policy.her.replay_buffer
    for i in range(len(replay_buffer)):
        print(i, replay_buffer[i].time, replay_buffer[i].rew, replay_buffer[i].terminate, replay_buffer[i].obs, replay_buffer[i].obs_next)
        print(replay_buffer[i].info)