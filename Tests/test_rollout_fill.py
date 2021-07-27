from Environments.environment_initializer import initialize_environment
from Rollouts.rollouts import ObjDict
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
import cv2
import numpy as np


def test_rollout_fill():
    args = ObjDict()
    args.seed = 0
    args.env = "SelfBreakout"
    args.record_rollouts = ""
    args.save_recycle = -1
    args.save_raw = False
    args.drop_stopping = False
    args.true_environment = False
    environment, environment_model, args = initialize_environment(args)

    data = read_obj_dumps(pth="data/unit_test/rollout_fill/", i=-1, rng = 100, filename='object_dumps.txt')
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)
    print("difference between insert and data, should be zero: ", np.sum(rollouts.get_values("state") - environment_model.flatten_factored_state(data)))