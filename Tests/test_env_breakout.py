from Environments.environment_initializer import initialize_environment
from Rollouts.rollouts import ObjDict
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Tests.test_util import compare_factored
import cv2
import numpy as np


def test_block():
    args = ObjDict()
    args.seed = 0
    args.env = "SelfBreakout"
    args.record_rollouts = ""
    args.save_recycle = -1
    args.save_raw = False
    args.drop_stopping = False
    args.true_environment = False
    environment, environment_model, args = initialize_environment(args)
    environment.reset()
    factored_states = read_obj_dumps(pth="data/unit_test/bounce_test/", filename="bounce_test_start.txt")
    factored_state_targets = read_obj_dumps(pth="data/unit_test/bounce_test/", filename="bounce_test_target.txt")
    for factored_state, factored_state_target in zip(factored_states, factored_state_targets):
        environment_model.set_from_factored_state(factored_state)
        for i in range(10):
            state, reward, done, info = environment_model.step(0)
            cv2.imshow("state", state['raw_state'])
            cv2.waitKey(1000)
        success = compare_factored(state['factored_state'], factored_state_target)
        print("compared block final: ", success)

