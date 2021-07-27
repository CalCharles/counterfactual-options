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
    test_paths  = ["data/unit_test/bounce_test/", "data/unit_test/paddle_test/"]
    filenames = ["bounce_test", "paddle_test"]
    action_sequences = ["zeros", "paddle_bounce"]
    run_files(test_paths, filenames, action_sequences)