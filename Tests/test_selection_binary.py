from Environments.environment_initializer import initialize_environment
from Rollouts.rollouts import ObjDict
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Network.networm import pytorch_model
from Test.test_util import compare_full_prediction
import cv2
import numpy as np
from DistributionalModels.InteractionModels.interaction_model import default_model_args, load_hypothesis_model


def test_selection_binary():
    args = ObjDict()
    args.active_epsilon = 0.5

    data = read_obj_dumps(pth="data/unit_test/active_set/", i=-1, rng = 3000, filename='object_dumps.txt')
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)

    target_selection_binary = np.array([0,0,1,1,0])
    samples = np.array([[0,0,-1,-1,0], [0,0,-2,-1,0], [0,0,-1,1,0], [0,0,-2,1,0]])
    hypothesis_model = load_hypothesis_model("data/unit_test/active_set/")
    hypothesis_model.active_epsilon = args.active_epsilon
    hypothesis_model.environment_model = environment_model
    hypothesis_model.cpu()
    hypothesis_model.cuda()
    hypothesis_model.determine_active_set(rollouts)
    hypothesis_model.collect_samples(rollouts)
    print("binaries should match: ", np.linalg.norm(hypothesis_model.selection_binary - target_selection_binary))
    print("samples should match: ", np.linalg.norm(np.array(hypothesis_model.sample_able.vals) - samples))
