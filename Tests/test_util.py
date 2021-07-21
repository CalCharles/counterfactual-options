import numpy as np
from file_management import read_obj_dumps, load_from_pickle, read_action_dumps
import cv2


EPSILON = .000001

def compare_factored(state1, state2):
	for n in state1.keys():
		s1 = np.array(state1[n]) if type(state1[n]) == list else state1[n]
		s2 = np.array(state2[n]) if type(state2[n]) == list else state2[n]
		if np.linalg.norm(s1 - s2) > EPSILON:
			return False
	return True

def compare_final(test_paths, filenames, action_sequences)
    for pth, filename, acts in zip(test_paths, filenames, action_sequences):
        factored_states = read_obj_dumps(pth=pth, filename=filename+"_start.txt")
        factored_state_targets = read_obj_dumps(pth=pth, filename=filename+"_target.txt")
        actions = read_action_dumps(pth=pth, filename=acts)
        print("running: ", filename)
        for factored_state, factored_state_target, al in zip(factored_states, factored_state_targets, actions):
            environment_model.set_from_factored_state(factored_state)
            for i in al:
                state, reward, done, info = environment_model.step(i)
                cv2.imshow("state", state['raw_state'])
                cv2.waitKey(1000)
            success = compare_factored(state['factored_state'], factored_state_target)
            print("compared final: ", success)