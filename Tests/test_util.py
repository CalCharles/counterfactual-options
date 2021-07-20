import numpy as np

EPSILON = .000001

def compare_factored(state1, state2):
	for n in state1.keys():
		s1 = np.array(state1[n]) if type(state1[n]) == list else state1[n]
		s2 = np.array(state2[n]) if type(state2[n]) == list else state2[n]
		if np.linalg.norm(s1 - s2) > EPSILON:
			return False
	return True