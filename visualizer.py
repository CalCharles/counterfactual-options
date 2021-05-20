import numpy as np

EPSILON = .01

def visualize(frame, object_state, param, mask):
	new_frame = frame.copy()
	if sum(mask[:2]) > EPSILON: # TODO: don't hardcode values [:2]
		draw_target(new_frame, param[:2])
	if sum(mask[2:4]) > EPSILON:
		draw_velocity(new_frame, object_state[:2], param[2:4])
	if mask[4] > EPSILON:
		draw_target(new_frame, object_state[:2])
	return new_frame

RADIUS = 3
COLOR = 1

def draw_target(frame, loc):
	for i in range(loc[0] - RADIUS, loc[0] + RADIUS):
		for j in range(loc[1] - RADIUS, loc[1] + RADIUS):
			if np.sqrt(i ** 2 + j ** 2) < RADIUS:
				frame[i,j] = 1

LINELEN = 5

def draw_line(frame, loc, direc):
	direction = direc / np.linalg.norm(direc)
	for i in range(LINELEN):
		pt = np.round(loc + direction * i)
		frame[int(pt[0]), int(pt[1])] = COLOR
