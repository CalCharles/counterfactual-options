import numpy as np

EPSILON = .01

def visualize(frame, object_state, param, mask):
	new_frame = frame.copy()
	if sum(mask[:2]) > EPSILON: # TODO: don't hardcode values [:2]
		loc = param.copy()
		if mask[0] == 0:
			loc[0] = object_state[0]
		if mask[1] == 0:
			loc[1] = object_state[1]
		draw_target(new_frame, loc[:2])
	if sum(mask[2:4]) > EPSILON:
		draw_line(new_frame, object_state[:2], param[2:4])
	if mask[4] > EPSILON:
		draw_target(new_frame, object_state[:2])
	return new_frame

RADIUS = 3
COLOR = 255

def draw_target(frame, loc):
	for i in range(int(np.floor(loc[0] - RADIUS)), int(np.ceil(loc[0] + RADIUS))):
		for j in range(int(np.floor(loc[1] - RADIUS)), int(np.ceil(loc[1] + RADIUS))):
			if np.sqrt((i-loc[0]) ** 2 + (j-loc[1]) ** 2) < RADIUS:
				frame[i,j] = COLOR

LINELEN = 8

def draw_line(frame, loc, direc):
	direction = direc / np.linalg.norm(direc)
	for i in range(LINELEN):
		pt = np.round(loc + direction * i)
		frame[int(pt[0]), int(pt[1])] = COLOR
