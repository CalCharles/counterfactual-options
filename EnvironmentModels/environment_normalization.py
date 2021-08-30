import numpy as np
from Networks.network import pytorch_model

breakout_action_norm = (np.array([0,0,0,0,1.5]), np.array([1,1,1,1,1.5]))
breakout_paddle_norm = (np.array([72, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_state_norm = (np.array([84 // 2, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_block_norm = (np.array([84 // 2, 84 // 2, 0,0,.5]), np.array([.1, .1, .1,.1,.5]))
breakout_relative_norm = (np.array([0,0,0,0,.5]), np.array([84 // 2, 84 // 2,2,1,.5]))

# .10, -.31
# .21, -.31
# .915, .83

# -.105, .2
# -.05, .26
# .8725, .0425
robopush_action_norm = (np.array([0,0,0]), np.array([1,1,1]))
robopush_gripper_norm = (np.array([-.105,-.05,.8725]), np.array([.2,.26,.0425]))
robopush_state_norm = (np.array([-.105,-.05,.802]), np.array([.2,.26,.001]))
robopush_relative_norm = (np.array([0,0,0]), np.array([.2,.26,.0425]))
robopush_relative_surface_norm = (np.array([0,0,0]), np.array([.2,.2,.002]))


robo_norms = {"Action": robopush_action_norm, "Gripper": robopush_gripper_norm, "Block": robopush_state_norm, "Target": robopush_state_norm, "RelativeGripper": robopush_relative_norm, "RelativeBlock": robopush_relative_norm, "RelativeTarget": robopush_relative_norm}
breakout_norms = {"Action": breakout_action_norm, "Paddle": breakout_paddle_norm, "Ball": breakout_state_norm, "Block": breakout_block_norm, "RelativeBall": breakout_relative_norm, "RelativePaddle": breakout_relative_norm, "RelativeBlock": breakout_relative_norm}

def hardcode_norm(env_name, obj_names):
	if env_name == "Breakout":
		norm_dict = breakout_norms
	elif env_name == "RoboPushing":
		norm_dict = robo_norms
	# ONLY IMPLEMENTED FOR Breakout, Robosuite Pushing
	norm_mean = list()
	norm_var = list()
	for n in obj_names:
		norm_mean.append(norm_dict[n][0])
		norm_var.append(norm_dict[n][1])
	return np.concatenate(norm_mean, axis=0), np.concatenate(norm_var, axis=0)