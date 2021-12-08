import numpy as np
from Networks.network import pytorch_model

breakout_action_norm = (np.array([0,0,0,0,1.5]), np.array([1,1,1,1,1.5]))
breakout_paddle_norm = (np.array([72, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_state_norm = (np.array([84 // 2, 84 // 2, 0,0,1]), np.array([84 // 2, 84 // 2, 2,1,1]))
breakout_block_norm = (np.array([27, 42, 0,0,.5]), np.array([84 // 2, 84 // 2, 84 // 2,84 // 2,1]))
breakout_relative_norm = (np.array([0,0,0,0,1]), np.array([84 // 2, 84 // 2,2,1, 1]))
breakout_relative_block_norm = (np.array([0,0,0,0,0]), np.array([84 // 2, 84 // 2,2,1,1]))
breakout_hot_norm = (np.array([0,0,0,0]), np.array([1,1,1,1]))
breakout_block_binary_norm = (np.array([0 for _ in range(100)]), np.array([1 for _ in range(100)]))

# .10, -.31
# .21, -.31
# .915, .83

# -.105, .2
# -.05, .26
# .8725, .0425
robopush_action_norm = (np.array([0,0,0]), np.array([1,1,1]))
robopush_gripper_norm = (np.array([0.0,-.05,.8725]), np.array([.2,.26,.0425]))
robopush_state_norm = (np.array([0.0,-.05,.824]), np.array([.2,.26,.1]))
robopush_relative_norm = (np.array([0,0,0]), np.array([.2,.26,.0425]))
robopush_relative_surface_norm = (np.array([0,0,0]), np.array([.2,.2,.1]))

robostick_action_norm = (np.array([0,0,0,0]), np.array([1,1,1,1]))
robostick_gripper_norm = (np.array([-0.2, 0,.8725,0]), np.array([.1,.15,.0425,1]))
robostick_stick_norm = (np.array([-0.1,0.0,.824]), np.array([.2,.15,.1]))
robostick_block_norm = (np.array([-0.1,0.0,.802]), np.array([.1,.15,.1]))
robostick_target_norm = (np.array([0.0,0.0,.802]), np.array([.07,.15,.1]))
robostick_stick_bounds = (np.array([-0.3, -.1, .824]), np.array([0.06, 0.1, .925]))
robostick_relative_norm = (np.array([0.0,0.0,0.0]), np.array([.3,.3,.2]))




robo_norms = {"Action": robopush_action_norm, "Gripper": robopush_gripper_norm, "Block": robopush_state_norm, "Target": robopush_state_norm, 
				"RelativeGripper": robopush_relative_norm, "RelativeBlock": robopush_relative_norm, "RelativeTarget": robopush_relative_norm}
stick_norms = {"Action": robostick_action_norm, "Gripper": robostick_gripper_norm, "Stick":robostick_stick_norm, "Block": robostick_block_norm, "Target": robostick_target_norm, 
				"RelativeGripper": robostick_relative_norm, "RelativeStick": robostick_relative_norm, "RelativeBlock": robostick_relative_norm, "RelativeTarget": robostick_relative_norm}

breakout_norms = {"Hot": breakout_hot_norm, "Action": breakout_action_norm, "Paddle": breakout_paddle_norm, "Ball": breakout_state_norm, "Block": breakout_block_norm,
				'block_binary': breakout_block_binary_norm,
				"RelativeBall": breakout_relative_norm, "RelativePaddle": breakout_relative_norm, "RelativeBlock": breakout_relative_block_norm}

def hardcode_norm(env_name, obj_names):
	if env_name == "SelfBreakout":
		norm_dict = breakout_norms
	elif env_name == "RoboPushing":
		norm_dict = robo_norms
	elif env_name == "RoboStick":
		norm_dict = stick_norms
	# ONLY IMPLEMENTED FOR Breakout, Robosuite Pushing
	norm_mean = list()
	norm_var = list()
	norm_inv_var = list()
	for n in obj_names:
		norm_mean.append(norm_dict[n][0])
		norm_var.append(norm_dict[n][1])
		norm_inv_var.append(1.0/norm_dict[n][1])
	return np.concatenate(norm_mean, axis=0), np.concatenate(norm_var, axis=0), np.concatenate(norm_inv_var, axis=0)