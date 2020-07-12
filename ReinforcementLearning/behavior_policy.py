def sample_actions( probs, deterministic): # TODO: why is this here?
    if deterministic is False:
        cat = torch.distributions.categorical.Categorical(probs.squeeze())
        action = cat.sample()
        action = action.unsqueeze(-1).unsqueeze(-1)
    else:
        action = probs.max(1)[1]
    return action

# TODO: doesn't handle a combination of continuous and discrete action spaces (i.e. actions and paddle simultaniously)
class BehaviorPolicy():
	def __init__(self, args, num_tail, continuous):
		self.continuous = continuous
		self.num_tail = num_tail
		self.num_outputs = args.num_outputs
		self.epsilon = args.epsilon

	def get_action(self, rl_output):
		return 0

class Probs():
	def get_action(self, rl_output):
		action_idx = 0 
		if self.num_tail > 0:
			action_idx = sample_actions(rl_output.action_idx, deterministic =True)
		if self.continuous:
	        action = rl_output.probs.dist.sample()
		else:
	        action = sample_actions(rl_output.probs, deterministic =True)
	        if np.random.rand() < self.epsilon:
	            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = rl_output.probs.shape[0]), cuda = True)
        return action


class GreedyQ():
	def get_action(self, rl_output):
		action_idx = 0 
		if self.num_tail > 0:
			action_idx = sample_actions(rl_output.action_idx, deterministic =True)
		if self.continuous:
	        action = rl_output.probs.dist.sample()
		else:
			action = sample_actions(F.softmax(rl_output.Q_vals, dim=1), deterministic =True)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = rl_output.Q_vals.shape[0]), cuda = True)
        return action_idx, action


