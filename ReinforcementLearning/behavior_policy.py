from ReinforcementLearning.Policy.policy import pytorch_model
import torch.nn.functional as F
import torch
import numpy as np

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
    def __init__(self, args):
        self.continuous = args.continuous
        self.num_outputs = args.num_outputs
        self.epsilon = args.epsilon
        self.iscuda = args.cuda

    def get_action(self, rl_output):
        return 0

class Probs(BehaviorPolicy):
    def get_action(self, rl_output):
        if self.continuous:
            action = rl_output.probs.dist.sample()
        else:
            action = sample_actions(rl_output.probs, deterministic =False)
            if np.random.rand() < self.epsilon:
                action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = rl_output.probs.shape[0]), cuda = self.iscuda)
        return action


class GreedyQ(BehaviorPolicy):
    def get_action(self, rl_output):
        if self.continuous:
            action = rl_output.probs.dist.sample()
        else:
            action = sample_actions(F.softmax(rl_output.Q_vals, dim=1), deterministic =True)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = rl_output.Q_vals.shape[0]), cuda = self.iscuda)
        return action

class SoftQ(BehaviorPolicy):
    def get_action(self, rl_output):
        if self.continuous:
            action = rl_output.probs.dist.sample()
        else:
            action = sample_actions(F.softmax(rl_output.Q_vals, dim=1), deterministic =False)
        if np.random.rand() < self.epsilon:
            action = pytorch_model.wrap(np.random.randint(self.num_outputs, size = rl_output.Q_vals.shape[0]), cuda = self.iscuda)
        return action

behavior_forms = {"prob": Probs, "greedyQ": GreedyQ, "softQ": SoftQ}