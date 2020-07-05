import numpy as np
import os, cv2, time
import torch
from Rollouts.rollouts import Rollouts


class RLRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is 1 for discrete, needs to be specified for continuous
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ["state", "next_state", "state_diff", "action", 'probs', 'value', 'param', 'mask', 'reward', "done"]
        self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})

class LearnerRollouts(Rollouts):
    def __init__(self, length, shapes_dict):
        '''
        action shape is 1 for discrete, needs to be specified for continuous
        only stores one action, so the Model rollouts currently does NOT store the full action chain for an option
        shapes dict should have the shape information for all the types inside
        '''
        super().__init__(length, shapes_dict)
        self.names = ['probs', 'value', 'Q_value', 'returns', 'dist_entropy']
        self.values = ObjDict({n: self.init_or_none(self.shapes[n]) for n in self.names})

    def compute_return(self, gamma, start_at=0, return_max = 20):
        '''
        Computes the return for the rollout
        '''
        gamma = args.gamma
        tau = args.tau
        if start_at + rewards.size(1) > self.length:
            roll_num = max(start_at - self.length + rewards.size(1), 0)
            self.values.returns = self.returns.roll(-roll_num, 1)
            self.values.returns[-roll_num:] = 0
        # must call reset_lists afterwards
        update_last = min(start_at, return_max)
        last_values = (torch.arange(start=update_last-1, end = -1, step=-1).float() * torch.ones(update_last)).t().flatten().detach()
        if self.iscuda:
            last_values = last_values.cuda()
        rewards = rewards.clone()
        for i, rew in enumerate(rewards):
            # update the last ten returns
            i = rewards.size(1) - i
            if start_at - i > 0:
                self.values.returns[max(start_at - update_last - i, 0):start_at - i] += (torch.pow(gamma,last_values[-min(start_at - i, update_last):]) * rew).unsqueeze(1).detach()
        if self.return_form == 'value':
            self.values.returns[self.last_start_at:start_at] += (torch.pow(gamma,last_values[-(start_at - self.last_start_at):]) * next_value[idx]).unsqueeze(1).detach()
