# terminate reward done
import numpy as np
from Networks.network import pytorch_model

class TerminateReward():
    def __init__(self, args):
        self.reward = args.reward
        self.term = args.termination
        self.state_extractor = args.state_extractor
        self.dataset_model = args.dataset_model
        self.true_interaction = args.true_interaction
        # self.init_term = args.init_term # whether to be terminated on the first state after a reset, false except for primitive terminate reward

        self.time_cutoff = args.time_cutoff
        self.timer = 0 # used for timed terminations

        self.epsilon_min = args.epsilon_min
        self.epsilon_close = args.epsilon_close
        self.epsilon_close_schedule = args.epsilon_close_schedule
        self.total_time = 0 # used for epsilon schedules

        self.interaction_prob_schedule = args.interaction_prediction
        self.interaction_probability = args.interaction_probability

    def reset(self):
        self.timer = 0
        return False

    def update(self, term):
        '''
        step necessary timers
        '''
        self.timer += 1
        self.total_time += 1
        if term:
            self.timer = 0

        # update ranges
        if self.epsilon_close_schedule > 0: self.epsilon_close = self.epsilon_min + (self.epsilon_close-self.epsilon_min) * (np.exp(-1.0 * (self.total_time)/self.epsilon_close_schedule))
        self.reward.epsilon_close = self.epsilon_close
        self.term.epsilon_close = self.epsilon_close

        if self.interaction_prob_schedule > 0: self.interaction_probability = self.interaction_probability + (1-self.interaction_probability) * (np.exp(-1.0 * (self.total_time)/self.interaction_prob_schedule))
        self.term.interaction_probability = self.interaction_probability

    def check_interaction(self, inter):
        return self.term.check_interaction(inter)

    def check(self, full_state, next_full_state, param, mask, inter_state = None, use_timer=True, true_inter=0, ignore_true=False):
        '''
        gathers necessary statistics
        inter state is the interaction stat, which replaces getting it from the full state
        use_timer determines if the timer should influence computation
        '''

        inter_state = self.state_extractor.get_inter(full_state) if inter_state is None else inter_state
        target_state = self.state_extractor.get_target(next_full_state)
        true_done = self.state_extractor.get_true_done(next_full_state) if not ignore_true else False
        true_reward = self.state_extractor.get_true_reward(next_full_state) if not ignore_true else 0
        # compute the interaction value
        if not self.true_interaction:
            inter, pred, var = self.dataset_model.hypothesize(inter_state)
            inter = pytorch_model.unwrap(inter)
        else:
            inter = true_inter

        # compute the termination and reward values
        term = self.term.check(inter, target_state, param, mask, true_done)
        rew = self.reward.get_reward(inter, target_state, param, mask, true_reward)
        # time cutoff indicates whether the termination was due to the timer cutoff
        time_cutoff = False
        if self.timer == self.time_cutoff and use_timer:
            time_cutoff = bool(True and not term)
            term = True
        return term, rew, inter, time_cutoff
