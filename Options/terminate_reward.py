# terminate reward done

class TerminateReward():
    def __init__(self, args):
        self.reward = args.reward
        self.term = args.termination
        self.state_extractor = args.state_extractor
        self.dataset_model = args.dataset_model
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
        self.epsilon_close = self.epsilon_min + (self.epsilon_close-self.epsilon_min) * (np.exp(-1.0 * (self.total_time)/self.epsilon_close_schedule))
        self.reward.epsilon_close = self.epsilon_close
        self.term.epsilon_close = self.epsilon_close

        self.interaction_probability = self.interaction_probability + (1-self.interaction_probability) * (np.exp(-1.0 * (self.total_time)/self.interaction_prob_schedule))
        self.term.interaction_probability = self.interaction_probability

    def check_interaction(self, inter):
        return self.term.check_interaction(inter)

    def check(self, full_state, next_full_state, inter_state = None):
        '''
        gathers necessary statistics
        '''

        inter_state = self.state_extractor.get_inter(full_state) if not inter_state else inter_state
        target_state = self.state_extractor.get_target(next_full_state)
        true_done = self.state_extractor.get_true_done(full_state)

        # compute the interaction vaalue
        inter, pred, var = self.dataset_model.hypothesize(inter_state)

        # compute the termination and reward values
        term = self.term.check(inter, target_state, param, mask, true_done)
        rew = self.reward.check(inter, target_state, param, mask, true_done)

        if self.timer == self.time_cutoff:
            term = True
            time_cutoff = True
        return term, rew
