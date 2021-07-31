class TemporalExtensionManager():
    def __init__(self, args):
        self.needs_sample = True
        self.act = None
        self.chain = None
        self.policy_batch = None
        self.state = None
        self.masks = None

        self.ext_cutoff = args.temporal_extend
        self.timer = 0
        self.reset()

    def reset(self):
        self.needs_sample = True

    def update(self, act, chain, TEterm, masks):
        # update the policy action and mapped action chain and the timer
        self.timer += 1
        if TEterm:
            self.timer = 0
        self.act = act # this is the policy action ONLY for the highest option
        self.chain = chain
        self.masks = masks

    def update_policy(self, policy_batch, state):
        '''
        updates special policy outputs, (TODO: which I rarely use), called in
        '''
        self.policy_batch = policy_batch
        self.state = state

    def check(self, terminate, ext_term):
        '''
        returns the last action, chain, policy_batch and state if still temporally extending
        otherwise returns 
        '''
        if self.needs_sample:
            # the case where we are in the first state
            needs_sample = True
            self.needs_sample = False
        else:
            # a temporally extended action just finished
            needs_sample = self.get_extension(terminate, ext_term)
        return needs_sample, self.act, self.chain, self.policy_batch, self.state, self.masks

    def get_extension(self, terminate, ext_term):
        # Contains any logic for deciding whether to end temporally extension (either timer, or terminate, or action terminate)
        return terminate or ext_term or self.timer == self.ext_cutoff
