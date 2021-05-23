class DoneModel():
	def __init__(self, **kwargs):
		self.use_termination = kwargs["use_termination"]
		self.use_timer = kwargs["time_cutoff"]
		self.use_true_done = kwargs["true_done_stopping"]


	def check(self, termination, timer, true_done):
		return (termination * self.use_termination) or (timer == self.use_timer - 1) or (self.use_true_done * true_done)