class DoneModel():
	def __init__(self, **kwargs):
		self.use_termination = kwargs["use_termination"]
		self.use_timer = kwargs["time_cutoff"]
		self.use_true_done = kwargs["true_done_stopping"]


	def check(self, termination, timer, true_done):
		term = (termination * self.use_termination)
		tim = (timer == self.use_timer - 1)
		tru = (self.use_true_done * true_done)
		# print(term, tim, tru)
		if term:
			# print("term done")
			return term
		elif tim:
			# print("timer done")
			return tim
		elif tru:
			# print("true done")
			return tru
		return term or tim or tru