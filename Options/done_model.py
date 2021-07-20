import numpy as np

class DoneModel():
	def __init__(self, **kwargs):
		self.use_termination = kwargs["use_termination"]
		self.use_timer = kwargs["time_cutoff"]
		self.use_true_done = kwargs["true_done_stopping"]
		self.timer= 0

	def update(self, done):
		self.timer += 1
		if done:
			self.timer = 0


	def check(self, termination, true_done):
		term, tim, tru = self.done_check(termination, true_done)
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

	def done_check(self, termination, true_done):
		if type(termination) == np.ndarray: termination = termination.squeeze() # troublesome line
		term = (termination * self.use_termination)
		tim = (self.timer == self.use_timer)
		if type(true_done) == np.ndarray: true_done = true_done.squeeze() # troublesome line
		tru = (self.use_true_done * true_done)
		return term, tim, tru