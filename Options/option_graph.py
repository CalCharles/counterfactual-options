import numpy as np
import os, cv2, time

class OptionNode():
# TODO: handling of multiple incoming edges to a node
	def __init__(self, name, option, action_shape, num_params):
		self.name = name
		self.option = option
		self.action_shape = action_shape
		self.num_params = num_params

class OptionEdge():
	def __init__(self, head, tail, option):
		self.head = head
		self.tail = tail

class OptionGraph():
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges

	def add_node(self, node):
		self.nodes[node.name] = node

	def add_edge(self, edge):
		self.edges[(edge.tail, edge.head)] = edge
