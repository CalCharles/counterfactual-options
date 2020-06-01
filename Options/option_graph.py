import numpy as np
import os, cv2, time
from Options.option import OptionLayer

class OptionNode():
# TODO: handling of multiple incoming edges to a node
	def __init__(self, name, option_layer):
		self.name = name
		self.option_layer = option_layer

class OptionEdge():
	def __init__(self, head, tail, option_layer):
		self.head = head
		self.tail = tail
		self.options = option_layer

class OptionGraph():
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges

	def add_node(self, node):
		self.nodes[node.name] = node

	def add_edge(self, edge):
		self.edges[(edge.tail, edge.head)] = edge
		self.nodes[edge.head].option_layer.options += edge.options
