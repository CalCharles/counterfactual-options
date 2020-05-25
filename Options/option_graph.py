import numpy as np
import os, cv2, time

class OptionNode():
# TODO: handling of multiple incoming edges to a node
	def __init__(self, name):
		self.name = name
		self.option_layer = OptionLevel()

class OptionEdge():
	def __init__(self, head, tail, option_level):
		self.head = head
		self.tail = tail
		self.options = option_level

class OptionGraph():
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges

	def add_node(self, node):
		self.nodes[node.name] = node

	def add_edge(self, edge):
		self.edges[(edge.tail, edge.head)] = edge
		self.nodes[edge.head].option_layer.options += edge.options
