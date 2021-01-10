import numpy as np
import os, cv2, time
from file_management import save_to_pickle, load_from_pickle

class OptionNode():
# TODO: handling of multiple incoming edges to a node
    def __init__(self, name, option, action_shape):
        self.name = name # the name of the object being controlled
        self.option = option # stores most of the necessary information
        self.action_shape = action_shape

class OptionEdge():
    def __init__(self, head, tail):
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

    def save_graph(self, save_dir, simplify):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        node_policies = dict()
        for name, node in self.nodes.items():
            iscuda = node.option.iscuda
            print(node)
            node.option.cpu()
            node.option.save(save_dir)
            node_policies[name] = (node.option.policy, iscuda)
            node.option.policy = None
            if len(simplify) > 0 and node.option.dataset_model is not None:
                node.option.dataset_model.reduce_range(simplify + [node.option.object_name])
        save_to_pickle(os.path.join(save_dir, "graph.pkl"), self)
        for name, (policy, iscuda) in node_policies.items():
            self.nodes[name].option.policy = policy 
            if iscuda:
                node.option.cuda()

    def load_environment_model(self, environment_model):
        for node in self.nodes.values():
            node.option.environment_model = environment_model

def load_graph(load_dir):
    print(os.path.join(load_dir, "graph.pkl"))
    graph = load_from_pickle(os.path.join(load_dir, "graph.pkl"))
    print("loaded graph")
    for node in graph.nodes.values():
        print(node.name, load_dir, node.option.object_name +"_policy")
        node.option.load_policy(load_dir)
        print("loaded object")
    for node in graph.nodes.values():
        print(node.name, node.option.policy)
    return graph
