import numpy as np
import os, cv2, time
from ReinforcementLearning.rollouts import RLRollouts
from file_management import save_to_pickle, load_from_pickle
from Options.option import PrimitiveOption
from Options.action_map import PrimitiveActionMap
from EnvironmentModels.environment_model import ControllableFeature

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
    def __init__(self, nodes, edges, cfs):
        self.nodes = nodes
        self.edges = edges
        self.cfs = cfs

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_edge(self, edge):
        self.edges[(edge.tail, edge.head)] = edge

    def save_graph(self, save_dir, simplify, cuda=False):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        node_policies = dict() # don't save policies or rollouts
        node_rollouts = dict()
        es = dict()
        for name, node in self.nodes.items():
            iscuda = node.option.iscuda
            # print(node)
            node.option.cpu()
            # print(name)
            if hasattr(node.option, 'environment_model'):
                es[name] = node.option.environment_model.environment
                print(node.option.environment_model.environment, es[name])
                node.option.environment_model.environment = None
            node_policies[name] = node.option.save(save_dir, clear=True)
            # node_policies[name] = (node.option.policy, iscuda)
            # node.option.policy = None
            # if len(simplify) > 0 and node.option.dataset_model is not None:
            #     node.option.dataset_model.reduce_range(simplify + [node.option.object_name])
        save_to_pickle(os.path.join(save_dir, "graph.pkl"), self)
        for name, node in self.nodes.items():
            if hasattr(node.option, 'environment_model'):
                # print(name, es[name])
                node.option.environment_model.environment = es[name]
        for name, policy in node_policies.items():
            self.nodes[name].option.policy = policy
        # for name, rollouts in node_rollouts.items():
        #     self.nodes[name].option.rollouts = rollouts
        if cuda:
            for name, node in self.nodes.items():
                node.option.cuda()
            # if iscuda:
            #     node.option.cuda()

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
        # if node.name != 'Action':
        #     node.option.rollouts = RLRollouts(length, node.option.rollout_params[1])
        print("loaded object")
    for node in graph.nodes.values():
        print(node.name, node.option.policy)
    return graph

def graph_construct_load(args, environment, environment_model):
    '''
    tries to load the model, and in the case that it fails, tries to construct the model 
    '''
    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        args.primitive_action_map = PrimitiveActionMap(args)
        afs = environment_model.construct_action_selector()
        if environment.discrete_actions:
            controllable_feature_selectors = [ControllableFeature(afs, [0,environment.num_actions-1],1)]
        else:
            controllable_feature_selectors = list()
            for i, af in enumerate(afs):
                step = (environment.action_space.high[i] - environment.action_space.low[i]) / 3
                controllable_feature_selectors.append(ControllableFeature(af, [environment.action_space.low[i],environment.action_space.high[i]],step))
        actions = PrimitiveOption(args, None)

        if environment.discrete_actions:
            nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        else:
            nodes = {'Action': OptionNode('Action', actions, action_shape = environment.action_shape)}
        graph = OptionGraph(nodes, dict(), controllable_feature_selectors)
    return graph, controllable_feature_selectors, args
