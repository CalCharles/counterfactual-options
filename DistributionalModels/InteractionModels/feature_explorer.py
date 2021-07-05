# Feature Search Function
import numpy as np
import os, cv2, time, copy
import torch
from collections import Counter
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts
from DistributionalModels.InteractionModels.interaction_model import interaction_models, default_model_args, nf5, nflen
from Networks.input_norm import InterInputNorm

class FeatureExplorer():
    def __init__(self, graph, controllable_feature_selectors, environment_model, model_args):
        self.cfs = controllable_feature_selectors
        self.em = environment_model
        self.model_args = model_args
        self.graph = graph

    def search(self, rollouts, train_args):
        # only search between entities (so that it's easier)
        gamma_size = 1
        delta_size = 1
        found = False
        gamma_tested = set()
        self.model_args["cuda"] = train_args.cuda
        print(self.graph.edges)
        # edge_set = {e for e in self.graph.edges.keys()}

        # while not found:
        cfslist = copy.copy(self.cfs)
        additional = [[]] + [[cfs.feature_selector.get_entity()[0]] for cfs in cfslist]
        cfslist.reverse()
        print("controllable objects", [c.object() for c in cfslist])
        # HACKED LINE TO SPEED UP TRAINING
        # for cfs in [cfslist[0]]:
        for cfs in cfslist:
            controllable_entity = cfs.feature_selector.get_entity()[0]
            if controllable_entity not in gamma_tested:
                for additional_feature in additional:
                # for additional_feature in [["Action"]]:
                    if len(additional_feature) > 0 and additional_feature[0] == cfs.object():
                        continue # don't add additionally the same feature being tested
                    delta_tested = set()
                    # HACKED LINE TO SPEED UP TRAINING
                    # for name in ["Ball"]:
                    # for name in ["Block"]:
                    for name in self.em.object_names:
                        if name != controllable_entity and name not in delta_tested and (controllable_entity, name) not in self.graph.edges:
                            names = [controllable_entity] + additional_feature + [name]
                            entity_selection = self.em.create_entity_selector(names)
                            model, test, gamma_new, delta_new = self.train(cfs, additional_feature, rollouts, train_args, entity_selection, name)
                            comb_passed, combined = self.pass_criteria(train_args, model, test, train_args.model_error_significance)
                            gamma_comb = gamma_new
                            delta_comb = delta_new
                            # must include the delta as an input
                            # entity_selection = self.em.create_entity_selector([controllable_entity])
                            # model, test, gamma_new, delta_new = self.train(cfs, rollouts, train_args, entity_selection, name)
                            # sep_passed, sep = self.pass_criteria(model, test, train_args.model_error_significance)
                            # if sep_passed or comb_passed:
                            if comb_passed:
                                train_args.separation_difference = 1e-1# Change this line later
                                # if sep >= combined - train_args.separation_difference:
                                print("selected combined")
                                gamma = gamma_comb
                                delta = delta_comb
                                # else:
                                #     print("selected separate")
                                #     gamma = gamma_new
                                #     delta = delta_new
                                found = True
                                break
                            delta_tested.add(name)
                    gamma_tested.add(controllable_entity)
                    if found:
                        break
                if found:
                    break
        if not found:
            return None
        return model, gamma, delta

    def pass_criteria(self, args, model, test, model_error_significance): # TODO: using difference from passive is not a great criteria since the active follows a difference loss once interaction is added in
        forward_error, passive_error = model.assess_error(test, passive_error_cutoff=args.passive_error_cutoff)
        passed = forward_error < (passive_error - model_error_significance)
        print("comparison", forward_error, passive_error, model_error_significance, passed)
        return passed, forward_error-passive_error

    def train(self, cfs, additional_object, rollouts, train_args, entity_selection, name):
        print("Edge ", cfs.object(), "-> ", name)
        aosize = 0
        model_name = cfs.object() + "->"+ name
        if len(additional_object) > 0:
            additional_object = additional_object[0]
            ao_size = self.em.object_sizes[additional_object] * self.em.object_num[additional_object]
            print("Training ", cfs.object(), " + ", additional_object, " -> ", name)
            model_name = cfs.object() + "+" + additional_object+ "->" + name
        self.model_args['name'] = model_name
        self.model_args['gamma'] = entity_selection
        self.model_args['delta'] = self.em.create_entity_selector([name])
        self.model_args['object_dim'] = self.em.object_sizes[name]
        self.model_args['output_dim'] = self.em.object_sizes[name]
        self.model_args['first_obj_dim'] = self.em.object_sizes[cfs.object()]
        nout = self.em.object_sizes[name] * self.em.object_num[name]
        nin = self.em.object_sizes[cfs.object()] * self.em.object_num[cfs.object()] + nout
        input_norm_fun = InterInputNorm()
        input_norm_fun.compute_input_norm(entity_selection(rollouts.get_values("state")))
        delta_norm_fun = InterInputNorm()
        delta_norm_fun.compute_input_norm(self.model_args['delta'](rollouts.get_values("state")))
        self.model_args['normalization_function'] = input_norm_fun#nflen(nin)
        self.model_args['delta_normalization_function'] = delta_norm_fun#nflen(nout) if not train_args.predict_dynamics else nf5

        dma = default_model_args(train_args.predict_dynamics, train_args.policy_type, input_norm_fun, delta_norm_fun)
        # self.model_args['normalization_function'] = dma['normalization_function']
        print(entity_selection.output_size())
        self.model_args['num_inputs'] = self.model_args['gamma'].output_size()
        self.model_args['num_outputs'] = self.model_args['delta'].output_size()
        self.model_args['multi_instanced'] = train_args.multi_instanced
        model = interaction_models[self.model_args['model_type']](**self.model_args)
        print(model)
        train, test = rollouts.split_train_test(train_args.ratio)
        train.cpu(), test.cpu()
        save_to_pickle("data/train.pkl", train)
        save_to_pickle("data/test.pkl", test)
        # train = load_from_pickle("data/train.pkl")
        # test = load_from_pickle("data/test.pkl")
        train.cuda(), test.cuda()
        model.train(train, train_args, control=cfs, target_name=name)
        return model, test, self.model_args['gamma'], self.model_args['delta']

