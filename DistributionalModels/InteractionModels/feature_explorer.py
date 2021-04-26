# Feature Search Function
import numpy as np
import os, cv2, time, copy
import torch
from collections import Counter
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts
from DistributionalModels.InteractionModels.interaction_model import interaction_models, default_model_args

class FeatureExplorer():
    def __init__(self, graph, controllable_feature_selectors, environment_model, model_args):
        self.cfs = controllable_feature_selectors
        self.em = environment_model
        self.model_args = model_args

    def search(self, rollouts, train_args):
        # only search between entities (so that it's easier)
        gamma_size = 1
        delta_size = 1
        found = False
        gamma_tested = set()
        self.model_args["cuda"] = train_args.cuda

        # while not found:
        cfslist = copy.copy(self.cfs)
        cfslist.reverse()
        print([c.object() for c in cfslist])
        # HACKED LINE TO SPEED UP TRAINING
        for cfs in [cfslist[0 ]]:
        # for cfs in cfslist:
            controllable_entity = cfs.feature_selector.get_entity()[0]
            if controllable_entity not in gamma_tested:
                delta_tested = set()
                # HACKED LINE TO SPEED UP TRAINING
                for name in ["Ball"]:
                # for name in self.em.object_names:
                    if name != controllable_entity and name not in delta_tested:
                        entity_selection = self.em.create_entity_selector([controllable_entity, name])
                        model, test, gamma_new, delta_new = self.train(cfs, rollouts, train_args, entity_selection, name)
                        comb_passed, combined = self.pass_criteria(model, test, train_args.model_error_significance)
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
        if not found:
            return None
        return model, gamma, delta

    def pass_criteria(self, model, test, model_error_significance): # TODO: using difference from passive is not a great criteria since the active follows a difference loss once interaction is added in
        forward_error, passive_error = model.assess_error(test)
        print(forward_error, passive_error, model_error_significance)
        passed = forward_error < (passive_error - model_error_significance)
        return passed, forward_error-passive_error

    def train(self, cfs, rollouts, train_args, entity_selection, name):
        print("Training ", cfs.object(), "-> ", name)
        self.model_args['gamma'] = entity_selection
        self.model_args['delta'] = self.em.create_entity_selector([name])
        dma = default_model_args(train_args.predict_dynamics, entity_selection.output_size(), self.model_args['delta'].output_size())
        self.model_args['normalization_function'] = dma['normalization_function']
        print(entity_selection.output_size())
        self.model_args['num_inputs'] = self.model_args['gamma'].output_size()
        self.model_args['num_outputs'] = self.model_args['delta'].output_size()
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

