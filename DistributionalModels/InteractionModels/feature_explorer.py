# Feature Search Function
import numpy as np
import os, cv2, time, copy
import torch
from collections import Counter
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts
from DistributionalModels.InteractionModels.interaction_model import interaction_models, default_model_args, nf5, nflen
from Networks.input_norm import InterInputNorm, PointwiseNorm, PointwiseConcatNorm
from EnvironmentModels.environment_normalization import hardcode_norm

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


        # creates a list of all the controllable entities from the controllable features, where each
        # entity only exists in cfslist once, based on the first time it appears.
        cfsnames = [cfs.feature_selector.get_entity()[0] for cfs in self.cfs]
        cfsdict = dict()
        cfslist = list()
        for cn, cfs in zip(cfsnames, self.cfs):
            print(cn)
            if cn not in cfslist: cfslist.append(cn)
            if cn in cfsdict: cfsdict[cn].append(cfs)
            else: cfsdict[cn] = [copy.deepcopy(cfs)]

        # additional is extra objects added to the input state. Necessary for pushing
        additional = [[]] + [[cfs] for cfs in cfslist]

        # test pair allows for training only one pair instead of greedy search
        if len(train_args.train_pair) > 0:
            if len(train_args.train_pair) > 2:
                additional = [train_args.train_pair[1:-1]]
            cfslist = [train_args.train_pair[0]]
            target_names = [train_args.train_pair[-1]]
        else:
            target_names = self.em.object_names

        cfslist.reverse()
        print("controllable objects", [c for c in cfslist])
        # HACKED LINE TO SPEED UP TRAINING
        for cfs in cfslist:
            controllable_entity = cfs
            if controllable_entity not in gamma_tested:
                for additional_feature in additional:
                    if len(additional_feature) > 0 and additional_feature[0] == cfs:
                        continue # don't add additionally the same feature being tested, TODO: not the same target feature also
                    delta_tested = set()
                    # HACKED LINE TO SPEED UP TRAINING
                    for name in target_names:
                        if name != controllable_entity and name not in delta_tested and (controllable_entity, name) not in self.graph.edges:
                            names = [controllable_entity] + additional_feature + [name]
                            entity_selection = self.em.create_entity_selector(names)
                            model, test, gamma_new, delta_new = self.train(cfs, cfsdict[cfs], additional_feature, rollouts, train_args, entity_selection, name)
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

    def train(self, cfs, cfss, additional_object, rollouts, train_args, entity_selection, name):
        print("Edge ", cfs, "-> ", name)
        aosize = 0
        model_name = cfs + "->"+ name
        if len(additional_object) > 0:
            additional_obj = additional_object[0]
            ao_size = self.em.object_sizes[additional_obj] * self.em.object_num[additional_obj]
            print("Training ", cfs, " + ", additional_obj, " -> ", name)
            model_name = cfs + "+" + additional_obj+ "->" + name
        self.model_args['name'] = model_name
        self.model_args['gamma'] = entity_selection
        self.model_args['delta'] = self.em.create_entity_selector([name])
        self.model_args['object_dim'] = self.em.object_sizes[name]
        self.model_args['output_dim'] = self.em.object_sizes[name]
        self.model_args['first_obj_dim'] = self.em.object_sizes[cfs]
        nout = self.em.object_sizes[name] * self.em.object_num[name]
        nin = self.em.object_sizes[cfs] * self.em.object_num[cfs] + nout
        if train_args.multi_instanced:
            input_norm_fun = PointwiseConcatNorm(object_dim = self.model_args['object_dim'], first_obj_dim = self.model_args['first_obj_dim'])
            delta_norm_fun = PointwiseNorm(object_dim = self.model_args['object_dim'])
        else:            
            input_norm_fun = InterInputNorm()
            delta_norm_fun = InterInputNorm()
        if len(train_args.hardcode_norm) > 0:
            if train_args.multi_instanced:
                first_norm = hardcode_norm(train_args.hardcode_norm[0], [cfs])
                out_norm = hardcode_norm(train_args.hardcode_norm[0], [name])
                input_norm_fun.assign_mean_var(*(*first_norm, *out_norm))
                delta_norm_fun.assign_mean_var(*out_norm)
            else:
                gamma_names = [cfs] + additional_object + [name]
                gamma_norm = hardcode_norm(train_args.hardcode_norm[0], gamma_names)
                delta_norm = hardcode_norm(train_args.hardcode_norm[0], [name])
                input_norm_fun.assign_mean_var(*gamma_norm)
                delta_norm_fun.assign_mean_var(*delta_norm)
        else:
            input_norm_fun.compute_input_norm(entity_selection(rollouts.get_values("state")))
            delta_norm_fun.compute_input_norm(self.model_args['delta'](rollouts.get_values("state")))
        self.model_args['normalization_function'] = input_norm_fun#nflen(nin)
        self.model_args['delta_normalization_function'] = delta_norm_fun#nflen(nout) if not train_args.predict_dynamics else nf5
        self.model_args['base_variance'] = train_args.base_variance

        print(entity_selection.output_size())
        self.model_args['num_inputs'] = self.model_args['gamma'].output_size()
        self.model_args['num_outputs'] = self.model_args['delta'].output_size()
        self.model_args['multi_instanced'] = train_args.multi_instanced
        model = interaction_models[self.model_args['model_type']](**self.model_args)
        print(model)
        if not train_args.load_intermediate:
            train, test = rollouts.split_train_test(train_args.ratio)
            train.cpu(), test.cpu()
        else:
            train = load_from_pickle("data/temp/train.pkl")
            test = load_from_pickle("data/temp/test.pkl")
        if train_args.save_intermediate:
            save_to_pickle("data/temp/train.pkl", train)
            save_to_pickle("data/temp/test.pkl", test)
        print(train.filled, rollouts.filled)
        train.cuda(), test.cuda()
        model.train(train, test, train_args, control=cfs, controllers=cfss, target_name=name)
        return model, test, self.model_args['gamma'], self.model_args['delta']

