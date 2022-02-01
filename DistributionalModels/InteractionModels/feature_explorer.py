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
from EnvironmentModels.environment_normalization import hardcode_norm, position_mask

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
                            print(cfsdict)
                            model, test, gamma_new, delta_new = self.train(cfs, cfsdict[cfs], additional_feature, rollouts, train_args, name)
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

    def train(self, cfs, cfss, additional_object, rollouts, train_args, target):
        print("Edge ", cfs, "-> ", target)
        aosize = 0
        model_name = cfs + "->"+ target
        train_args.position_mask = position_mask(train_args)
        if len(additional_object) > 0: 
            self.model_args['additional_dim'] = 0
            self.model_args['last_additional'] = self.em.object_sizes[additional_object[-1]] # TODO: only the last additional object can be multi_instanced
            print("Training ", cfs, " + ", additional_object, " -> ", target)
            additional_str = ""
            for additional_obj in additional_object:
                ao_size = self.em.object_sizes[additional_obj] * self.em.object_num[additional_obj]
                additional_str += additional_obj + "+"
                self.model_args['additional_dim'] += self.em.object_sizes[additional_obj]
            model_name = cfs + "+" + additional_str[:-1]+ "->" + target
        self.model_args['name'] = model_name
        gamma_names = [cfs] + additional_object + [target]
        if train_args.param_contained:
            gamma_names = [cfs] + [target] + additional_object
        add_relative = train_args.observation_setting[0] == 1
        print(gamma_names)
        entity_selection = self.em.create_entity_selector(gamma_names, add_relative=add_relative)
        self.model_args['gamma'] = entity_selection
        self.model_args['delta'] = self.em.create_entity_selector([target])
        self.model_args.zeta = self.em.create_entity_selector([cfs])
        self.model_args['object_dim'] = self.em.object_sizes[target]
        self.model_args['output_dim'] = self.em.object_sizes[target]
        self.model_args['first_obj_dim'] = self.em.object_sizes[cfs] if not train_args.param_contained else self.em.object_sizes[cfs] + self.em.object_sizes[target] 
        self.model_args['aggregate_final'] = not train_args.multi_instanced
        nout = self.em.object_sizes[target] * self.em.object_num[target]
        nin = self.em.object_sizes[cfs] * self.em.object_num[cfs] + nout
        print(train_args.multi_instanced, train_args.hardcode_norm, nin, nout, train_args)
        # TODO: add_relative does not play nice with hardcode norms at the moment
        if train_args.multi_instanced:
            input_norm_fun = PointwiseConcatNorm(object_dim = self.model_args['object_dim'], first_obj_dim = self.model_args['first_obj_dim'])
            delta_norm_fun = PointwiseNorm(object_dim = self.model_args['object_dim'])
        elif train_args.instanced_additional:
            self.model_args['first_obj_dim'] = self.model_args['first_obj_dim'] + self.model_args['additional_dim'] - self.model_args['last_additional']
            input_norm_fun = PointwiseConcatNorm(object_dim = self.model_args['last_additional'], first_obj_dim = self.model_args['first_obj_dim'])
            delta_norm_fun = InterInputNorm()
        else:            
            input_norm_fun = InterInputNorm()
            delta_norm_fun = InterInputNorm()
        print(train_args.hardcode_norm)
        if len(train_args.hardcode_norm) > 0:
            if train_args.multi_instanced or train_args.instanced_additional:
                first_norm = hardcode_norm(train_args.hardcode_norm[0], [cfs])
                target_norm = hardcode_norm(train_args.hardcode_norm[0], [target])
                delta_norm_fun.assign_mean_var(*target_norm)
                if train_args.instanced_additional:
                    additional_norm = hardcode_norm(train_args.hardcode_norm[0], [additional_object[-1]])
                    if len(additional_object) > 1:
                        anorm = hardcode_norm(train_args.hardcode_norm[0], additional_object[:-1])
                        first_norm = (np.concatenate([first_norm[0], target_norm[0], anorm[0]]), np.concatenate([first_norm[1], target_norm[1], anorm[1]]), np.concatenate([first_norm[2], target_norm[2], anorm[2]]))
                    else:
                        first_norm = (np.concatenate([first_norm[0], target_norm[0]]), np.concatenate([first_norm[1], target_norm[1]]), np.concatenate([first_norm[2], target_norm[2]]))
                    print(first_norm)
                    input_norm_fun.assign_mean_var(*(*first_norm, *additional_norm))
                    print(additional_norm)
                if train_args.multi_instanced:
                    input_norm_fun.assign_mean_var(*(*first_norm, *target_norm))
            else:
                gamma_names = [cfs] + additional_object + [target]
                gamma_norm = hardcode_norm(train_args.hardcode_norm[0], gamma_names)
                delta_norm = hardcode_norm(train_args.hardcode_norm[0], [target])
                input_norm_fun.assign_mean_var(*gamma_norm)
                delta_norm_fun.assign_mean_var(*delta_norm)
        else:
            input_norm_fun.compute_input_norm(entity_selection(rollouts.get_values("state")))
            delta_norm_fun.compute_input_norm(self.model_args['delta'](rollouts.get_values("state")))
        self.model_args['normalization_function'] = input_norm_fun#nflen(nin)
        self.model_args['delta_normalization_function'] = delta_norm_fun#nflen(nout) if not train_args.predict_dynamics else nf5
        self.model_args['base_variance'] = train_args.base_variance
        print(train_args.base_variance)
        print(entity_selection.output_size())
        self.model_args['num_inputs'] = self.model_args['gamma'].output_size()
        self.model_args['num_outputs'] = self.model_args['delta'].output_size()
        self.model_args['multi_instanced'] = train_args.multi_instanced
        self.model_args['instanced_additional'] = train_args.instanced_additional
        if train_args.instanced_additional:
            self.model_args['passive_class'] = 'basic' # assumes that the passive model class is 
        model = interaction_models[self.model_args['model_type']](**self.model_args)
        print(model)
        print(self.model_args)
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
        model.train(train, test, train_args, control=cfs, controllers=cfss, target_name=target)
        return model, test, self.model_args['gamma'], self.model_args['delta']

