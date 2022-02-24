from EnvironmentModels.environment_normalization import hardcode_norm, position_mask
import numpy as np
import os, cv2, time, copy, psutil
import torch
from collections import Counter
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts
from DistributionalModels.InteractionModels.interaction_model import interaction_models, default_model_args, nf5, nflen
from Networks.input_norm import InterInputNorm, PointwiseNorm, PointwiseConcatNorm
from EnvironmentModels.environment_normalization import hardcode_norm, position_mask
from DistributionalModels.InteractionModels.InteractionTraining.train_full import train_full
from DistributionalModels.InteractionModels.InteractionTraining.assessment_functions import assess_error


def init_model(model_args, em, cfs, cfss, additional_object, train, test, train_args, target):
    print("Edge ", cfs, "-> ", target)
    aosize = 0
    model_name = cfs + "->"+ target
    train_args.position_mask = position_mask(train_args.env)
    if len(additional_object) > 0: 
        model_args['additional_dim'] = 0
        model_args['last_additional'] = em.object_sizes[additional_object[-1]] # TODO: only the last additional object can be multi_instanced
        print("Training ", cfs, " + ", additional_object, " -> ", target)
        additional_str = ""
        for additional_obj in additional_object:
            ao_size = em.object_sizes[additional_obj] * em.object_num[additional_obj]
            additional_str += additional_obj + "+"
            model_args['additional_dim'] += em.object_sizes[additional_obj]
        model_name = cfs + "+" + additional_str[:-1]+ "->" + target
    model_args['name'] = model_name
    gamma_names = [cfs] + additional_object + [target]
    if train_args.param_contained:
        gamma_names = [cfs] + [target] + additional_object
    add_relative = train_args.observation_setting[0] == 1
    print(gamma_names)
    entity_selection = em.create_entity_selector(gamma_names, add_relative=add_relative)
    model_args['gamma'] = entity_selection
    model_args['delta'] = em.create_entity_selector([target])
    model_args.zeta = em.create_entity_selector([cfs])
    model_args['object_dim'] = em.object_sizes[target]
    model_args['output_dim'] = em.object_sizes[target]
    model_args['first_obj_dim'] = em.object_sizes[cfs] if not train_args.param_contained else em.object_sizes[cfs] + em.object_sizes[target] 
    model_args['aggregate_final'] = not train_args.multi_instanced
    nout = em.object_sizes[target] * em.object_num[target]
    nin = em.object_sizes[cfs] * em.object_num[cfs] + nout
    print(train_args.multi_instanced, train_args.hardcode_norm, nin, nout, train_args)
    # TODO: add_relative does not play nice with hardcode norms at the moment
    if train_args.multi_instanced:
        input_norm_fun = PointwiseConcatNorm(object_dim = model_args['object_dim'], first_obj_dim = model_args['first_obj_dim'])
        delta_norm_fun = PointwiseNorm(object_dim = model_args['object_dim'])
    elif train_args.instanced_additional:
        model_args['first_obj_dim'] = model_args['first_obj_dim'] + model_args['additional_dim'] - model_args['last_additional']
        input_norm_fun = PointwiseConcatNorm(object_dim = model_args['last_additional'], first_obj_dim = model_args['first_obj_dim'])
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
        input_norm_fun.compute_input_norm(entity_selection(train.get_values("state")))
        delta_norm_fun.compute_input_norm(model_args['delta'](train.get_values("state")))
    model_args['normalization_function'] = input_norm_fun#nflen(nin)
    model_args['delta_normalization_function'] = delta_norm_fun#nflen(nout) if not train_args.predict_dynamics else nf5
    model_args['base_variance'] = train_args.base_variance[0] if len(train_args.base_variance) == 1 else train_args.base_variance
    print(train_args.base_variance)
    print(entity_selection.output_size())
    model_args['num_inputs'] = model_args['gamma'].output_size()
    model_args['num_outputs'] = model_args['delta'].output_size()
    model_args['multi_instanced'] = train_args.multi_instanced
    model_args['instanced_additional'] = train_args.instanced_additional
    if train_args.instanced_additional:
        model_args['passive_class'] = 'basic' # assumes that the passive model class is 
    model = interaction_models[model_args['model_type']](**model_args)
    return model