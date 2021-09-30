import os, torch
from arguments import get_args
from file_management import read_obj_dumps, read_action_dumps, load_from_pickle, save_to_pickle, numpy_factored
from Rollouts.param_buffer import SamplerBuffer

from Networks.input_norm import InterInputNorm, PointwiseNorm, PointwiseConcatNorm
from DistributionalModels.InteractionModels.samplers import PredictiveSampling

from Environments.environment_initializer import initialize_environment
from EnvironmentModels.environment_normalization import hardcode_norm
from Options.option_graph import OptionGraph, OptionNode, load_graph, OptionEdge, graph_construct_load

from Networks.network import pytorch_model
import numpy as np
import sys

actions = np.array([[-1,-1], [-2,-1], [-2,1], [-1,1]])

def action_assignment(factored_state):
    idx = -1
    if 68 <= factored_state["Ball"][0] <= 69 and factored_state["Ball"][2] < 0:
        action_vals = np.linalg.norm(actions - factored_state["Ball"][2:4], axis=1)
        idx = np.argmin(action_vals)
        print(factored_state["Ball"], idx)
    return idx

def param_assignment(param):
    action_vals = np.linalg.norm(actions - param[2:4], axis=1)
    idx = np.argmin(action_vals)
    return idx

if __name__ == '__main__':
    print(sys.argv)
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=3000, linewidth=120)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, environment_model, args = initialize_environment(args, set_save=False)
    args.environment, args.environment_model = environment, environment_model
    graph, controllable_feature_selectors, args = graph_construct_load(args, environment, environment_model)
    graph.load_environment_model(environment_model)
    data = read_obj_dumps(args.record_rollouts, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    try:
        params = read_action_dumps(args.record_rollouts, i=-1, rng=args.num_frames, filename='param_dumps.txt')
        has_params = True
    except OSError as e:
        has_params = False

    entity_selector = environment_model.create_entity_selector(args.train_pair)
    target_selector = environment_model.create_entity_selector([args.object])

    last_state = None
    ACT_DIM = 4
    normalization_function = InterInputNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
    input_vals = hardcode_norm(args.hardcode_norm[0], ["Ball", "Block"])
    normalization_function.assign_mean_var(*input_vals)
    sampler_type = 'bin'
    kwargs = dict()
    kwargs["num_bases"] = [1, 12, 12, 12, 1, 1, 6, 1, 1, 1]
    kwargs["variance"] = .1
    kwargs["range"] = .7
    kwargs["basis_function"] = "rbf"
    sampler = PredictiveSampling(sampler_type='flat', sampler_train_rate=1, buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector,
                                sampler_grad_epoch=args.grad_epoch, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]],
                                init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, init_form=args.init_form, activation=args.activation,
                                hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, normalization_function=normalization_function, base_variance=args.base_variance,
                                mask=torch.zeros(1), action_prediction=1, noise_actions=0.0, **kwargs)

    # normalization_function = PointwiseConcatNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
    # first_norm_vals = hardcode_norm(args.hardcode_norm[0], ["Hot", "Ball"])
    # target_vals = hardcode_norm(args.hardcode_norm[0], ["Block"])
    # normalization_function.assign_mean_var(*(*first_norm_vals, *target_vals))
    # sampler = PredictiveSampling(sampler_type='bin', sampler_train_rate=1, buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector,
    #                             sampler_grad_epoch=args.grad_epoch, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]],
    #                             init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, init_form=args.init_form, activation=args.activation,
    #                             hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, normalization_function=normalization_function, base_variance=args.base_variance,
    #                             mask=torch.zeros(1), action_prediction=0, noise_actions=0)

    if not args.load_intermediate:
        for i in range(len(data)):
            data_dict = data[i]
            if has_params: param_val = np.array(params[i])
            batch = sampler.assign_data({'factored_state': numpy_factored(data_dict)}, data_dict["Action"][0])
            # batch.act = action_assignment(numpy_factored(data_dict))
            if has_params:
                batch.act = param_assignment(param_val)
            else:
                batch.act = action_assignment(numpy_factored(data_dict))
            # print(batch.act, param_val[2:4], data_dict["Ball"][:4])
            sampler.aggregate(batch)
        if args.save_intermediate:
            save_to_pickle("data/temp/buffer.pkl", sampler.buffer)
            save_to_pickle("data/temp/test_buffer.pkl", sampler.test_buffer)
    else:
        sampler.buffer = load_from_pickle("data/temp/buffer.pkl")
        sampler.test_buffer = load_from_pickle("data/temp/test_buffer.pkl")

    for i in range(len(sampler.buffer)):
        print(i, sampler.buffer.obs[i], sampler.buffer.act[i], sampler.buffer.instance_binary[i], sampler.buffer.target[i])
    if args.cuda: sampler.cuda()
    for i in range(args.num_iters):
        total_loss, pb, ib, a, ta = sampler.update(np.zeros(1), np.zeros(1))
        print("train loss", i, total_loss)
        print("train results", torch.cat([pb[:50], ib[:50], a[:50], ta[:50]], dim=1))
        test_loss, pb, ib, a, ta = sampler.assess_test()
        print("test loss", i, test_loss)
        print("test results", torch.cat([pb[:100], ib[:100], a[:100], ta[:100]], dim=1))
