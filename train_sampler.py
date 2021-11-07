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

class binary_selector():
    def __init__(self, entity_selector, first_dim=0):
        self.first_dim = first_dim
        self.flat_features = entity_selector.flat_features
        self.factored_features = entity_selector.factored_features
        self.feature_match = entity_selector.feature_match # a dict of name: [[factored feature, flat feature]...]
        self.names= entity_selector.names
        self.add_relative = entity_selector.add_relative
        self.relative_indexes = entity_selector.relative_indexes
        self.relative_flat_indexes = entity_selector.relative_flat_indexes
        self.entity_selector = entity_selector
        self.object_dim=5

    def __call__(self, states):
        estate = self.entity_selector(states)
        idxes = [self.first_dim + 4 + i * 5 for i in range((estate.shape[-1] - self.first_dim) // 5)]
        return np.concatenate([estate[:self.first_dim], estate[idxes]], axis=-1)

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
    try: # adjusts for offset
        params, param_idxes, term_info = read_action_dumps(args.record_rollouts, i=-1, rng=args.num_frames + 1000, filename='param_dumps.txt', indexed=True)
        has_params = True
    except OSError as e:
        has_params = False

    entity_selector = environment_model.create_entity_selector(args.train_pair)
    target_selector = environment_model.create_entity_selector([args.object])

    # sampler for big block domain
    if args.sampler_type == "flat":
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
                                    init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, init_form=args.init_form, activation=args.activation, has_params=has_params,
                                    hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, normalization_function=normalization_function, base_variance=args.base_variance,
                                    mask=torch.zeros(1), action_prediction=1, noise_actions=0.0 **kwargs)

    # # POINTWISE CONCAT FOR MULTIBLOCK
    elif args.sampler_type == "bin":
        if has_params:
            terminations = np.array(term_info)[...,1]
            param_term_dict = dict()
            for i in range(len(param_idxes)):
                # print(i, len(param_idxes), len(params), len(terminations), param_idxes[i])
                param_term_dict[param_idxes[i]] = (params[i], terminations[i])
        last_state = None
        ACT_DIM = 4
        delay_act = True
        normalization_function = PointwiseConcatNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
        input_vals_first = hardcode_norm(args.hardcode_norm[0], ["Hot", "Ball"])
        input_vals_obj = hardcode_norm(args.hardcode_norm[0], ["Block"])
        normalization_function.assign_mean_var(*(*input_vals_first, *input_vals_obj))
        kwargs = dict()
        sampler = PredictiveSampling(sampler_type='bin', sampler_train_rate=1, buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector,
                                    sampler_grad_epoch=args.grad_epoch, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]],
                                    init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, init_form=args.init_form, activation=args.activation, has_params=has_params,
                                    hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, normalization_function=normalization_function, base_variance=args.base_variance,
                                    mask=torch.zeros(1), action_prediction=0, noise_actions=0.0, diff_binary=True, **kwargs)

    # CONVOLUTIONAL FOR MULTIBLOCK
    elif args.sampler_type == "binconv":
        entity_selector = binary_selector(entity_selector, 5)

        if has_params:
            terminations = np.array(term_info)[...,1]
            param_term_dict = dict()
            for i in range(len(param_idxes)):
                # print(i, len(param_idxes), len(params), len(terminations), param_idxes[i])
                param_term_dict[param_idxes[i]] = (params[i], terminations[i])
        last_state = None
        ACT_DIM = 4
        delay_act = True
        check_on_binary = True # change this to cchange the binary used
        normalization_function = PointwiseConcatNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
        input_vals_first = hardcode_norm(args.hardcode_norm[0], ["Hot", "Ball"])
        input_vals_obj = hardcode_norm(args.hardcode_norm[0], ["block_binary"])
        normalization_function.assign_mean_var(*(*input_vals_first, *input_vals_obj))
        kwargs = dict()
        sampler = PredictiveSampling(sampler_type='binconv',  num_objects= 100, input_dims=[5,20,1], stride=1, padding=1, kernel=3, hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, init_form=args.init_form, activation=args.activation,
        buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]], has_params=has_params,
        sampler_train_rate=1, sampler_grad_epoch=args.grad_epoch, action_prediction=0, noise_actions=0.0,  diff_binary=True, 
        init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, mask=torch.zeros(1), check_on_binary=args.check_on_binary,
        normalization_function=normalization_function, base_variance=args.base_variance, **kwargs)

    # TRANSFORMER FOR MULTIBLOCK
    elif args.sampler_type == "bintrans":
        if has_params:
            terminations = np.array(term_info)[...,1]
            param_term_dict = dict()
            for i in range(len(param_idxes)):
                # print(i, len(param_idxes), len(params), len(terminations), param_idxes[i])
                param_term_dict[param_idxes[i]] = (params[i], terminations[i])
        last_state = None
        ACT_DIM = 4
        delay_act = True
        check_on_binary = True
        normalization_function = PointwiseConcatNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
        input_vals_first = hardcode_norm(args.hardcode_norm[0], ["Hot", "Ball"])
        input_vals_obj = hardcode_norm(args.hardcode_norm[0], ["Block"])
        normalization_function.assign_mean_var(*(*input_vals_first, *input_vals_obj))
        kwargs = dict()
        sampler = PredictiveSampling(sampler_type='bintrans',  num_objects= 100, hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, init_form=args.init_form, activation=args.activation,
        buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]], has_params=has_params,
        sampler_train_rate=1, sampler_grad_epoch=args.grad_epoch, action_prediction=0, noise_actions=0.0,  diff_binary=True, 
        init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, mask=torch.zeros(1), check_on_binary=args.check_on_binary,
        normalization_function=normalization_function, base_variance=args.base_variance, **kwargs)

    # # TRANSFORMER FOR MULTIBLOCK
    elif args.sampler_type == "binmht":
        if has_params:
            terminations = np.array(term_info)[...,1]
            param_term_dict = dict()
            for i in range(len(param_idxes)):
                # print(i, len(param_idxes), len(params), len(terminations), param_idxes[i])
                param_term_dict[param_idxes[i]] = (params[i], terminations[i])
        last_state = None
        ACT_DIM = 4
        delay_act = True
        check_on_binary = True
        normalization_function = PointwiseConcatNorm(object_dim = environment_model.object_sizes[args.object], first_obj_dim = ACT_DIM + environment_model.object_sizes[args.train_pair[0]])
        input_vals_first = hardcode_norm(args.hardcode_norm[0], ["Hot", "Ball"])
        input_vals_obj = hardcode_norm(args.hardcode_norm[0], ["Block"])
        normalization_function.assign_mean_var(*(*input_vals_first, *input_vals_obj))
        kwargs = dict()
        sampler = PredictiveSampling(sampler_type='binmht',  num_objects= 100, hidden_sizes=args.hidden_sizes, use_layer_norm=args.use_layer_norm, init_form=args.init_form, activation=args.activation, num_heads = 10,
        buffer_len=len(data), entity_selector=entity_selector, target_selector=target_selector, object_dim=environment_model.object_sizes[args.object], first_obj_dim=ACT_DIM + environment_model.object_sizes[args.train_pair[0]], has_params=has_params,
        sampler_train_rate=1, sampler_grad_epoch=args.grad_epoch, action_prediction=0, noise_actions=0.0,  diff_binary=True, 
        init_state={'factored_state': numpy_factored(data[0])}, num_actions=ACT_DIM, mask=torch.zeros(1), check_on_binary=args.check_on_binary,
        normalization_function=normalization_function, base_variance=args.base_variance, **kwargs)



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

        first_term, first_aggregate=False, True
        current_act, next_act, next_next_act = None, None, None
        for i in range(len(data)):
            data_dict = data[i]
            if int(np.array(data_dict["ITR"]).squeeze()) not in param_term_dict:
                break
            if has_params: param_val = np.array(param_term_dict[int(np.array(data_dict["ITR"]).squeeze())][0])
            batch = sampler.assign_data({'factored_state': numpy_factored(data_dict)}, data_dict["Action"][0])
            # batch.act = action_assignment(numpy_factored(data_dict))
            if has_params:
                batch.terminate = param_term_dict[int(data_dict["ITR"].squeeze())][1]
                if batch.terminate:
                    first_term = True
                    current_act = param_assignment(param_val)#next_act
                    next_act = param_assignment(param_val)
                batch.act = current_act
            else:
                batch.terminate = batch.done
                batch.act = action_assignment(numpy_factored(data_dict))
            if first_term and batch.act is not None: # skip all the ones up to this one
                # print(batch.terminate, batch.act, param_val[2:4], data_dict["Ball"][:4])
                sampler.aggregate(batch, first=first_aggregate)
                first_aggregate = False
            last_first_term = first_term
        if args.save_intermediate:
            save_to_pickle("/hdd/datasets/counterfactual_data/temp/buffer.pkl", sampler.buffer)
            save_to_pickle("/hdd/datasets/counterfactual_data/temp/test_buffer.pkl", sampler.test_buffer)
    else:
        sampler.buffer = load_from_pickle("/hdd/datasets/counterfactual_data/temp/buffer.pkl")
        sampler.test_buffer = load_from_pickle("/hdd/datasets/counterfactual_data/temp/test_buffer.pkl")
    # for i in range(len(sampler.buffer)):
    #     print(i, sampler.buffer.obs[i], sampler.buffer.act[i], sampler.buffer.instance_binary[i], sampler.buffer.target[i])
    if args.cuda: sampler.cuda()
    for i in range(args.num_iters):
        total_loss, pb, ib, a, ta = sampler.update(np.zeros(1), np.zeros(1))
        pb[pb < .5] = 0
        ib[ib < .5] = 0
        print("train loss", i, total_loss)
        print(pb[0])
        print("train results", pb[:50].nonzero(), ib[:50].nonzero())
        test_loss, pb, ib, a, ta = sampler.assess_test()
        pb[pb < .5] = 0
        ib[ib < .5] = 0
        print("test loss", i, test_loss)
        print("test results", pb[:100].nonzero(), ib[:100].nonzero())
