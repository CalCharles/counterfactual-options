from Environments.environment_initializer import initialize_environment
from Rollouts.rollouts import ObjDict
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Network.networm import pytorch_model
from Test.test_util import compare_full_prediction
import cv2
import numpy as np

def test_passive_model():
    args = ObjDict()
    args.seed = 0
    args.env = "RoboPushing"
    args.record_rollouts = ""
    args.save_recycle = -1
    args.save_raw = False
    args.drop_stopping = False
    args.true_environment = False
    args.predict_dynamics = False
    args.policy_type = "basic"
    args.hidden_sizes = [1024, 1024]
    args.interaction_binary = [-1, -13, -13]
    args.interaction_prediction = 0.3
    args.init_form = "none"
    args.activation = 'relu'
    args.action_shift = True
    args.base_variance = 0.01
    args.multi_instanced = False
    args.lr = 1e-3
    args.eps = 1e-5
    args.betas=(0.9, 0.999)
    args.weight_decay = 0
    args.batch_size = 10
    args.num_iters = 5

    environment, environment_model, args = initialize_environment(args)

    data = read_obj_dumps(pth="data/unit_test/robopush_train/", i=-1, rng = 3000, filename='object_dumps.txt')
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, instanced=True, action_shift = args.action_shift)
        rollouts.append(**insert_dict)

    aosize = 0
    model_args.hidden_sizes, model_args.interaction_binary, model_args.interaction_prediction, model_args.init_form, model_args.activation = args.hidden_sizes, args.interaction_binary, args.interaction_prediction, args.init_form, args.activation
    model_args['controllable'], model_args['environment_model'] = controllable_feature_selectors, environment_model
    model_name = "Gripper" + "->"+ "Block"
    model_args = default_model_args(args.predict_dynamics, args.policy_type)
    model_args['name'] = model_name
    model_args['gamma'] = environment_model.create_entity_selector(["Block"])
    model_args['delta'] = environment_model.create_entity_selector(["Block"])
    model_args['object_dim'] = environment_model.object_sizes["Block"]
    model_args['output_dim'] = environment_model.object_sizes["Block"]
    model_args['first_obj_dim'] = environment_model.object_sizes["Block"]
    nout = environment_model.object_sizes["Block"] * environment_model.object_num["Block"]
    nin = environment_model.object_sizes["Block"] * environment_model.object_num["Block"] + nout
    input_norm_fun = InterInputNorm()
    input_norm_fun.compute_input_norm(entity_selection(rollouts.get_values("state")))
    delta_norm_fun = InterInputNorm()
    delta_norm_fun.compute_input_norm(model_args['delta'](rollouts.get_values("state")))
    model_args['normalization_function'] = input_norm_fun#nflen(nin)
    model_args['delta_normalization_function'] = delta_norm_fun#nflen(nout) if not args.predict_dynamics else nf5
    model_args['base_variance'] = args.base_variance

    model_args['num_inputs'] = model_args['gamma'].output_size()
    model_args['num_outputs'] = model_args['delta'].output_size()
    model_args['multi_instanced'] = args.multi_instanced
    model = interaction_models[model_args['model_type']](**model_args)

    model.forward_model = torch.load("data/unit_test/interaction_test/forward_model.pt")
    model.interaction_model = torch.load("data/unit_test/interaction_test/interaction_model.pt")
    model.passive_model = torch.load("data/unit_test/interaction_test/passive_model.pt")

    # define names
    model.control_feature = control # the name of the controllable object
    model.controllers = controllers
    control_name = model.control_feature
    model.target_name = target_name
    model.name = control + "->" + target_name
    model.predict_dynamics = args.predict_dynamics
    
    # initialize the optimizers
    active_optimizer = optim.Adam(model.forward_model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
    passive_optimizer = optim.Adam(model.passive_model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
    interaction_optimizer = optim.Adam(model.interaction_model.parameters(), args.lr, eps=args.eps, betas=args.betas, weight_decay=args.weight_decay)
    
    # compute maximum and minimum of target values
    minmax = model.delta(rollouts.get_values('state'))
    model.control_min = np.amin(pytorch_model.unwrap(minmax), axis=1)
    model.control_max = np.amax(pytorch_model.unwrap(minmax), axis=1)

    # Computes the target normalization value, get normalization values
    output_norm_fun = InterInputNorm()
    output_norm_fun.compute_input_norm(model.delta(model.get_targets(rollouts)))
    model.output_normalization_function = output_norm_fun
    model.nf = model.output_normalization_function # temporarily to save length
    model.rv = model.output_normalization_function.reverse # same as above

    # pre-initialize batches because it accelerates time
    batchvals = type(rollouts)(args.batch_size, rollouts.shapes)
    pbatchvals = type(rollouts)(args.batch_size, rollouts.shapes)

    idxes_sets = [list(range(10)), list(range(10,20)), list(range(20,30)), list(range(30,40)), list(range(40,50))]
    outputs = model._train_passive(rollouts, args, batchvals, active_optimizer, passive_optimizer, idxes_sets=idxes_sets)
    target_passive = load_from_pickle("data/unit_test/robopush_train/passive_output.pkl") # contains active and passive outputs
    target_model = torch.load("data/unit_test/robopush_train/target_interaction_model.pt")
    for o, tp in zip(outputs, target_passive):
    	print("passive output difference (should be zero):", compare_full_prediction(o[0], tp[0]), compare_full_prediction(o[1], tp[1]))
    for p, tp in zip(model.passive_model.get_parameters(), target_model.passive_model.get_parameters()):
    	print("passive weight difference: ", np.linalg.norm(pytorch_model.unwrap(p) - pytorch_model.unwrap(tp)))

    interaction_schedule = lambda i: np.power(0.5, (i/2))
    passive_error_all = torch.ones(rollouts.filled)
    weights, use_weights = torch.ones(rollouts.filled) / rollouts.filled, torch.ones(rollouts.filled) / rollouts.filled
    total_live, total_dead = 0, 0
    outputs = model._train_combined(rollouts, train_args, batchvals, 
    None, weights, use_weights, passive_error_all, interaction_schedule,
    active_optimizer, passive_optimizer, interaction_optimizer)        
    target_passive = load_from_pickle("data/unit_test/robopush_train/active_output.pkl") # contains active and passive outputs
    for o, tp in zip(outputs, target_passive):
    	print("active output difference (should be zero):", compare_full_prediction(o[0], tp[0]), compare_full_prediction(o[1], tp[1]) np.linalg.norm(pytorch_model.unwrap(o[2] - tp[2])))
    for p, tp in zip(model.forward_model.get_parameters(), target_model.forward_model.get_parameters()):
    	print("active weight difference: ", np.linalg.norm(pytorch_model.unwrap(p) - pytorch_model.unwrap(tp)))
