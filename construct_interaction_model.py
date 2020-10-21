import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from EnvironmentModels.environment_model import ModelRollouts
from Environments.SelfBreakout.breakout_screen import Screen
from Counterfactual.passive_active_dataset import HackedPassiveActiveDataset
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption
from DistributionalModels.InteractionModels.interaction_model import HackedInteractionModel

if __name__ == '__main__':
    args = get_args()
    data = read_obj_dumps(args.dataset_dir, i=-1, rng = args.num_frames, filename='object_dumps.txt')
    environment = Screen(frameskip=args.frameskip)
    environment_model = BreakoutEnvironmentModel(environment)
    pa_data = HackedPassiveActiveDataset(environment_model) # change this when it's no longer hacked
    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, None, None, None, environment_model, None, None, "Action", ["Actions"], num_params=environment.num_actions)
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,), num_params=environment.num_actions)}
        graph = OptionGraph(nodes, dict())
    graph.load_environment_model(environment_model)
    rollouts = ModelRollouts(len(data), environment_model.shapes_dict)
    last_state = None
    graph.nodes[args.object].option.set_behavior_epsilon(0)
    for data_dict, next_data_dict in zip(data, data[1:]):
        insert_dict, last_state = environment_model.get_insert_dict(data_dict, next_data_dict, last_state, typed=True)
        rollouts.append(**insert_dict)
    identifiers, passive, contingent_active, irrelevant = pa_data.generate_dataset(rollouts, args.target, [n for n in graph.nodes.values() if n.name != args.target])
    # dataset_model.sample_zero = args.sample_zero
    interaction_model = HackedInteractionModel(environment_model = environment_model, target_name=args.target, contingent_nodes=[n for n in graph.nodes.values() if n.name != args.target])
    interaction_model.train(identifiers, passive, contingent_active, irrelevant, rollouts)
    interaction_model.save(args.dataset_dir)
    print("forward", interaction_model.forward_model)
    print("state", interaction_model.state_model)
    print("state counts", interaction_model.difference_counts)
    print(interaction_model.input_mask, interaction_model.output_mask)
    # for (diff, mask), (out, mask), (pdif, pmask), count in zip(dataset_model.observed_differences["Ball"], dataset_model.observed_outcomes["Ball"], dataset_model.observed_outcomes["Paddle"], dataset_model.outcome_counts["Ball"]):
    #     print(diff, out, pdif, mask, diff - pdif, count)
    # print(dataset_model.observed_differences["Ball"])
    # save_to_pickle(os.path.join(args.dataset_dir, "dataset_model.pkl"), dataset_model)