import numpy as np
import os, cv2, time
import torch
from collections import Counter
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts

class CounterfactualStateDataset():
    def __init__(self, environment_model):
        self.model = environment_model
        self.known_objects = ["Action"]

    def generate_dataset(self, rollouts, controllable_feature_selectors):
        '''
        given a set of rollouts without all_state filled out, fill out all_state
        each controllable feature selector only applies to one feature
        '''
        # initialize space for all_state_next features
        total_features = 0
        for cfs in controllable_feature_selectors:
            num_features = cfs.get_num_steps()
            total_features += num_features 
        rollouts.initialize_shape({'all_state_next': (rollouts.length, total_features, rollouts.values.state.shape[1])})

        for i, (odiff, ostate, oaction) in enumerate(zip(rollouts.get_values("state_diff"), rollouts.get_values("state"), rollouts.get_values("action"))):
            j = 0
            for cfs in controllable_feature_selectors: # can only handle varying one cfs at a time
                for v in cfs.get_steps():
                    nstate = ostate.clone()
                    cfs.feature_selector.assign_feature((cfs.feature_selector.flat_feature[0], v))
                    self.model.set_from_factored_state(
                        self.model.unflatten_state(nstate), seed_counter=seed
                    )  # TODO: setting criteria might change (for more limited models
                    self.model.step(oaction)
                    selected_next_state = self.model.flatten_factored_state(self.model.get_factored_state())
                    rollouts.values.all_state_next[i,j] = selected_next_state



class CounterfactualDataset():
    def __init__(self, environment_model):
        self.model = environment_model
        self.known_objects = ["Action"]

    def generate_dataset(self, rollouts, option_node):
        '''
        generates a single step counterfactual rollout, meaning that if the actions from the option level (possibly temporally extended)
        have an effect on the single step outcomes of the state, then the states are added.
        rollouts contains a set of rollouts including the factored states
        option_level includes a level of trained options to check for counterfactual relationships
        returns three components:
            relevant_states: A list of datasets all states traversed by the option, and the state the option was initiated in, for each option (on dataset per option)
            irrelevant_outcomes: a list of datasets of all outcomes where 
        '''
        initialize_rollout = lambda length: ModelRollouts(
                length=length, shapes_dict={
                "state":rollouts.shapes["state"],
                "state_diff":rollouts.shapes["state_diff"],
                "action":option_node.action_shape,
                "done":rollouts.shapes["done"]
                }
            )

        option = option_node.option
        initialize_multiple = lambda :[
            initialize_rollout(int(rollouts.length * option.time_cutoff)) for i in range(option_node.num_params)
        ]
        counter_rollouts = initialize_multiple() # the trajectory experienced by the option
        counter_outcome_rollouts = initialize_multiple() # what happens as a result of running the option to completion
        # collecting rollouts
        k = 0
        seed = 0
        for odiff, ostate in zip(rollouts.get_values("state_diff"), rollouts.get_values("state")):
            for i, (param, mask) in enumerate(option_node.option.get_possible_parameters()):
                factored_ostate = self.model.unflatten_state(ostate)
                self.model.set_from_factored_state(
                    self.model.unflatten_state(ostate), seed_counter=seed
                )  # TODO: setting criteria might change (for more limited models
                # reset internal variables
                action_chain, rlout = option.sample_action_chain(factored_ostate, param)
                option.last_factor = option.get_flattened_object_state(factored_ostate)
                ###############
                done = False
                last_state = self.model.get_factored_state()
                last_diff = odiff
                while not done:
                    state = self.model.get_factored_state()
                    # print(param, option_node.option.iscuda)
                    action_chain, rlout = option.sample_action_chain(state, param)
                    # TODO: handle case where not done, but significant change occurs, and put in counter_outcome_rollouts
                    last_state = state
                    self.model.step(action_chain[0])
                    next_state = self.model.get_factored_state()
                    # print(action_chain, next_state["Paddle"])
                    done, reward, all_reward, option_diff, last_done = option.terminate_reward(next_state, param, action_chain)
                    # print("done", done, option.timer, state["Paddle"], next_state["Paddle"], option_diff)
                    diff = self.model.flatten_factored_state(next_state) - self.model.flatten_factored_state(state)
                    # print(done, option.timer)
                    counter_rollouts[i].append(
                        **{"state": self.model.flatten_factored_state(state), "state_diff": last_diff, "action": action_chain[-1], "done": done}
                    )
                    last_diff = diff
                # The last one goes into the outcome rollouts part
                counter_outcome_rollouts[i].append(
                     **{"state": self.model.flatten_factored_state(next_state), "state_diff": diff, "action": action_chain[-1], "done": done}
                )
                # print(action_chain, param, next_state["Ball"], next_state["Paddle"])
                #     print(action_chain, param, done)
                # print("next")
            seed += 1
            k += 1
        splitted = []
        print(counter_rollouts)
        for cr in counter_rollouts:
            splitted.append(cr.split_trajectories())
        relevant_states = []
        irrelevant_outcomes = initialize_rollout(rollouts.length*option_node.num_params) # only use outcomes, or use all states?
        outcomes = initialize_rollout(rollouts.length*option_node.num_params)
        print(np.array(splitted).shape, [len(splitted[i]) for i in range(len(splitted))])
        for i in range(len(splitted[0])):  # all splitted should have the same length
            ### TODO: commented checks if the relative state differs, while current only checks if the outcome state differs
            # if (
            #     len(set([len(splitted[j][i]) for j in len(splitted)])) > 1
            # ):  # if not all the same length, then add to the CF dataset
            #     relevant_states += [relevant_states[j] + splitted[j][i] for j in range(len(splitted))]
            #     outcomes.append(out.values_at(i))
            # else:
            # s = splitted[0][i]
            s = counter_outcome_rollouts[0]
            # if sum([int(not s.state_equals(splitted[j][i])) for j in range(1, len(splitted))]):  # if there are any state dissimilarities
            if sum(
                [
                    int(not s.state_equals(counter_outcome_rollouts[j], at=i))
                    for j in range(1, option_node.num_params)
                ]
            ):  # if there are any state dissimilarities
                for j in range(option_node.num_params): 
                    relevant_states += [splitted[j][i]] # list of counterfactual rollout for option duration for each option
                    outcomes.append(**counter_outcome_rollouts[j].values_at(i))
            else: # keep the non-counterfactual data as well
                irrelevant_outcomes.append(**counter_outcome_rollouts[0].values_at(i)) # only need one outcome since they are all the same
        # print(relevant_states)
        relevant_states = merge_rollouts(relevant_states, set_dones=True)
        return relevant_states, irrelevant_outcomes, outcomes

def counterfactual_mask(names, num_params, outcome_rollouts):
    state_full = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
    name_counts = Counter()
    counterfactual_data = []
    counterfactual_masks = []
    EPSILON = 1e-10 # TODO: move this outside?
    counterfactual_component_probabilities = torch.zeros(state_full.shape)
    for i in [k*num_params for k in range(outcome_rollouts.filled // num_params)]: # TODO: assert evenly divisible
        s = state_full[i]
        state_equals = lambda x,y: (x-y).norm(p=1) < EPSILON
        components_unequal = lambda x,y: ((x-y).abs() > EPSILON).float()
        # print(s.cpu().numpy().tolist()[:15], state_full[i+1].cpu().numpy().tolist()[:15], state_full[i+2].cpu().numpy().tolist()[:15], state_full[i+3].cpu().numpy().tolist()[:15])
        if sum([int(not state_equals(s, state_full[i+j])) for j in range(1, num_params)]):
            counterfactual_data += [state_full[i+j] for j in range(num_params)]
            masks = [components_unequal(s, state_full[i+j]) for j in range(1, num_params)]
            combined_masks = sum(masks).clamp(0,1)
            counterfactual_masks += [combined_masks.clone() for j in range(num_params)] # the mask is repeated for each counterfactual
            counterfactual_component_probabilities += combined_masks
        # print(combined_masks.cpu().numpy().tolist()[:15], s.cpu().numpy().tolist()[:15], state_full[i+1].cpu().numpy().tolist()[:15])
        # print(i, outcome_rollouts.filled // num_options, len(counterfactual_masks), [state_full[i+j] for j in range(1, num_options)], s)
    for n in names:
        counterfactual_component_probabilities = counterfactual_component_probabilities / (outcome_rollouts.filled // num_params)
    return torch.stack(counterfactual_masks, dim = 0), counterfactual_component_probabilities