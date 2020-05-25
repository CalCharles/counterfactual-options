import numpy as np
import os, cv2, time
from EnvironmentModel.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts

class CounterfactualDataset:
    def __init__(self, environment_model):
        self.model = environment_model
        self.known_objects = ["Action"]

    def generate_dataset(self, rollouts, option_level):
    	'''
		generates a single step counterfactual rollout, meaning that if the actions from the option level (possibly temporally extended)
		have an effect on the single step outcomes of the state, then the states are added.
		rollouts contains a set of rollouts including the factored states
		option_level includes a level of trained options to check for counterfactual relationships
		returns three components:
            relevant_states: A list of datasets all states traversed by the option, and the state the option was initiated in, for each option (on dataset per option)
            irrelevant_outcomes: a list of datasets of all outcomes where 
    	'''
    	initialize_rollout = lambda :[
            ModelRollouts(
                length=rollouts.length, shapes_dict={
                "state":rollouts.shapes["state"],
                "raw":rollouts.shapes["raw"],
                "state_diff":rollouts.shapes["state_diff"]
                "action":option_level.action_shape
                }
            )
            for i in range(option_level.num_options)
        ]
        counter_rollouts = initialize_rollout()
        counter_outcome_rollouts = initialize_rollout()
        # collecting rollouts
        for state in rollouts.values["state"]:
            for i, option in enumerate(option_level.options):
                self.model.set_from_factored_state(
                    self.model.unflatten_state(state)
                )  # TODO: setting criteria might change (for more limited models)
                done = False
                last_state = self.model.flatten_factored_state(self.model.get_factored_state())
                while not done:
                    state = self.model.flatten_factored_state(self.model.get_factored_state())
                    action_chain, done = option.get_action(state)
                    counter_rollouts[i].append(
                        {"state": state, "state_diff": state-last_state, "action": action_chain[-1], "done": done}
                    )
                    # TODO: handle case where not done, but significant change occurs, and put in counter_outcome_rollouts
                    last_state = state
                    self.model.step()
                state = self.model.get_factored_state() # add the last state
                action_chain, done = option.get_action(state)
                counter_outcome_rollouts[i].append(
                    {"state": state, "state_diff": state-last_state, "action": action_chain[-1], "done": done}
                )
        splitted = []
        for cr in counter_rollouts:
            splitted.append(cr.split_trajectories())
        relevant_states = [[] for i in range(len(splitted))]
        irrelevant_outcomes = initialize_rollout() # only use outcomes, or use all states?
        outcomes = initialize_rollout()
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
                    int(not s.state_equals(splitted[j]), at=i)
                    for j in range(1, option_level.num_options)
                ]
            ):  # if there are any state dissimilarities
                for j in range(option_level.num_options): 
                    relevant_states[j] += splitted[j][i] # list of counterfactual rollout for option duration for each option
                    outcomes[j].append(**counter_outcome_rollouts[j].values_at(i))
            else: # keep the non-counterfactual data as well
            	irrelevant_outcomes.append(out.values_at(i))
        relevant_states = [merge_rollouts(relevant_states[j], set_dones=True) for j in range(len(splitted))]
        return relevant_states, irrelevant_outcomes, outcomes

def counterfactual_mask(outcome_rollouts):
    state_full = torch.cat((outcome_rollouts.get_values("state"), outcome_rollouts.get_values("state_diff")), dim = 1)
    name_counts = Counter()
    counterfactual_data = []
    counterfactual_masks = []
    EPSILON = 1e-10 # TODO: move this outside?
    counterfactual_component_probabilities = torch.zeros(state_full.shape)
    for i in [k*self.num_options for k in range(outcome_rollouts.length // self.num_options)]: # TODO: assert evenly divisible
        s = state_full[i]
        state_equals = lambda x,y: (x-y).norm(p=1) < EPSILON
        components_unequal = lambda x,y: ((x-y).abs() > EPSILON).float()
        if sum([int(not state_equals(s, state_full[i+j])) for j in range(1, self.num_options)]):
            counterfactual_data += [state_full[i+j] for j in range(self.num_options)]
            masks = [components_unequal(s, state_full[i+j]) for j in range(1, self.num_options)]
            counterfactual_masks += masks
            counterfactual_component_probabilities += sum(masks).clamp(0,1)
    for n in self.names:
        counterfactual_component_probabilities = counterfactual_component_masks / (outcome_rollouts.length // self.num_options)
    return counterfactual_masks, counterfactual_component_probabilities