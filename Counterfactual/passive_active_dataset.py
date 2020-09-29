import numpy as np
import os, cv2, time
import torch
from EnvironmentModels.environment_model import ModelRollouts
from Rollouts.rollouts import merge_rollouts


class PassiveActiveDataset:
    def __init__(self, environment_model):
        self.model = environment_model
        self.known_objects = ["Action"]

    def train(self, rollouts, option_node):
        '''
        If some form of classification, some form of the passive model, the active contingent model, or some other form should be trained,
        then this function will train and retain those values
        '''
        return

    def generate_dataset(self, rollouts, option_node, contingent_nodes):
        '''
        takes in a dataset of rollouts, and determines which states are part of the passive set and which are part of the active contingent set
        defined for the option node as the target. This function will not call train, but will use the existing trained values.
        It returns a -1-0-1 vector, where -1 is active not active contingent, 0 is passive, 1 is active, as well as the split rollouts
        '''

        return None, None, None
        # return relevant_states, irrelevant_outcomes, outcomes

class HackedPassiveActiveDataset:
    def generate_dataset(self, rollouts, option_node, contingent_nodes):
        initialize_rollout = lambda length: ModelRollouts(
                length=length, shapes_dict={
                "state":rollouts.shapes["state"],
                "state_diff":rollouts.shapes["state_diff"],
                "action":option_node.action_shape,
                "done":rollouts.shapes["done"]
                }
            )
        identifiers = list()
        contingent_names = [n.name for n in contingent_nodes]
        passive = initialize_rollout(rollouts.length)
        irrelevant = initialize_rollout(rollouts.length)
        contingent_active = initialize_rollout(rollouts.length)
        for odiff, ostate, nstate, oaction in zip(rollouts.get_values("state_diff"), rollouts.get_values("state"), rollouts.get_values("next_state"), rollouts.get_values("action")):
            factored_ostate = self.model.unflatten_state(ostate)
            self.model.set_from_factored_state(
                self.model.unflatten_state(ostate), seed_counter=seed
            )  # TODO: setting criteria might change (for more limited models)
            self.model.step(oaction)
            interactions = self.model.get_interaction_trace(option_node)
            all_interactions = np.sum(interactions, axis=1)
            if len(all_interactions) == 0:
                identifiers.append(0)
                passive.append(
                    **{"state": ostate, "next_state": nstate, "state_diff": odiff, "action": oaction, "done": done}
                )
            else:
                interacted = False
                for n in contingent_names:
                    if n in all_interactions:
                        identifiers.append(1)
                        contingent_active.append(
                            **{"state": ostate, "next_state": nstate, "state_diff": odiff, "action": oaction, "done": done}
                        )
                        interacted = True
                        break
                if not interacted:
                    identifiers.append(-1)
                    irrelevant.append(
                        **{"state": ostate, "next_state": nstate, "state_diff": odiff, "action": oaction, "done": done}
                    )
        return np.array(identifiers), passive, contingent_active, irrelevant