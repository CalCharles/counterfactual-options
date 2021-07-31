import numpy as np
from tianshou.data import Collector, Batch, ReplayBuffer
from Networks.network import pytorch_model
from ReinforcementLearning.learning_algorithms import HER


class TestExtractor():
    def __init__(self, target="Paddle"):
        self.target = target

    def get_true_done(self, full_state):
        return full_state['factored_state']['Done']

    def get_obs(self, full_state, param, mask=None):
        if type(full_state) == Batch or type(full_state) == dict:
            shape = np.array(full_state['factored_state']["Action"]).shape
            state = [full_state['factored_state']["Action"], full_state['factored_state']["Paddle"], full_state['factored_state']["Ball"], param]
            state = [np.array(s).squeeze() for s in state]
            if len(shape) > 1 and shape[0] == 1:
                state = [np.expand_dims(np.array(s), axis=0) for s in state]
            return np.concatenate(state, axis=len(shape) - 1)
        else:
            shape = full_state.shape
            if len(shape) == 1: return full_state[:15]
            if len(shape) == 2: return full_state[:,:15]
            if len(shape) == 3: return full_state[:,:,:15]


    def get_inter(self, full_state):
        if type(full_state) == Batch or type(full_state) == dict:
            shape = full_state['factored_state']["Action"].shape
            state = [full_state['factored_state']["Paddle"], full_state['factored_state']["Ball"]]
            return np.concatenate(state, axis=len(shape) - 1)
        else:
            shape = full_state.shape
            if len(shape) == 1: return np.concatenate([full_state[:15], param], axis=0)
            if len(shape) == 2: return np.concatenate([full_state[:,:15], param], axis=1)
            if len(shape) == 3: return np.concatenate([full_state[:,:,:15], param], axis=2)

    def get_target(self, full_state):
        if type(full_state) == Batch or type(full_state) == dict:
            return full_state['factored_state'][self.target]
        else:
            shape = full_state.shape
            if len(shape) == 1: return full_state[5:10]
            if len(shape) == 2: return full_state[:,5:10]
            if len(shape) == 3: return full_state[:,:,5:10]


    def assign_param(self, full_state, obs, param, mask):
        shape = obs.shape
        if len(shape) == 1: obs[15:20] = param.squeeze() * mask.squeeze()
        else: obs[:, 15:20] = np.stack([param.squeeze() * mask.squeeze() for i in range(shape[0])], axis=0)
        return obs


class TestDatasetModel():
    def __init__(self, multi_instanced = False):
        self.interaction_prediction = .3
        self.interaction_minimum = .9
        self.multi_instanced = multi_instanced

    def hypothesize(self, state): # gives an "interaction" at some random locations
        if type(state) == np.ndarray: v = state[6]
        else:
            istate = state['factored_state']['Ball'].squeeze()
            shape = istate.shape
            if len(shape) == 1: v = istate[1]
            else: v = istate[:,1]
        if int(v + .5) % 3 == 0:
            return np.array(1), np.array(0), np.array(0)
        return np.array(0), np.array(0), np.array(0)

class TestPolicy():
    def init_HER(self, args, option):

        self.her = HER(args, option)

    def collect(self, aggregated_batch, single_data, skipped, added):
        self.her.record_state(aggregated_batch, single_data, skipped, added)
        return 0

    def exploration_noise(self, act, batch):
        return act

class TestOption():
    def __init__(self, models):
        self.sampler = models.sampler
        self.state_extractor = models.state_extractor
        self.terminate_reward = models.terminate_reward
        self.dataset_model = models.dataset_model
        self.temporal_extension_manager = models.temporal_extension_manager
        self.done_model = models.done_model
        self.policy = models.policy
        self.driving = False

    def reset(self, full_state):
        return [False,False]

    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False):
        if np.random.randint(2) == 0 or self.driving:
            p = batch.param.squeeze()
            t = batch.next_target.squeeze() if 'next_target' in batch else batch.target.squeeze()
            target = (t - p)[1]
            if abs(target) < 1:
                act = 0
            elif target > 0:
                act = 2
            else:
                act = 3
        else:
            act = np.random.randint(4)
        chain = [act]
        policy_batch = dict()
        state = None
        masks = [[0,1,0,0,0]]
        resampled = True
        return act, chain, policy_batch, state, masks, resampled
    
    def terminate_reward_chain(self, full_state, next_full_state, param, chain, mask, mask_chain):
        array_factored(full_state['factored_state'])
        array_factored(next_full_state['factored_state'])
        termination, reward, inter, time_cutoff = self.terminate_reward.check(full_state, next_full_state, param, mask)
        ext_term = self.temporal_extension_manager.get_extension(termination, np.random.randint(2))
        done = self.done_model.check(termination, np.array(next_full_state['factored_state']['Done']).squeeze())

        rewards, terminations, ext_term = list() + [reward], list() + [termination], list() + [ext_term]
        
        self.driving = True
        if termination: self.driving = False
        return done, rewards, terminations, ext_term, inter, time_cutoff

    def update(self, buffer, done, last_state, act, chain, term_chain, param, masks, update_policy=True):
        return

class TestOptionBall(TestOption):
    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False):
        if np.random.randint(2) == 0 or (self.driving and np.random.randint(10) > 2):
            p = batch.obs.squeeze()[5:10]
            t = batch.next_target.squeeze() if 'next_target' in batch else batch.target.squeeze()
            target = (p - t)[1]
            if abs(target) < 1:
                act = 0
            elif target > 0:
                act = 2
            else:
                act = 3
        else:
            act = np.random.randint(4)
        chain = [act]
        policy_batch = dict()
        state = None
        masks = [[0,0,1,1,0]]
        resampled = True
        return act, chain, policy_batch, state, masks, resampled

def array_factored(factored_state):
    for k,v in factored_state.items():
        factored_state[k] = np.array(v)

class TestSampler():
    def get_param(self, full_state, terminate):
        if terminate:
            states = full_state['factored_state']
            shape = np.array(states['Action']).shape
            self.mask = np.array([0,1,0,0,0])
            self.param = np.array([0, np.random.rand() * (71.5-11.5) + 11.5, 0, 0, 0])
        return self.param, self.mask, terminate

    def get_mask_param(self, param, mask):
        return param * mask

class TestSamplerBall():
    def get_param(self, full_state, terminate):
        if terminate:
            states = full_state['factored_state']
            shape = np.array(states['Action']).shape
            self.mask = np.array([0,0,1,1,0])
            self.param = np.array([0, 0, np.random.choice([-2, -1]), np.random.choice([-1,1]), 0])
        return self.param, self.mask, terminate

    def get_mask_param(self, param, mask):
        return param * mask

class TestSamplerBlock():
    def get_param(self, full_state, terminate):
        if terminate:
            states = full_state['factored_state']
            shape = np.array(states['Action']).shape
            self.mask = np.array([0,0,0,0,1])
            chosen = np.random.choice([exp for exp in environment_model.environment.exposed_blocks.values()])
            self.param = np.array([exp[0], exp[1], 0, 0, 0])
        return self.param, self.mask, terminate

    def get_mask_param(self, param, mask):
        return param


class TestTemporalExtensionManager():
    def reset(self):
        return

    def get_extension(self, terminate, ext_term):
        return terminate + ((np.random.randint(2) == 1) * (np.random.randint(2) == 1)) # .25 probability of temporal extension

class TestTemporalExtensionManagerBall():
    def reset(self):
        return

    def get_extension(self, terminate, ext_term):
        return terminate + ((np.random.randint(6) == 0))

class TestTemporalExtensionManagerBall():
    def reset(self):
        return

    def get_extension(self, terminate, ext_term):
        return terminate + ext_term


class TestDoneModel():
    def done_check(term, true_done):
        return term + true_done

class TestTerminateReward():
    def check(self, full_state, next_full_state, param, mask, inter_state = None, use_timer=True):
        return 1, 0, 0

class TestTerminator():
    def check(self, inter, state, param, mask, true_done=0):
        return np.random.randint(2)


class TestRewarder():
    def get_reward():
        return np.random.rand()