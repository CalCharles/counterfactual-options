import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Independent, Normal
import torch.optim as optim
import copy, os, cv2
from file_management import default_value_arg
from Networks.network import Network, pytorch_model
from Networks.tianshou_networks import networks
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
cActor, cCritic = Actor, Critic
from tianshou.utils.net.discrete import Actor, Critic
dActor, dCritic = Actor, Critic
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
import tianshou as ts
import gym
from typing import Any, Dict, Tuple, Union, Optional, Callable
from ReinforcementLearning.learning_algorithms import HER
from Rollouts.rollouts import ObjDict


_actor_critic = ['ddpg', 'sac']
_double_critic = ['sac']

# TODO: redo this one
class TSPolicy(nn.Module):
    '''
    wraps around a TianShao Base policy, but supports most of the same functions to interface with Option.option.py
    Note that TianShao Policies also support learning

    '''
    def __init__(self, input_shape, paction_space, action_space, max_action, discrete_actions, **kwargs):
        super().__init__()
        args = ObjDict(kwargs)
        self.algo_name = kwargs["learning_type"] # the algorithm being used
        self.is_her = self.algo_name[:3] == "her" # her is always a prefix
        self.collect = None
        self.lookahead = args.lookahead
        self.option = args.option
        if self.is_her: 
            self.algo_name = self.algo_name[3:]
            self.learning_algorithm = HER(ObjDict(kwargs), kwargs['option'])
            self.collect = self.learning_algorithm.record_state # TODO: not sure what to initialize with
            self.sample_buffer = self.learning_algorithm.sample_buffer
        self.action_space = action_space
        print(paction_space)
        kwargs["actor"], kwargs["actor_optim"], kwargs['critic'], kwargs['critic_optim'], kwargs['critic2'], kwargs['critic2_optim'] = self.init_networks(args, input_shape, paction_space.shape or paction_space.n, discrete_actions, max_action=max_action)
        kwargs["exploration_noise"] = GaussianNoise(sigma=args.epsilon)
        kwargs["action_space"] = action_space
        kwargs["discrete_actions"] = discrete_actions
        self.algo_policy = self.init_algorithm(**kwargs)
        self.parameterized = kwargs["parameterized"]
        self.param_process = None
        # self.map_action = self.algo_policy.map_action
        self.exploration_noise = self.algo_policy.exploration_noise
        self.grad_epoch = kwargs['grad_epoch']

    def init_networks(self, args, input_shape, action_shape, discrete_actions, max_action = 1):
        if discrete_actions:
            Actor, Critic = dActor, dCritic
        else:
            Actor, Critic = cActor, cCritic
        print(input_shape, action_shape, max_action, discrete_actions)
        actor, critic, critic2 = None, None, None
        actor_optim, critic_optim, critic2_optim = None, None, None
        PolicyType = networks[args.policy_type]
        device = 'cpu' if not args.cuda else 'cuda:' + str(args.gpu)
        if self.algo_name in _actor_critic:
            if discrete_actions: # handle both discrete and continuous
                cinp_shape = input_shape
                cout_shape = args.hidden_sizes[-1]
                aout_shape = args.hidden_sizes[-1]
                hidden_sizes = args.hidden_sizes[:-1]
            else:
                cinp_shape = int(input_shape + np.prod(action_shape))
                cout_shape = 1
                aout_shape = action_shape
                hidden_sizes = args.hidden_sizes

            actor = PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=aout_shape, hidden_sizes = hidden_sizes, preprocess=args.preprocess)
            critic = PolicyType(cuda=args.cuda, num_inputs=cinp_shape, num_outputs=cout_shape, hidden_sizes = hidden_sizes, preprocess=args.preprocess)
            if discrete_actions: critic = Critic(critic, last_size=action_shape, device=device).to(device)
            else: critic = Critic(critic, device=device).to(device)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
            if self.algo_name in _double_critic:
                if discrete_actions: actor = Actor(actor, action_shape, device=device).to(device)
                else: actor = ActorProb(actor, action_shape, device=device, max_action=max_action, unbounded=True, conditioned_sigma=True).to(device)
                critic2 = PolicyType(cuda=args.cuda, num_inputs=cinp_shape, num_outputs=cout_shape, hidden_sizes=hidden_sizes, preprocess=args.preprocess)
                if discrete_actions: critic2 = Critic(critic2, last_size=action_shape, device=device).to(device)
                else: critic2 = Critic(critic2, device=device).to(device)
                critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
            else:
                actor = Actor(actor, action_shape, device=device, max_action=max_action).to(device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

        elif self.algo_name in ['dqn']:
            critic = PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=action_shape, hidden_sizes = args.hidden_sizes, preprocess=args.preprocess)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        elif self.algo_name in ['ppo']:
            if discrete_actions:
                net = PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=args.hidden_sizes[-1], hidden_sizes = args.hidden_sizes[:-1], preprocess=args.preprocess)
                actor = Actor(net, action_shape, device=device).to(device)
                critic = Critic(net, device=device).to(device)
            else:
                net = PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=action_shape, hidden_sizes = args.hidden_sizes[:-1], preprocess=args.preprocess)
                actor = ActorProb(net, action_shape, max_action=max_action, device=device).to(device)
                critic = Critic(PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=1, preprocess=args.preprocess, hidden_sizes=args.hidden_sizes), device=device).to(device)
            actor_optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=args.actor_lr)
        return actor, actor_optim, critic, critic_optim, critic2, critic2_optim

    def set_eps(self, epsilon): # not all algo policies have set eps
        if hasattr(self.algo_policy, "set_eps"):
            self.algo_policy.set_eps(epsilon)

    def init_algorithm(self, **kwargs):
        args = ObjDict(kwargs)
        noise = GaussianNoise(sigma=args.epsilon) if args.epsilon > 0 else None
        if self.algo_name == "dqn": 
            policy = ts.policy.DQNPolicy(args.critic, args.critic_optim, discount_factor=args.discount_factor, estimation_step=args.lookahead, target_update_freq=int(args.tau))
            policy.set_eps(args.epsilon)
        elif self.algo_name == "ppo": 
            if args.discrete_actions:
                policy = ts.policy.PPOPolicy(args.actor, args.critic, args.actor_optim, torch.distributions.Categorical, discount_factor=args.discount_factor, max_grad_norm=None,
                                    eps_clip=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95, # parameters hardcoded to defaults
                                    reward_normalization=args.reward_normalization, dual_clip=None, value_clip=False,
                                    action_space=args.action_space)

            else:
                def dist(*logits):
                    return Independent(Normal(*logits), 1)
                policy = ts.policy.PPOPolicy(
                    args.actor, args.critic, args.actor_optim, dist, discount_factor=args.discount_factor, max_grad_norm=None, eps_clip=0.2, vf_coef=0.5, 
                    ent_coef=0.01, reward_normalization=args.reward_normalization, advantage_normalization=1, recompute_advantage=0, 
                    value_clip=False, gae_lambda=0.95, action_space=args.action_space)
        elif self.algo_name == "ddpg": 
            policy = ts.policy.DDPGPolicy(args.actor, args.actor_optim, args.critic, args.critic_optim,
                                                                            tau=args.tau, gamma=args.gamma,
                                                                            exploration_noise=GaussianNoise(sigma=args.epsilon),
                                                                            estimation_step=args.lookahead, action_space=args.action_space,
                                                                            action_bound_method='clip')
        elif self.algo_name == "sac":
            if args.discrete_actions:
                policy = ts.policy.DiscreteSACPolicy(
                        args.actor, args.actor_optim, args.critic, args.critic_optim, args.critic2, args.critic2_optim,
                        tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha, estimation_step=args.lookahead,
                        reward_normalization=args.reward_normalization, deterministic_eval=args.deterministic_eval)
            else:
                policy = ts.policy.SACPolicy(args.actor, args.actor_optim, args.critic, args.critic_optim, args.critic2, args.critic2_optim,
                                                                            tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha,
                                                                            exploration_noise=GaussianNoise(sigma=args.epsilon),
                                                                            estimation_step=args.lookahead, action_space=args.action_space,
                                                                            action_bound_method='clip', deterministic_eval=args.deterministic_eval)
        # support as many algos as possible, at least ddpg, dqn SAC
        return policy

    def save(self, pth, name):
        torch.save(self, os.path.join(pth, name + ".pt"))

    def add_param(self, batch, indices = None):
        orig_obs, orig_next = None, None
        if self.parameterized:
            orig_obs, orig_next = batch.obs, batch.obs_next
            if self.param_process is None:
                param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
            else:
                param_process = self.param_process
            if indices is None:
                batch['obs'] = param_process(batch['obs'], batch['param'])
                if type(batch['obs_next']) == np.ndarray: batch['obs_next'] = param_process(batch['obs_next'], batch['param']) # relies on batch defaulting to Batch, and np.ndarray for all other state representations
            else: # indices indicates that it is handling a buffer
                batch.obs[indices] = param_process(batch.obs[indices], batch.param[indices])
                if type(batch.obs_next[indices]) == np.ndarray: batch.obs_next[indices] = param_process(batch.obs_next[indices], batch.param[indices])                
                # print(batch.obs[indices].shape, batch.obs_next.shape)
        return orig_obs, orig_next

    def restore_obs(self, batch, orig_obs, orig_next):
        if self.parameterized:
            batch['obs'], batch['obs_next'] = orig_obs, orig_next

    def restore_buffer(self, buffer, orig_obs, orig_next, rew, done, idices):
        if self.parameterized:
            buffer.obs[idices], buffer.obs_next[idices], buffer.rew[idices], buffer.done[idices] = orig_obs, orig_next, rew, done

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """COPIED FROM BASE: Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.algo_policy.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)  # type: ignore
            elif self.algo_policy.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.algo_policy.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def reverse_map_action(self, mapped_act):
        # reverse the effect of map_action, not one to one because information might be lost (ignores clipping)
        if self.algo_policy.action_scaling:
            low, high = self.action_space.low, self.action_space.high
            act = ((mapped_act - low) / (high - low)) * 2 - 1
        if self.algo_policy.action_bound_method == "tanh":
            act = np.arctanh(act)
        return act


    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs", **kwargs: Any):
        '''
        Matches the call for the forward of another algorithm method. Calls 
        '''
            # not cloning batch could result in issues
        # print(batch.obs.shape, batch.obs_next.shape)
        vals= self.algo_policy(batch, state = state, input=input, **kwargs)
        return vals

    # def add_param_buffer(self, buffer, indices):
    #     '''
    #     edits obs, obs_next, reward and done based on the parameter
    #     '''
    #     if self.parameterized:
    #         new_indices = [indices]
    #         for _ in range(self.lookahead - 1):
    #             new_indices.append(buffer.next(new_indices[-1]))
    #         # new_indices.pop(0)
    #         new_indices_stack = np.stack(new_indices).flatten()
    #         # terminal indicates buffer indexes nstep after 'indice',
    #         # and are truncated at the end of each episode

    #         # Insert parameter into observations
    #         param = buffer.param[indices[0]]
    #         broadcast_object_state = np.stack([param.copy() for _ in range(new_indices_stack.shape[0])], axis=0)
    #         buffer.param[new_indices_stack] = broadcast_object_state
    #         param = broadcast_object_state
    #         input_state = buffer.obs_next[new_indices_stack]
    #         object_state = buffer.next_target[new_indices_stack]
    #         true_done = buffer.true_done[new_indices_stack]
    #         true_reward = buffer.true_reward[new_indices_stack]
    #         # apply termination and reward to buffer indices
    #         rew = buffer.rew[new_indices_stack]
    #         done = buffer.done[new_indices_stack]
    #         buffer.done[new_indices_stack] = self.option.termination.check(input_state, object_state, param, true_done)
    #         buffer.rew[new_indices_stack] = self.option.reward.get_reward(input_state, object_state, param, true_reward)

    #         orig_obs, orig_next = self.add_param(buffer, indices=new_indices_stack)
    #         return orig_obs, orig_next, rew, done, new_indices_stack
    #     return None, None, None, None, None # none of thes should be used if not parameterized

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        '''
        don't call the algo_policy update, but carries almost the same logic
        however, inserting the param needs to be handled.
        '''
        for i in range(self.grad_epoch):
            use_buffer = buffer
            if self.is_her:
                use_buffer = self.sample_buffer(buffer)
                # print(len(use_buffer))
            if use_buffer is None:
                return {}
            batch, indice = use_buffer.sample(sample_size)
            # print(batch)
            self.algo_policy.updating = True
            # print(use_buffer.param.shape, batch.param.shape, batch.obs_next.shape)
            # orig_obs, orig_next = self.add_param(batch)
            # orig_obs_buffer, orig_next_buffer, buffer_idces = self.add_param_buffer(use_buffer, indice)
            # for done, frame, last_frame,param in zip(batch.done, batch.obs_next, batch.obs, batch.param):
            #     target = np.argwhere(frame[:,:,2] == 10.0)[0]
            #     pos = np.argwhere(frame[:,:,1] == 10.0)[0]
            #     target_last = np.argwhere(frame[:,:,2] == 10.0)[0]
            #     pos_last = np.argwhere(last_frame[:,:,1] == 10.0)[0]
            #     p = np.argwhere(param == 10.0)[0]
            #     print(done, target, target_last, pos, pos_last, p)
            # print(type(batch["obs_next"]), batch["obs_next"].shape, self.param_process)
            # print(len(batch))
            batch = self.algo_policy.process_fn(batch, use_buffer, indice)
            # for o,on,r,d,a in zip(batch.obs, batch.obs_next, batch.rew, batch.done, batch.act):
            #     print(r,d,a)
            #     cv2.imshow('state', o)
            #     cv2.waitKey(500)
            #     cv2.imshow('state', on)
            #     cv2.waitKey(500)
            kwargs["batch_size"] = sample_size
            kwargs["repeat"] = 2
            result = self.algo_policy.learn(batch, **kwargs)
            self.algo_policy.post_process_fn(batch, use_buffer, indice)
            # self.restore_obs(batch, orig_obs, orig_next)
            # self.restore_buffer(orig_obs_buffer, orig_next_buffer, buffer_idces)
            self.algo_policy.updating = False
        return result


def dummy_RLoutput(n, num_actions, cuda):
    one = torch.tensor([n, 0])
    actions = torch.tensor([n, num_actions])
    if cuda:
        one = one.cuda()
        actions = actions.cuda()
    return RLoutput(one.clone(), )

class RLoutput():
    def __init__(self, values = None, dist_entropy = None, probs = None, log_probs = None, action_values = None, std = None, Q_vals = None, Q_best = None, dist = None):
        self.values = values 
        self.dist_entropy = dist_entropy 
        self.probs = probs
        self.log_probs = log_probs 
        self.action_values = action_values 
        self.std = std 
        self.Q_vals = Q_vals
        self.Q_best = Q_best
        self.dist = dist

    def values(self):
        return self.values, self.dist_entropy, self.probs, self.log_probs, self.action_values, self.std, self.Q_vals, self.Q_best, self.dist


# class pytorch_model():
#     def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
#         # should have customizable combiner and loss, but I dont.
#         self.cuda=cuda
#         self.reduce_size = 2 # someday this won't be hard coded either

#     @staticmethod
#     def wrap(data, dtype=torch.float, cuda=True):
#         # print(Variable(torch.Tensor(data).cuda()))
#         if type(data) == torch.Tensor:
#             v = data.clone().detach()
#         else:
#             v = torch.tensor(data, dtype=dtype)
#         if cuda:
#             return v.cuda()
#         return v


#     @staticmethod
#     def unwrap(data):
#         return data.clone().detach().cpu().numpy()

#     @staticmethod
#     def concat(data, axis=0):
#         return torch.cat(data, dim=axis)

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class QFunction(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs["hidden_size"]
        self.double_layer = kwargs["double_layer"] # remove double layer support unless important
        if self.double_layer:
            self.l1 = nn.Linear(self.num_inputs + self.hidden_size, self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, 1)
        else:
            self.l1 = nn.Linear(self.num_inputs + self.hidden_size, 1)

    def scale_last(self):
        self.l1.weight.data.mul_(0.1)
        self.l1.bias.data.mul_(0.1)
        if self.double_layer:
            self.l2.weight.data.mul_(0.1)
            self.l2.bias.data.mul_(0.1)

    def forward(self, hidden, action):
        x = torch.cat([hidden, action], dim=1)
        x = self.l1(x)
        if self.double_layer:
            x = F.relu(x)
            x = self.l2(x)
        return x

class Policy(nn.Module):
    def __init__(self, **kwargs):
        super(Policy, self).__init__()
        self.num_inputs = default_value_arg(kwargs, 'num_inputs', 10)
        print("num_inputs", self.num_inputs)
        self.no_preamble = default_value_arg(kwargs, 'no_preamble', False)
        self.param_size = default_value_arg(kwargs, 'param_size', 10)
        self.num_outputs = default_value_arg(kwargs, 'num_outputs', 1)
        self.factor = default_value_arg(kwargs, 'factor', None)
        # self.minmax = default_value_arg(kwargs, 'minmax', None)
        # self.use_normalize = self.minmax is not None
        self.name = default_value_arg(kwargs, 'name', 'option')
        self.iscuda = default_value_arg(kwargs, 'cuda', True) # TODO: don't just set this to true
        self.mean = pytorch_model.wrap(np.zeros(self.num_inputs), cuda=self.iscuda)
        self.init_form = default_value_arg(kwargs, 'init_form', 'xnorm') 
        self.scale = default_value_arg(kwargs, 'scale', 1) 
        self.activation = default_value_arg(kwargs, 'activation', 'relu') 
        self.test = not default_value_arg(kwargs, 'train', True) # test is also true if the policy is not trainable
        self.Q_critic = default_value_arg(kwargs, 'Q_critic', False) 
        self.continuous = default_value_arg(kwargs, 'continuous', False)
        model_form = default_value_arg(kwargs, 'train', 'basic') 
        self.has_final = default_value_arg(kwargs, 'needs_final', True)
        self.concatenate_param = default_value_arg(kwargs, 'concatenate_param', True)
        self.last_param = default_value_arg(kwargs, 'last_param', False)
        self.normalize = default_value_arg(kwargs, 'normalize', False)
        self.normalized_actions = default_value_arg(kwargs, 'normalized_actions', False)
        self.option_values = torch.zeros(1, self.param_size) # changed externally to the parameters
        self.num_layers = default_value_arg(kwargs, 'num_layers', 1)

        # self.double_layer = kwargs['double_layer']

        if self.num_layers == 0:
            self.insize = self.num_inputs
            if self.concatenate_param:
                self.insize = self.num_inputs + self.param_size
            # self.insize = self.num_inputs
        else:
            self.insize = self.factor * self.factor * self.factor // min(self.factor, 4)
        
        # self.insize = 564


        if self.last_param:
            self.insize += self.param_size
        self.layers = []
        if self.has_final:
            self.init_last(self.num_outputs)
        if self.activation == "relu":
            self.acti = F.relu
        elif self.activation == "sin":
            self.acti = torch.sin
        elif self.activation == "sigmoid":
            self.acti = torch.sigmoid
        elif self.activation == "tanh":
            self.acti = torch.tanh
        self.parameter_count = -1
        print("activation", self.acti)
        print("current insize", self.insize)
            
    def init_last(self, num_outputs):
        self.critic_linear = nn.Linear(self.insize, 1)
        self.sigma = nn.Linear(self.insize, 1)
        if self.continuous:
            self.QFunction = QFunction(hidden_size=self.insize, num_outputs=1, num_inputs=num_outputs, activation=self.activation, init_form=self.init_form, double_layer=False)
        else:
            self.QFunction = nn.Linear(self.insize, num_outputs)
        self.action_eval = nn.Linear(self.insize, num_outputs)
        if len(self.layers) > 0:
            self.layers = self.layers[5:]
        self.layers = [self.critic_linear, self.sigma, self.QFunction, self.action_eval] + self.layers

    def scale_last(self):
        if type(self.QFunction) == nn.Linear:
            self.QFunction.weight.data.mul_(0.1)
            self.QFunction.bias.data.mul_(0.1)
        else:
            self.QFunction.scale_last()
        self.action_eval.weight.data.mul_(0.1)
        self.action_eval.bias.data.mul_(0.1)
        self.sigma.weight.data.mul_(0.4)
        self.sigma.bias.data.mul_(0.4)


    def set_mean(self, rollouts):
        s = rollouts.get_values("state")
        self.mean = pytorch_model.wrap(s.mean(dim=0), cuda=self.iscuda)

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            if issubclass(type(layer), Policy):
                layer.reset_parameters()
            elif type(layer) == nn.Conv2d and self.init_form != 'none':
                print("layer", layer, self.init_form)
                if self.init_form == "orth":
                    # print(layer.weight.shape, layer.weight)
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif type(layer) == nn.Parameter and self.init_form != 'none':
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            else:
                fulllayer = layer
                if self.init_form == "none":
                    print("no initialization", layer)
                    continue
                print("did not continue")
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    print("layer", layer, self.init_form)
                    # print("layer", self, layer)
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                        # print(layer.weight[10:,10:])
                    if self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 1.5 / layer.weight.data.shape[0])
                    if self.init_form == "smalluni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "knorm":
                        torch.nn.init.kaiming_normal_(layer.weight.data)
                    elif self.init_form == "kuni":
                        torch.nn.init.kaiming_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if layer.bias is not None:                
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
        # if self.has_final:
        #     nn.init.orthogonal_(self.action_probs.weight.data, gain=0.01)
        print("parameter number", self.count_parameters(reuse=False))

    def last_layer(self, x, param, xcritic=None, set_actions=None):
        '''
        input: [batch size, insize]
        output [batch size, 1], [batch size, 1], [batch_size, num_actions], [batch_size, num_actions], [batch_size, num_actions]
        '''
        if self.last_param:
            x = torch.cat((x,param), dim=1)
        if xcritic is None:
            xcritic = x
        dist = None
        std = self.sigma(x)
        if set_actions is None:
            action_values = self.action_eval(x)
            if self.normalized_actions:
                std = torch.tanh(std) # might not want to normalize here
                action_values = torch.tanh(action_values) # might not want to normalize outputs
        else:
            action_values = set_actions
        if self.continuous:
            Q_vals = self.QFunction(xcritic, action_values)
            dist = FixedNormal(action_values, std)
            # print(action_values, dist, std)
            log_probs = dist.log_probs(action_values)
            dist_entropy = dist.entropy().mean()
            probs = torch.exp(log_probs)
            Q_best = Q_vals # Doesn't actually optimize the Q space for best action
            values = Q_vals # again, doesn't optimize
        else:
            Q_vals = self.QFunction(xcritic)
            probs = F.softmax(action_values, dim=1) 
            log_probs = F.log_softmax(action_values, dim=1)
            dist_entropy = action_values - action_values.logsumexp(dim=-1, keepdim=True)
            Q_best = Q_vals.max(dim=1)[1]

            # print("act", action_values,"prob", probs, "logp", log_probs,"de", dist_entropy)
            if self.Q_critic:
                values = Q_vals.max(dim=1)[0]
            else:
                values = self.critic_linear(x)
        # return values, None, None, None, None, None, Q_vals, None, None
        return values, dist_entropy, probs, log_probs, action_values, std, Q_vals, Q_best, dist

    def preamble(self, x, p):
        if not self.last_param:
            if self.concatenate_param:
                x = torch.cat((x,p), dim=1)
            # y = x - p
            # x = torch.cat((x,y), dim=1)
        if self.normalize:
            # x = x - self.mean
            return (x / 84 - .5) * self.scale # normalizes the parameter for better or worse
        return x

    def compute_Q(self, state, param, action):
        x = self.preamble(state, param) # TODO: if necessary, a preamble can be added back in
        x = self.hidden(x, param)
        return self.QFunction(x, action)

    def hidden(self, x, p):
        pass # defined in subclass classes

    def forward(self, x, p):
        if not self.no_preamble:
            x = self.preamble(x, p)
        x = self.hidden(x, p)
        values, dist_entropy, probs, log_probs, action_values, std, Q_vals, Q_best, dist = self.last_layer(x, p)
        # print(action_values)
        return RLoutput(values, dist_entropy, probs, log_probs, action_values, std, Q_vals, Q_best, dist)

    def save(self, pth, name):
        torch.save(self, os.path.join(pth, name + ".pt"))

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def set_parameters(self, param_val): # sets the parameters of a model to the parameter values given as a single long vector
        if len(param_val) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_val = param_val[pval_idx : pval_idx+param_size]
            if type(cur_param_val) == torch.Tensor:
                param.data = cur_param_val.reshape(param.size()).float().clone()
            else:
                param.data = torch.from_numpy(cur_param_val) \
                              .reshape(param.size()).float()
            pval_idx += param_size
        if self.iscuda:
            self.cuda()

    # count number of parameters
    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to the model, for exploration"""
        params = self.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    # TODO: write code to remove last layer if unnecessary
    def remove_last(self):
        self.critic_linear = None
        self.QFunction = None
        self.action_probs = None
        self.layers = self.layers[3:]


class BasicPolicy(Policy):    
    def __init__(self, **kwargs):
        super(BasicPolicy, self).__init__(**kwargs)
        self.hidden_size = self.factor*self.factor*self.factor // min(4,self.factor)
        self.use_layer_norm = default_value_arg(kwargs, "use_layer_norm", False)
        print("Network Sizes: ", self.num_inputs, self.insize, self.hidden_size)
        # self.l1 = nn.Linear(self.num_inputs, self.num_inputs*factor*factor)
        
        # remove this line
        # self.double_layer = kwargs["double_layer"]
        # if self.double_layer:
        #     self.num_layers = self.num_layers - 1

        if not self.last_param and self.concatenate_param:
            self.num_inputs += self.param_size
        print("num_inputs", self.num_inputs, self.concatenate_param)        
        # self.num_inputs = self.param_size # Turn this on to only input the parameter (also line in hidden)
        print(self.last_param, self.num_inputs)
        if self.num_layers == 1:
            self.l1 = nn.Linear(self.num_inputs,self.insize)
            if self.use_layer_norm:
                self.ln1 = nn.LayerNorm(self.insize)
        elif self.num_layers == 2:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size, self.insize)
            if self.use_layer_norm:
                self.ln1 = nn.LayerNorm(self.hidden_size)
                self.ln2 = nn.LayerNorm(self.insize)
        elif self.num_layers == 3:
            self.l1 = nn.Linear(self.num_inputs,self.hidden_size)
            self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
            self.l3 = nn.Linear(self.hidden_size, self.insize)
            if self.use_layer_norm:
                self.ln1 = nn.LayerNorm(self.hidden_size)
                self.ln2 = nn.LayerNorm(self.hidden_size)
                self.ln3 = nn.LayerNorm(self.insize)
        if self.num_layers > 0:
            self.layers.append(self.l1)
            if self.use_layer_norm:
                self.layers.append(self.ln1)
        if self.num_layers > 1:
            self.layers.append(self.l2)
            if self.use_layer_norm:
                self.layers.append(self.ln2)
        if self.num_layers > 2:
            self.layers.append(self.l3)
            if self.use_layer_norm:
                self.layers.append(self.ln3)
        self.train()
        self.reset_parameters()
        self.scale_last()

    def hidden(self, x, p):
        # print(x.shape, p.shape, x, p)
        # x = p # turn this on to only input the parameter
        if self.num_layers > 0:
            x = self.l1(x)
            if self.use_layer_norm:
                x = self.ln1(x)
        if self.num_layers > 1:
            x = self.acti(x)
            x = self.l2(x)
            if self.use_layer_norm:
                x = self.ln2(x)
        if self.num_layers > 2:
            x = self.acti(x)
            x = self.l3(x)
            if self.use_layer_norm:
                x = self.ln3(x)
        x = self.acti(x)
        return x

    def compute_layers(self, x):
        layer_outputs = []
        if self.minmax is not None and self.use_normalize:
            x = self.normalize(x)
        if self.num_layers > 0:
            x = self.l1(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())
        if self.num_layers > 1:
            x = self.l2(x)
            x = F.relu(x)
            layer_outputs.append(x.clone())

        return layer_outputs

class BasicActorCriticPolicy(Policy):
    def __init__(self, **kwargs):
        # kwargs["double_layer"] = False
        kwargs["needs_final"] = False
        super(BasicActorCriticPolicy, self).__init__(**kwargs)
        kwargs["needs_final"] = True
        self.actor = BasicPolicy(**kwargs)
        # kwargs["double_layer"] = True
        self.critic = BasicPolicy(**kwargs)
        self.hidden = self.critic.hidden
        self.QFunction = self.critic.QFunction
        # self.compute_Q = self.critic.compute_Q
        self.actor.train()
        self.critic.train()
        # self.set_mean = self.actor.set_mean
        # self.train()
        # self.reset_parameters()

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()

    def forward(self, x, p): # almost the same as in Policy
        if not self.actor.no_preamble:
            x = self.actor.preamble(x, p)
        xpolicy = self.actor(x, p)
        xcritic = self.critic.hidden(x, p)
        xcritic = RLoutput(*self.critic.last_layer(xcritic,p,set_actions=xpolicy.action_values))
        values, dist_entropy, probs, log_probs, action_values, std, Q_vals, Q_best, dist = self.last_layer(xpolicy, p, xcritic)
        # print(action_values)
        return RLoutput(values, dist_entropy, probs, log_probs, action_values, std, Q_vals, Q_best, dist)

    def last_layer(self, pout, param, cout):
        return cout.values, pout.dist_entropy, pout.probs, pout.log_probs, pout.action_values, pout.std, cout.Q_vals, cout.Q_best, pout.dist

# TODO: make a network with completely separate channels

class ImagePolicy(Policy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: assumes images of size 84x84, make general
        self.no_preamble = True
        self.num_stack = 4
        factor = self.factor
        self.conv1 = nn.Conv2d(self.num_stack, 2 * factor, 8, stride=4)
        self.conv2 = nn.Conv2d(2 * factor, 4 * factor, 4, stride=2)
        self.conv3 = nn.Conv2d(4 * factor, 2 * factor, 3, stride=1)
        self.viewsize = 7
        self.reshape = kwargs["reshape"]
        # if self.args.post_transform_form == 'none':
        #     self.linear1 = None
        #     self.insize = 2 * self.factor * self.viewsize * self.viewsize
        #     self.init_last(self.num_outputs)
        # else:
        self.linear1 = nn.Linear(2 * factor * self.viewsize * self.viewsize, self.insize)
        self.layers.append(self.linear1)
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        self.layers.append(self.conv3)
        self.reset_parameters()

    def hidden(self, inputs, p):
        if self.reshape[0] != -1:
            inputs = inputs.reshape(-1, *self.reshape)
        norm_term = 1.0
        if self.normalize:
            norm_term =  255.0
        x = self.conv1(inputs / norm_term)
        x = self.acti(x)

        x = self.conv2(x)
        x = self.acti(x)

        x = self.conv3(x)
        x = self.acti(x)
        x = x.view(-1, 2 * self.factor * self.viewsize * self.viewsize)
        x = self.acti(x)
        if self.linear1 is not None:
            x = self.linear1(x)
            x = self.acti(x)
        return x

class GridWorldPolicy(Policy):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        N = 20 # hardcoded at the moment
        H, W = N, N
        self.H, self.W = H, W
        self.C = 3
        self.Chid = 32
        self.Chid2 = 64
        self.Chid3 = 64
        self.no_preamble = True
        self.reshape = kwargs["reshape"]
        self.mean = pytorch_model.wrap(np.zeros(self.reshape), cuda=self.iscuda)

        self.conv1 = torch.nn.Conv2d(in_channels=self.C,out_channels=self.Chid,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.Chid,out_channels=self.Chid2,kernel_size=3,stride=1,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.Chid2,out_channels=self.Chid3,kernel_size=3,stride=1,padding=1)
        self.fc1 = torch.nn.Linear(int(self.Chid3*H*W/16),self.insize)
        # self.fc2 = torch.nn.Linear(564,self.insize)
        self.train()
        self.reset_parameters()
        
    def hidden(self,x, p):
        # may need some reassignment logic
        x = x.reshape(-1,*self.reshape)
        np.set_printoptions(threshold=np.inf, precision=4)
        # print("input")
        # print(pytorch_model.unwrap(x))
        x[:,:,:,2] = p.reshape(-1,*self.reshape[:-1])
        # print("assign parameter")
        # print(pytorch_model.unwrap(x))

        # x = x - self.mean.reshape(*self.reshape)
        
        # state = x.detach().cpu().numpy()[0]
        # print(np.argwhere(pytorch_model.unwrap(self.mean.reshape(*self.reshape)) != 0))
        # cv2.imshow("state", pytorch_model.unwrap(x[0].reshape(*self.reshape)) * 255)
        # cv2.waitKey(200)
        
        x = x.transpose(3,2).transpose(2,1)
        
        # print("transposed")
        # print(pytorch_model.unwrap(x))
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        # print("layer 1")
        # print(pytorch_model.unwrap(x))
        x = F.relu(self.conv2(x))
        # print("layer 2")
        # print(pytorch_model.unwrap(x))
        x = F.max_pool2d(F.relu(self.conv3(x)),2)
        # print("layer 3")
        # print(pytorch_model.unwrap(x))
        x = x.reshape(batch_size,(self.Chid3*self.H*self.W)//16)
        x = F.relu(self.fc1(x))
        # print("last linear")
        # print(pytorch_model.unwrap(x))
        return x



policy_forms = {"basic": BasicPolicy, "image": ImagePolicy, 'grid': GridWorldPolicy, 'actorcritic': BasicActorCriticPolicy}
