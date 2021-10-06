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
        self.lookahead = args.lookahead # lookahead for RL, computes \sum^lookahead R + Q(s^lookahead+1)
        # self.option = args.option
        self.use_input_norm = args.input_norm
        self.input_var = np.ones(input_shape) # only if input variance and mean are used
        self.input_mean = np.zeros(input_shape)
        self.sample_merged = self.is_her and args.sample_merged # always sample merged if using HER right now
        self.learning_algorithm = None
        self.collect = None
        self.sample_buffer = None
        if self.is_her: 
            self.algo_name = self.algo_name[3:]
            self.learning_algorithm = HER(ObjDict(kwargs), kwargs['option'])
            self.collect = self.learning_algorithm.record_state # a function to record a new state to HER buffer
            self.sample_buffer = self.learning_algorithm.sample_buffer # a function to choose between buffers
        self.action_space = action_space # policy action space
        self.epsilon_schedule = args.epsilon_schedule # if > 0, adjusts epsilon from 1->args.epsilon by exp(-steps/epsilon schedule)
        self.epsilon_timer = 0 # timer to record steps
        self.epsilon = 1 if self.epsilon_schedule > 0 else args.epsilon
        self.epsilon_base = args.epsilon
        self.pretrain_iters = args.pretrain_iters # don't count the pretrain iterations (random actions)

        args.object_dim, args.first_obj_dim = kwargs["object_dim"], kwargs['first_obj_dim']
        kwargs["actor"], kwargs["actor_optim"], kwargs['critic'], kwargs['critic_optim'], kwargs['critic2'], kwargs['critic2_optim'] = self.init_networks(args, input_shape, paction_space.shape or paction_space.n, discrete_actions, max_action=max_action)
        kwargs["exploration_noise"] = GaussianNoise(sigma=1 if self.epsilon_schedule > 0 else args.epsilon)
        kwargs["action_space"] = action_space
        kwargs["discrete_actions"] = discrete_actions
        self.discrete_actions = discrete_actions
        self.algo_policy = self.init_algorithm(**kwargs)
        self.parameterized = kwargs["parameterized"]
        self.param_process = None
        # self.map_action = self.algo_policy.map_action
        self.exploration_noise = self.algo_policy.exploration_noise
        self.grad_epoch = kwargs['grad_epoch']
        self.input_norm_timer = 0

    def cpu(self):
        super().cpu()
        self.assign_device("cpu")

    def cuda(self, device=None):
        super().cuda()
        if device is not None:
            self.assign_device(device)


    def assign_device(self, device):
        '''
        Tianshou stores the device on a variable inside the internal models. This must be pudated when changing CUDA/CPU devices
        '''
        if type(device) == int:
            device = 'cuda:' + str(device)
        self.algo_policy.device = device
        if self.algo_name in ["ddpg"]:
            self.algo_policy.actor.last.device = device
            self.algo_policy.critic.last.device = device
            self.algo_policy.actor.device = device
            self.algo_policy.critic.device = device
        if self.algo_name in ["sac"]: # TODO: should have dependence on discrete actions 
            self.algo_policy.actor.mu.device = device
            self.algo_policy.actor.sigma.device = device
            self.algo_policy.critic1.last.device = device
            self.algo_policy.critic2.last.device = device
            self.algo_policy.actor.device = device
            self.algo_policy.critic1.device = device
            self.algo_policy.critic2.device = device
        if self.algo_name in ["ppo"]: # TODO: should have dependence on discrete actions
            self.algo_policy.actor.mu.device = device
            self.algo_policy.actor.sigma.device = device
            self.algo_policy.critic.last.device = device
            self.algo_policy.actor.device = device
            self.algo_policy.critic.device = device

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
                ainp_dim = 0
            else:
                ainp_dim = np.prod(action_shape)
                cinp_shape = int(input_shape + ainp_dim)
                cout_shape = 1
                aout_shape = action_shape
                hidden_sizes = args.hidden_sizes

            critic_bo = args.bound_output
            args.bound_output = 0
            actor = PolicyType(num_inputs=input_shape, num_outputs=aout_shape, aggregate_final=True, **args)
            args.bound_output = critic_bo
            if args.policy_type == 'pair': args.first_obj_dim = args.first_obj_dim + ainp_dim
            critic = PolicyType(num_inputs=cinp_shape, num_outputs=cout_shape, action_dim=ainp_dim, aggregate_final=True, continuous_critic=True, **args)
            if discrete_actions: critic = Critic(critic, last_size=action_shape, device=device).to(device)
            else: critic = Critic(critic, device=device).to(device)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
            if self.algo_name in _double_critic:
                if discrete_actions: actor = Actor(actor, action_shape, device=device).to(device)
                else: actor = ActorProb(actor, action_shape, device=device, max_action=max_action, unbounded=True, conditioned_sigma=True).to(device)
                critic2 = PolicyType(num_inputs=cinp_shape, num_outputs=cout_shape, action_dim=ainp_dim, aggregate_final=True, continuous_critic=True, **args)
                if discrete_actions: critic2 = Critic(critic2, last_size=action_shape, device=device).to(device)
                else: critic2 = Critic(critic2, device=device).to(device)
                critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
            else:
                actor = Actor(actor, action_shape, device=device, max_action=max_action).to(device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
            if args.sac_alpha == -1 and self.algo_name == "sac":
                target_entropy = -np.prod(self.action_space.shape)
                log_alpha = torch.zeros(1, requires_grad=True, device=device)
                alpha_optim = torch.optim.Adam([log_alpha], lr=1e-4) # TODO alpha learning rate not hardcoded
                args.sac_alpha = (target_entropy, log_alpha, alpha_optim)
        elif self.algo_name in ['dqn']:
            critic = PolicyType(num_inputs=input_shape, num_outputs=action_shape, aggregate_final=True, **args)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        elif self.algo_name in ['ppo']:
            if discrete_actions:
                hsizes = args.hidden_sizes
                args.hidden_sizes = args.hidden_sizes[:-1]
                net = PolicyType(num_inputs=input_shape, num_outputs=args.hidden_sizes[-1], aggregate_final=True, **args)
                args.hidden_sizes = hsizes
                actor = Actor(net, action_shape, device=device).to(device)
                critic = Critic(net, device=device).to(device)
            else: # there might be some issues with bound_output
                hsizes = args.hidden_sizes
                args.hidden_sizes = args.hidden_sizes[:-1]
                net = PolicyType(cuda=args.cuda, num_inputs=input_shape, num_outputs=action_shape, aggregate_final=True, **args)
                args.hidden_sizes = hsizes
                actor = ActorProb(net, action_shape, max_action=max_action, device=device).to(device)
                critic = Critic(PolicyType(cuda=args.cuda, num_inputs=cinp_shape, action_dim=ainp_dim, aggregate_final=True, continuous_critic=True, num_outputs=1, **args), device=device).to(device)
            actor_optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=args.actor_lr)
        return actor, actor_optim, critic, critic_optim, critic2, critic2_optim

    def set_eps(self, epsilon): # not all algo policies have set eps
        self.epsilon = epsilon
        if hasattr(self.algo_policy, "set_eps"):
            self.algo_policy.set_eps(epsilon)
        if hasattr(self.algo_policy, "set_exp_noise"):
            self.algo_policy.set_exp_noise(GaussianNoise(sigma=epsilon))


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
                                                                            exploration_noise=args.exploration_noise,
                                                                            estimation_step=args.lookahead, action_space=args.action_space,
                                                                            action_bound_method='clip')
        elif self.algo_name == "sac":
            print(args.sac_alpha)
            if args.discrete_actions:
                policy = ts.policy.DiscreteSACPolicy(
                        args.actor, args.actor_optim, args.critic, args.critic_optim, args.critic2, args.critic2_optim,
                        tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha, estimation_step=args.lookahead,
                        reward_normalization=args.reward_normalization, deterministic_eval=args.deterministic_eval)
            else:
                policy = ts.policy.SACPolicy(args.actor, args.actor_optim, args.critic, args.critic_optim, args.critic2, args.critic2_optim,
                                                                            tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha,
                                                                            exploration_noise=args.exploration_noise,
                                                                            estimation_step=args.lookahead, action_space=args.action_space,
                                                                            action_bound_method='clip', deterministic_eval=args.deterministic_eval)
        # support as many algos as possible, at least ddpg, dqn SAC
        return policy

    def save(self, pth, name):
        collect_fn = self.collect
        la = self.learning_algorithm
        sample_buffer = self.sample_buffer
        opt = self.option
        self.collect = None
        self.learning_algorithm = None
        self.sample_buffer = None
        self.option = None
        torch.save(self, os.path.join(pth, name + ".pt"))
        self.collect = collect_fn
        self.learning_algorithm = la
        self.sample_buffer = sample_buffer
        self.option = opt


    def compute_Q(
        self, batch: Batch, nxt: bool
    ) -> torch.Tensor:
        comp = batch.obs_next if nxt else batch.obs
        if self.algo_name in ['sac']:
            Q_val = self.algo_policy.critic1(comp, batch.act)
        if self.algo_name in ['ddpg']:
            Q_val = self.algo_policy.critic(comp, batch.act)
        if self.algo_name in ['dqn']:
            Q_val = self.algo_policy(batch, input="obs_next" if nxt else "obs").logits
        return Q_val

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
        # print("input: ", batch.obs, self.use_input_norm, self.input_mean, self.input_var)
        # print("forward call")
        batch = copy.deepcopy(batch) # make sure input norm does not alter the input batch
        # self.apply_input_norm(batch)
        vals= self.algo_policy(batch, state = state, input=input, **kwargs)
        return vals

    def compute_input_norm(self, buffer):
        if len(buffer) > 0:
            error
            avail = buffer.sample(0)[0]
            # print("trying compute", len(buffer), avail.obs.shape)
            # print(len(avail))
            if len (avail) >= 500: # need at least 500 values before applying input variance, typically this is the number of random actions
                if len(avail) > 20000: # only use the last 20k states
                    avail = avail[len(avail) - 20000:]
                self.input_var = np.sqrt(np.var(avail.obs, axis=0))
                self.input_var[self.input_var < .0001] = .0001 # to prevent divide by zero errors
                self.input_mean = np.mean(avail.obs, axis=0)
                # print("computing input norm", self.input_mean, self.input_var)
                if self.algo_name in _actor_critic + ['ppo']:
                    self.algo_policy.actor.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['sac']: 
                    self.algo_policy.critic1.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic1_old.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['ppo', 'ddpg']:
                    self.algo_policy.critic.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic_old.preprocess.update_norm(self.input_mean, self.input_var)
                if self.algo_name in ['dqn']:
                    self.algo_policy.model.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.model_old.update_norm(self.input_mean, self.input_var)
                if self.algo_name in _double_critic:
                    self.algo_policy.critic2.preprocess.update_norm(self.input_mean, self.input_var)
                    self.algo_policy.critic2_old.preprocess.update_norm(self.input_mean, self.input_var)
                return True
        return False

    def apply_input_norm(self, batch):
        if self.use_input_norm:
            batch.update(obs=(batch.obs - self.input_mean) / self.input_var)

    def update_time(self):
        self.epsilon_timer += 1
        if self.epsilon_schedule > 0 and self.epsilon_timer % self.epsilon_schedule == 0: # only updates every epsilon_schedule time steps
            self.epsilon = self.epsilon_base + (1-self.epsilon_base) * np.exp(-max(0, self.epsilon_timer - self.pretrain_iters)/self.epsilon_schedule) 
            self.set_eps(self.epsilon)

    def update_norm(self, buffer):
        if self.use_input_norm:
            if self.input_norm_timer == 1000:
                if self.compute_input_norm(buffer):
                    self.input_norm_timer = 0
            self.input_norm_timer += 1

    def update_la(self):
        if self.is_her:
            self.learning_algorithm.step()

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        '''
        don't call the algo_policy update, but carries almost the same logic
        however, inserting the param needs to be handled.
        '''
        for i in range(self.grad_epoch):
            use_buffer = buffer
            if self.sample_merged and len(self.learning_algorithm.replay_buffer) > 1000:
                her_buffer = self.learning_algorithm.replay_buffer
                if buffer is None or her_buffer is None:
                    return {}
                self.algo_policy.updating = True

                # sample from the main buffer and assign returns
                main_batch, main_indice = buffer.sample(sample_size)
                main_batch = self.algo_policy.process_fn(main_batch, buffer, main_indice)

                # sample from the HER buffer and assign returns
                her_batch, her_indice = her_buffer.sample(sample_size)
                her_batch = self.algo_policy.process_fn(her_batch, her_buffer, her_indice)
                
                num_her = int(sample_size * self.learning_algorithm.select_positive) # always samples the same ratio from HER and main
                # print([(k, batch[k].shape, main_batch[k].shape) for k in batch.keys()])
                batch = her_batch[:num_her]
                batch.cat_([main_batch[:sample_size-num_her]])
            else:
                if self.is_her:
                    use_buffer = self.sample_buffer(buffer)
                    # print(len(use_buffer))
                if use_buffer is None:
                    return {}
                batch, indice = use_buffer.sample(sample_size)
                self.algo_policy.updating = True
                batch = self.algo_policy.process_fn(batch, use_buffer, indice)

            kwargs["batch_size"] = sample_size
            kwargs["repeat"] = 2
            # print("process fn", batch.obs[:5])
            # print(batch.act.shape, use_buffer.act.shape)
            # self.apply_input_norm(batch)
            # print("input norm", indice, self.input_mean, self.input_var, batch.obs, batch.rew, batch.done)
            result = self.algo_policy.learn(batch, **kwargs)
            if i == 0: cumul_losses = result
            else: 
                for k in result.keys():
                    cumul_losses[k] += result[k] 
            # print("after learn")
            # print(result)
            if self.sample_merged and len(self.learning_algorithm.replay_buffer) > 1000:
                self.algo_policy.post_process_fn(main_batch, buffer, main_indice)
                self.algo_policy.post_process_fn(her_batch, her_buffer, her_indice)
            else:
                self.algo_policy.post_process_fn(batch, use_buffer, indice)                
            # self.restore_obs(batch, orig_obs, orig_next)
            # self.restore_buffer(orig_obs_buffer, orig_next_buffer, buffer_idces)
            self.algo_policy.updating = False
        return {k: cumul_losses[k] / self.grad_epoch for k in cumul_losses.keys()}



# policy_forms = {"basic": BasicPolicy, "image": ImagePolicy, 'grid': GridWorldPolicy, 'actorcritic': BasicActorCriticPolicy}
