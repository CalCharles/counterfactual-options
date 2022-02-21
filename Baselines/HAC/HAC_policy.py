


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
        elif self.algo_name == 'isl':
            policy = IteratedSupervisedPolicy(args.actor, args.actor_optim, label_smoothing=args.label_smoothing)
            self.sample_HER = True
        # support as many algos as possible, at least ddpg, dqn SAC
        return policy