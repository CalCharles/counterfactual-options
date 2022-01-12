from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen

from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D

from EnvironmentModels.Pushing.pushing_environment_model import PushingEnvironmentModel
from Environments.Pushing.screen import Pushing

from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel


def initialize_environment(args, set_save=True, render=False):

    args.normalized_actions = False
    args.concatenate_param = True
    args.preprocess = None
    args.grayscale = args.env in ["SelfBreakout"]

    if args.env == "SelfBreakout":
        args.continuous = False
        nhigh, nwide, maxheight, no_breakout, hit_reset = args.block_shape
        environment = Screen(drop_stopping=args.drop_stopping, target_mode=args.target_mode, 
            num_rows = nhigh, num_columns = nwide, max_block_height=maxheight, no_breakout=bool(no_breakout), hit_reset=hit_reset,
            breakout_variant=args.variant_name)
        environment.seed(args.seed)
        environment_model = BreakoutEnvironmentModel(environment)
    elif args.env == "Nav2D":
        args.continuous = False
        environment = Nav2D()
        environment.seed(args.seed)
        environment_model = Nav2DEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess
    elif args.env.find("2DPushing") != -1:
        args.continuous = False
        environment = Pushing(pushgripper=True)
        if args.env == "StickPushing":
            environment = Pushing(pushgripper=False)
        environment.seed(args.seed)
        environment_model = PushingEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess
    elif args.env[:6] == "gymenv":
        args.continuous = True
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name= args.env[6:])
        environment.seed(args.seed)
        environment_model = GymEnvironmentModel(environment)
        args.normalized_actions = True
    elif args.env.find("RoboPushing") != -1:
        from EnvironmentModels.RobosuitePushing.robosuite_pushing_environment_model import RobosuitePushingEnvironmentModel
        from Environments.RobosuitePushing.robosuite_pushing import RoboPushingEnvironment

        args.continuous = True
        environment = RoboPushingEnvironment(control_freq=2, horizon=args.time_cutoff, renderable=render, num_obstacles=args.num_obstacles,
         standard_reward=-1, goal_reward=1, obstacle_reward=-2, out_of_bounds_reward=-2)
        environment.seed(args.seed)
        environment_model = RobosuitePushingEnvironmentModel(environment)
    elif args.env.find("RoboStick") != -1:
        from EnvironmentModels.RobosuiteStick.robosuite_stick_environment_model import RobosuiteStickEnvironmentModel
        from Environments.RobosuiteStick.robosuite_stick import RoboStickEnvironment

        args.continuous = True
        environment = RoboStickEnvironment(control_freq=2, horizon=args.time_cutoff, renderable=render,
         standard_reward=-1, goal_reward=1, out_of_bounds_reward=-2)
        environment.seed(args.seed)
        environment_model = RobosuiteStickEnvironmentModel(environment)
    if set_save: 
        if render == "Test":            
            environment.set_save(0, args.record_rollouts + "/" + render, 10000, save_raw=True)
        else:
            environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    args.environment = environment
    args.environment_model = environment_model
    return environment, environment_model, args