from Environments.Pushing.screen import run, RandomPolicy
import sys
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train object recognition')
    parser.add_argument('savedir',
                        help='base directory to save results')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='number of training iterations')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion')
    parser.add_argument('--pushgripper', action='store_true', default=False,
                        help='run the pushing gripper domain')
    args = parser.parse_args()
    if args.demonstrate:
        demonstrate(args.savedir, args.num_frames, args.pushgripper)
    else:
        # run(args.savedir, args.num_frames, RandomPolicy(5), args.pushgripper)
        run(args.savedir, args.num_frames, RandomPolicy(5), args.pushgripper)
