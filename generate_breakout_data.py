# Create Breakout Dataset
from Environments.SelfBreakout.breakout_screen import Screen, RandomPolicy, BounceAnglePolicy
import sys

if __name__ == '__main__':
    screen = Screen()
    policy = RandomPolicy(4)
    # policy = BouncePolicy(4)
    if len(sys.argv) == 4:
        screen = Screen(target_mode=True)
        policy = BounceAnglePolicy(4)
        screen.run(policy, render=True, iterations = int(sys.argv[2]), duplicate_actions=1, save_path=sys.argv[1], save_raw = True, angle_mode=True)
    else:
        screen.run(policy, render=True, iterations = int(sys.argv[2]), duplicate_actions=1, save_path=sys.argv[1], save_raw = True)
    # demonstrate(sys.argv[1], 1000)
