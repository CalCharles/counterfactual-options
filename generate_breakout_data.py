# Create Breakout Dataset
from Environments.SelfBreakout.breakout_screen import Screen, RandomPolicy, AnglePolicy
import sys

if __name__ == '__main__':
    screen = Screen()
    policy = RandomPolicy(4)
    print(sys.argv) 
    # policy = BouncePolicy(4)
    if len(sys.argv) == 5:
        target_mode = False
        if sys.argv[3].find("tar") != -1:
            target_mode = True
        screen = Screen(target_mode=target_mode, breakout_variant=sys.argv[4])
        if sys.argv[3].find("ang") != -1:
            policy = AnglePolicy(4)
        print("running angle policy")
        screen.run(policy, render=True, iterations = int(sys.argv[2]), duplicate_actions=1, save_path=sys.argv[1], save_raw = True, angle_mode=True)
    else:
        screen.run(policy, render=True, iterations = int(sys.argv[2]), duplicate_actions=1, save_path=sys.argv[1], save_raw = True)
    # demonstrate(sys.argv[1], 1000)
