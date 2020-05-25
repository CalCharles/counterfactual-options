# Create Breakout Dataset
from SelfBreakout.breakout_screen import Screen, RandomPolicy

if __name__ == '__main__':
    screen = Screen()
    policy = RandomPolicy(4)
    # policy = BouncePolicy(4)
    screen.run(policy, render=True, iterations = 1000, duplicate_actions=1, save_path=sys.argv[1])
    # demonstrate(sys.argv[1], 1000)
