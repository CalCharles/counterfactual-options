import sys

from Environments.RobosuitePushing.robosuite_pushing import RoboPushingEnvironment


if __name__ == "__main__":
    # first argument is num frames, second argument is save path
    pushing = RoboPushingEnvironment(joint_mode=sys.argv[3].find("joint") != -1, hard_obstacles=sys.argv[3].find("hard") != -1, num_obstacles=10)
    pushing.set_save(0, sys.argv[2], -1, save_raw=False)
    for i in range(int(sys.argv[1])):
        action = pushing.action_space.sample()
        pushing.step(action)

