import argparse
from Tests.test_env_breakout import test_block
from Tests.test_terminate_reward import test_terminate_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unit_test')
    parser.add_argument('--tests', type=str, nargs='+', default=list(),
                        help='names of tests to run')
    args = parser.parse_args()
    if "breakout-env" in args.tests:
        test_block()
    if "terminate-reward" in args.tests:
        test_terminate_reward()
