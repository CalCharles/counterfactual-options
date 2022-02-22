import argparse

def add_env_args(parser):
    parser.add_argument('--variant', type=str, default="default")
    parser.add_argument('--use-angle-mode', action='store_true', default=False, help='use angle model in environment')
    parser.add_argument('--observation-type', choices=['delta', 'image', 'multi-block-encoding', 'sefull-encoding'])
