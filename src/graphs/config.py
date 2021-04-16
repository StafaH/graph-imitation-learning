#!/usr/bin/env python
import argparse


def get_base_parser():
    """To be augmented with algo-specific arguments."""
    parser = argparse.ArgumentParser(
        description='Graph with Imitation Learning')

    # bookkeeping stuff
    parser.add_argument('--tag', type=str, default='exp', help='experiment id')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('-r',
                        '--resume',
                        type=str,
                        default='',
                        help='path to last checkpoint (default = None)')
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/reach_target/',
                        help='path to imitation data')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs/',
                        help='path to experiment results')
    parser.add_argument("--eval",
                        action='store_true',
                        help="if to train or test")
    parser.add_argument("--dagger",
                        action='store_true',
                        help="if to train with dagger style (data aggregation)")

    # evaluation stuff
    parser.add_argument('--max_episode_length', type=int, default=200)
    parser.add_argument('--eval_interval',
                        type=int,
                        default=10,
                        help="evalution interval during training")
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        help='folder path to load checkpoint from')
    parser.add_argument("--render",
                        action='store_true',
                        help="if to render for test runs")
    parser.add_argument("--eval_when_train",
                        action='store_true',
                        help="if to evaluate during training")
    parser.add_argument('--eval_when_train_freq',
                        type=int,
                        default=25,
                        help="rollout interval during training")

    # model stuff
    parser.add_argument('--model_name', type=str, default='mlp')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--activation', type=str, default="tanh")
    parser.add_argument("--hidden_dims",
                        nargs='+',
                        type=int,
                        default=[64, 64],
                        help="mlp hidden layer dimensions")
    parser.add_argument("--use_dropout",
                        action='store_true',
                        help="if to use dropout before output layer")
    parser.add_argument('--num_stack',
                        type=int,
                        default=1,
                        help="how many (previous) features to use for input")
    parser.add_argument("--add_distractors",
                        action='store_true',
                        help="if to include distractors in features")

    # data stuff
    parser.add_argument('--episodes_per_update', type=int, default=10)
    parser.add_argument('--sub_epochs', type=int, default=10)
    parser.add_argument('--max_dataset_size', type=int, default=int(1e6))

    parser.add_argument('--action',
                        choices=['delta_nogripper, delta_withgripper', 'joint_velocity_nogripper', 'joint_velocity_withgripper'],
                        default='delta_nogripper',
                        help="The type of action the robot will use")

    return parser

def get_graph_parser():
    """ Parser with config variables specific to graph networks """
    
    parser = get_base_parser()

    parser.add_argument('--network',
                        choices=['gcn_state', 'gat_state', 'gcn_vision', 'gat_vision'],
                        required=True,
                        help="Network architecture to use: gcn_state, gat_state, gcn_vision, gat_vision")

    parser.add_argument("--graph_hidden_dims",
                        nargs='+',
                        type=int,
                        default=[64, 64, 64],
                        help="graph hidden embedding dimensions")
    
    parser.add_argument("--mlp_hidden_dims",
                        nargs='+',
                        type=int,
                        default=[64, 64, 64],
                        help="mlp hidden embedding dimensions")

    return parser


