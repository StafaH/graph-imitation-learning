#!/usr/bin/env python
import argparse


class Config(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<Config ' + dict.__repr__(self) + '>'


def load_default_config():

    config = Config({})
    # config.data_dir = 'C:/code/graph-imitation-learning/data/'
    config.data_dir = "./data/"
    config.log_dir = 'logs/'
    config.num_epochs = 500
    config.model_name = 'graph_model'
    config.batch_size = 64

    return config


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

    # evaluation stuff
    parser.add_argument('--max_episode_length', type=int, default=500)
    parser.add_argument('--eval_interval',
                        type=int,
                        default=10,
                        help="evalution interval during training")
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        help='folder path to load checkpoint from')

    # model stuff
    parser.add_argument('--model_name', type=str, default='mlp')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--hidden_dims",
                        nargs='+',
                        type=int,
                        default=[64, 64],
                        help="mlp hidden layer dimensions")

    return parser
