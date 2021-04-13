#!/usr/bin/env python

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
    config.data_dir = 'C:/code/graph-imitation-learning/data/reach_target/'
    config.log_dir = 'logs/'
    config.num_keypoints = 4
    config.num_channels = 3
    config.num_epochs = 100
    config.model_name = 'transporter_model'
    config.batch_size = 128

    return config
