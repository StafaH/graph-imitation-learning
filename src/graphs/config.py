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
    config.data_dir = 'C:/code/graph-imitation-learning/data/'
    config.log_dir = 'logs/'
    config.num_epochs = 1000
    config.model_name = 'graph_model'
    config.batch_size = 64

    return config
