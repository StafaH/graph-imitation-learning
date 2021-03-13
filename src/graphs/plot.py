import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt 



def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window - 1:], yw_func


def smooth(scalars, weight): 
    """ exponential moving average, weight in [0,1] """ 
    last = scalars[0]  
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                                                 
    return smoothed


def plot_from_tensorboard_logs(src_dirs, legends, out_path="temp.jpg", scalar_name=None, window=None):
    """ generate plot for each stat from tfb log file in source dir 
        references: 
        - https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter
        - https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py
    """ 
    assert scalar_name is not None, "Must provide a scalar name to plot"
    assert len(src_dirs) == len(legends), "number of logs and legends should be equal"

    plt.clf() 
    stats = defaultdict(list)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for d, l in zip(src_dirs, legends):
        event_acc = EventAccumulator(d)
        event_acc.Reload()
        _, x, y = zip(*event_acc.Scalars(scalar_name))
        x, y = np.asarray(x), np.asarray(y)
        stats[l] = (x,y)
        if window:
            x, y = window_func(x, y, window, np.mean)
        plt.plot(x, y, label=l)

    plt.title("Traing Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path)
    plt.show()

    return stats       





if __name__ == "__main__":
    # hack
    plot_from_tensorboard_logs(
        [
            "logs/gnn",
            "logs/hid64-64-64_seed0_Mar10_15-38-14"
        ],
        [
            "gnn",
            "mlp"
        ],
        "loss_curve.jpg",
        scalar_name="loss",
        window=None
    )