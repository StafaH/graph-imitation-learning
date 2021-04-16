import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

COLORS = [
    'blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'brown',
    'purple', 'pink', 'orange', 'teal', 'coral', 'lightblue', 'lime',
    'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',
    'lightpurple', 'darkred', 'darkblue'
]


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


def plot_from_tensorboard_logs(legend_dir_specs,
                               out_path="temp.jpg",
                               scalar_name=None,
                               window=None):
    """ generate plot for each stat from tfb log file in source dir 
        references: 
        - https://stackoverflow.com/questions/36700404/tensorflow-opening-log-data-written-by-summarywriter
        - https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py
    """
    assert scalar_name is not None, "Must provide a scalar name to plot"

    plt.clf()
    stats = defaultdict(list)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for l, d in legend_dir_specs.items():
        event_acc = EventAccumulator(d)
        event_acc.Reload()
        _, x, y = zip(*event_acc.Scalars(scalar_name))
        x, y = np.asarray(x), np.asarray(y)
        stats[l] = (x, y)
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


def plot_with_seeds(legend_dir_specs,
                    out_path="temp.jpg",
                    scalar_name=None,
                    title="Traing Curves",
                    xlabel="Epochs",
                    ylabel="Loss",
                    window=None):
    """Generates plot among algos, each with several seed runs."""
    assert scalar_name is not None, "Must provide a scalar name to plot"

    plt.clf()
    # algo name to list of seed runs
    stats = defaultdict(list)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # get all stats
    for l, dirs in legend_dir_specs.items():
        for d in dirs:
            event_acc = EventAccumulator(d)
            event_acc.Reload()
            _, x, y = zip(*event_acc.Scalars(scalar_name))
            x, y = np.asarray(x), np.asarray(y)
            if window:
                x, y = window_func(x, y, window, np.mean)
            stats[l].append([x, y])
            del event_acc

    # prepare for plot
    x_max = float("inf")
    for _, runs in stats.items():
        for x, y in runs:
            x_max = min(x_max, len(x))
    processed_stats = {}
    for name, runs in stats.items():
        x = np.array([x[:x_max] for x, _ in runs])[0]
        y = np.stack([y[:x_max] for _, y in runs])
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        processed_stats[name] = [x, y_mean, y_std]

    # actual plot
    for i, name in enumerate(processed_stats.keys()):
        color = COLORS[i]
        x, y_mean, y_std = processed_stats[name]
        plt.plot(x, y_mean, label=name, color=color)
        plt.fill_between(x,
                         y_mean + 1 * y_std,
                         y_mean - 1 * y_std,
                         alpha=0.3,
                         color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # # trianign loss for rt-mlp
    # plt.ylim(ymax=0.1e-5, ymin=-0.1e-6)
    # # trianign loss for pl-mlp
    # plt.ylim(ymax=0.001)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()

    return stats, processed_stats


if __name__ == "__main__":
    # hack
    # plot_from_tensorboard_logs(
    #     {
    #         "gnn": "logs/gnn",
    #         "mlp": "logs/hid64-64-64_seed0_Mar10_15-38-14"
    #     },
    #     "loss_curve.jpg",
    #     scalar_name="loss",
    #     window=None)

    # plot_with_seeds(
    #     {
    #         # "gnn":
    #         #     ["logs/rt_gnn/seed0", "logs/rt_gnn/seed1", "logs/rt_gnn/seed2"],
    #         "mlp": [
    #             "logs/rt_mlp_fix_64x3_lr/seed6_Apr14_22-18-52",
    #             "logs/rt_mlp_fix_64x3_lr/seed1_Apr15_15-23-38",
    #             "logs/rt_mlp_fix_64x3_lr/seed9_Apr15_15-43-02",
    #             "logs/rt_mlp_fix_64x3_lr/seed3_Apr15_16-39-38"
    #         ],
    #     },
    #     out_path="rt_mlp_loss.jpg",
    #     scalar_name="loss",
    #     title="Training Loss",
    #     xlabel="Epochs",
    #     ylabel="Loss",
    #     window=10)
    # # out_path="rt_mlp_reward.jpg",
    # # scalar_name="loss_eval/total_rewards",
    # # title="Average Reward",
    # # xlabel="Epochs",
    # # ylabel="Reward",
    # # window=10)

    plot_with_seeds(
        {
            # "gnn":
            #     ["logs/rt_gnn/seed0", "logs/rt_gnn/seed1", "logs/rt_gnn/seed2"],
            "mlp": [
                "logs/pl_mlp_fix_64x3_lr/seed6_Apr14_22-15-15",
                "logs/pl_mlp_fix_64x3_lr/seed1_Apr16_00-39-22",
                "logs/pl_mlp_fix_64x3_lr/seed9_Apr16_00-15-24",
                "logs/pl_mlp_fix_64x3_lr/seed3_Apr16_00-22-39"
            ],
        },
        # out_path="pl_mlp_loss.jpg",
        # scalar_name="loss",
        # title="Training Loss",
        # xlabel="Epochs",
        # ylabel="Loss",
        # window=10)
        out_path="pl_mlp_reward.jpg",
        scalar_name="loss_eval/total_rewards",
        title="Average Reward",
        xlabel="Epochs",
        ylabel="Reward",
        window=10)
