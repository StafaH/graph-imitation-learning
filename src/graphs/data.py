import os
import glob
import json
import numpy as np
import pickle
import random
import re

import torch
from torch_geometric.data import Data, DataLoader

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------


def ProcessStateToGraphData(file):
    data = np.load(file, allow_pickle=True)
    return data


def split_train_test(dataset, train_ratio=0.8):
    """Makes training and testing sets (could extend for validation)."""
    n = len(dataset)
    n_train = int(n * train_ratio)

    random.shuffle(dataset)
    d_train = dataset[:n_train]
    d_test = dataset[n_train:]
    return d_train, d_test


# -----------------------------------------------------------------------------------
#                   Reach Target
# -----------------------------------------------------------------------------------

target_enc = np.array([1, 0, 0])
distract_enc = np.array([0, 1, 0])
gripper_enc = np.array([0, 0, 1])


def load_data_to_graph(data_dir, use_relative_pos=True):
    """Constructs dataset for behavior cloning from JSON dataset."""
    # get all episodes
    pattern = os.path.join(data_dir, "state_data_*.json")
    episode_files = glob.glob(pattern)

    # construct dataset
    dataset = []

    for f_path in episode_files:
        with open(f_path, "r") as f:
            data = json.load(f)
        obs = data["obs"]

        for k in range(data["length"] - 1):
            # nodes
            node_num = len(obs[k])

            if use_relative_pos:
                target_node = np.concatenate([
                    np.array(obs[k]["target"]) - np.array(obs[k]["tip"]),
                    target_enc
                ])
                distract_node = np.concatenate([
                    np.array(obs[k]["distractor0"]) - np.array(obs[k]["tip"]),
                    distract_enc
                ])
                distract2_node = np.concatenate([
                    np.array(obs[k]["distractor1"]) - np.array(obs[k]["tip"]),
                    distract_enc
                ])
                gripper_node = np.concatenate(
                    [np.array(obs[k]["tip"]), gripper_enc])
            else:
                target_node = np.concatenate(
                    [np.array(obs[k]["target"]), target_enc])
                distract_node = np.concatenate(
                    [np.array(obs[k]["distractor0"]), distract_enc])
                distract2_node = np.concatenate(
                    [np.array(obs[k]["distractor1"]), distract_enc])
                gripper_node = np.concatenate(
                    [np.array(obs[k]["tip"]), gripper_enc])

            nodes = torch.tensor(
                [target_node, distract_node, distract2_node, gripper_node],
                dtype=torch.float)

            # edges
            edge_index = torch.tensor([[i, j]
                                       for i in range(node_num)
                                       for j in range(node_num)
                                       if i != j],
                                      dtype=torch.long)

            # label
            y = torch.tensor([np.array(obs[k + 1]["tip"])], dtype=torch.float)

            graph_data = Data(x=nodes,
                              edge_index=edge_index.t().contiguous(),
                              y=y)
            dataset.append(graph_data)

    return dataset


def preprocess_data(data_dir, out_dir):
    """Converts raw data from RLBench to desired format.

    Use JSON format for easy visualizaing. 
    """
    os.makedirs(out_dir, exist_ok=True)
    # get all episodes
    pattern = os.path.join(data_dir, "*/episodes/episode*")
    episode_dirs = glob.glob(pattern, recursive=True)

    for d in episode_dirs:
        # load data
        data_path = os.path.join(d, "low_dim_obs.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # convert data
        formatted_data = {"obs": [], "length": len(data._observations)}

        for obs in data._observations:
            vel = obs.joint_velocities
            state = obs.task_low_dim_state

            result = {
                "target": state[0].tolist(),
                "distractor0": state[1].tolist(),
                "distractor1": state[2].tolist(),
                "tip": state[3].tolist()
            }
            formatted_data["obs"].append(result)

        # save processed data
        variation_match = re.search(r"variation\d+", d)
        variation_num = int(variation_match.group().replace("variation", ""))
        episode_match = re.search(r"episodes/episode\d+", d)
        episode_num = int(episode_match.group().replace("episodes/episode", ""))
        data_out_path = os.path.join(
            out_dir, "state_data_{}_{}.json".format(variation_num, episode_num))
        with open(data_out_path, "w") as f:
            json.dump(formatted_data, f, indent=4)


# -----------------------------------------------------------------------------------
#                   Block Stacking
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
#                   Preprocessing only
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = "data/reach_target"
    out_dir = "data/reach_target_processed"
    preprocess_data(data_dir, out_dir)