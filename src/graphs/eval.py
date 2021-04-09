#!/usr/bin/env python
"""Reach target task 

Examples 

    $ python src/graphs/eval.py --eval --eval_batch_size 10 --max_episode_length 100 --checkpoint_dir logs/exp_seed53_Apr08_13-06-18 --seed 53

"""

import argparse
from datetime import datetime
import glob
import os
import pickle
from PIL import Image
import sys
from tqdm import tqdm
from types import SimpleNamespace as SN
import yaml
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from config import get_base_parser
from data import delta_in_pose, split_train_test
from model.graph import GATModel, GCNModel
from utils import set_manual_seed, save_checkpoint, save_config, save_command
from utils import pose_quat_to_rpy, pose_rpy_to_quat

TARGET_ENC = np.array([1, 0, 0])
DISTRACT_ENC = np.array([0, 1, 0])
GRIPPER_ENC = np.array([0, 0, 1])


class GraphAgent:

    def __init__(self, config, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = GCNModel(input_dim,
                     output_dim,
                     [64, 64, 64],
                     [64, 64, 64],
                     act="relu",
                     output_act=None)

    def to(self, device):
        self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def output(self, features):
        action = self.model(features)
        return action

    def act(self, obs, use_relative_position=True):
        dataset = []
        NUM_NODES = 4

        gripper_node = np.concatenate([obs.gripper_pose, GRIPPER_ENC])
        if use_relative_position:
            target_node = np.concatenate([
                delta_in_pose(obs.gripper_pose, obs.task_low_dim_state[0]),
                TARGET_ENC
            ])
            distract_node = np.concatenate([
                delta_in_pose(obs.gripper_pose, obs.task_low_dim_state[1]),
                DISTRACT_ENC
            ])
            distract2_node = np.concatenate([
                delta_in_pose(obs.gripper_pose, obs.task_low_dim_state[2]),
                DISTRACT_ENC
            ])
        else:
            target_node = np.concatenate([
                obs.task_low_dim_state[0],
                TARGET_ENC
            ])
            distract_node = np.concatenate([
                obs.task_low_dim_state[1],
                DISTRACT_ENC
            ])
            distract2_node = np.concatenate([
                obs.task_low_dim_state[2],
                DISTRACT_ENC
            ])

        nodes = torch.tensor(
            [target_node, distract_node, distract2_node, gripper_node],
            dtype=torch.float)

        # Build edge relationships (Fully Connected)
        edge_index = torch.tensor([[i, j]
                                   for i in range(NUM_NODES)
                                   for j in range(NUM_NODES)
                                   if i != j],
                                  dtype=torch.long)

        # Extract labels from future frame
        graph_data = Data(x=nodes,
                          edge_index=edge_index.t().contiguous())

        dataset.append(graph_data)
        loader = DataLoader(dataset, batch_size=1)

        for data in loader:         
            out = self.model(data.x, data.edge_index, data.batch)
            break

        arm = out.detach().numpy() # Only one thing in batch

        # Normalize to be unit quaternion
        x, y, z, q1x, q1y, q1z, q1w = arm[0]
        q1 = Quaternion(q1w, q1x, q1y, q1z)
        q1 = q1.unit
        qw, qx, qy, qz = list(q1)
        
        # Gripper always open
        return np.asarray([x, y, z, qx, qy, qz, qw, 1.0])


def make_agent(config):
    """Constructs agent based on config."""
    # if config.model_name == "mlp":
    #     agent = MLPAgent(config, INPUT_DIM, OUTPUT_DIM)
    # else:
    #     raise NotImplementedError
    agent = GraphAgent(config, 10, 7)
    return agent

def test_policy(config):
    """Evaluates trained policy."""
    
    # Make Environment
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    # Load Model and Restore Config
    config_path = os.path.join(config.checkpoint_dir, "config.yaml")
    with open(config_path, "r") as f:
        # NOTE: hack!
        old_config = SN()
        old_config.__dict__.update(yaml.load(f))
    agent = make_agent(old_config)

    # NOTE: hack!
    # checkpoint_path = os.path.join(config.checkpoint_dir, [
    #     f for f in os.listdir(config.checkpoint_dir) if '.pth' in f
    # ][0])
    checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pth")
    checkpoint = torch.load(checkpoint_path)
    agent.load_state_dict(checkpoint['model_state_dict'])

    # run trajectories
    total_lengths = []
    total_rewards = []

    for _ in range(config.eval_batch_size):
        descriptions, obs = task.reset()
        done = False
        length = 0
        total_r = 0

        while not done and length < config.max_episode_length:
            with torch.no_grad():
                action = agent.act(obs)

            obs, reward, done = task.step(action)
            print("step: {}, returns: {}".format(length, total_r))
            total_r += reward
            length += 1

        total_lengths.append(length)
        total_rewards.append(total_r)

    # log and clean up
    total_lengths = np.asarray(total_lengths)
    total_rewards = np.asarray(total_rewards)
    print("Eval traj lengths: {} +/- {}".format(total_lengths.mean(),
                                                total_lengths.std()))
    print("Eval traj returns: {} +/- {}".format(total_rewards.mean(),
                                                total_rewards.std()))

    print('Done')
    env.shutdown()


# -----------------------------------------------------------------------------------
#                   Execution
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = get_base_parser()
    config = parser.parse_args()
    
    test_policy(config)
