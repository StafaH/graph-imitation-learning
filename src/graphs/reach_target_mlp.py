#!/usr/bin/env python
"""Reach target task with MLP

Available keys in each observation
    * left_shoulder_rgb 
    * left_shoulder_depth 
    * left_shoulder_mask 
    * right_shoulder_rgb 
    * right_shoulder_depth 
    * right_shoulder_mask 
    * wrist_rgb 
    * wrist_depth 
    * wrist_mask 
    * front_rgb 
    * front_depth 
    * front_mask 
    * joint_velocities 
    * joint_positions 
    * joint_forces 
    * gripper_open 
    * gripper_pose 
    * gripper_matrix 
    * gripper_joint_positions 
    * gripper_touch_forces 
    * wrist_camera_matrix 
    * task_low_dim_state    
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

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from config import get_base_parser
from data import split_train_test
from model.mlp import MLP
from utils import set_manual_seed, save_checkpoint, save_config, save_command

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------

target_enc = np.array([1, 0, 0])
distract_enc = np.array([0, 1, 0])
gripper_enc = np.array([0, 0, 1])


def obs_to_graph(obs, relative_pos=True):
    """"Converts raw obs from rlbench to graph attributes."""
    node_num = len(obs.task_low_dim_state)

    # nodes
    if relative_pos:
        gripper_pos = obs.task_low_dim_state[3]
        target_node = np.concatenate(
            [obs.task_low_dim_state[0] - gripper_pos, target_enc])
        distract_node = np.concatenate(
            [obs.task_low_dim_state[1] - gripper_pos, distract_enc])
        distract2_node = np.concatenate(
            [obs.task_low_dim_state[2] - gripper_pos, distract_enc])
        gripper_node = np.concatenate(
            [obs.task_low_dim_state[3] - gripper_pos, gripper_enc])
    else:
        target_node = np.concatenate([obs.task_low_dim_state[0], target_enc])
        distract_node = np.concatenate(
            [obs.task_low_dim_state[1], distract_enc])
        distract2_node = np.concatenate(
            [obs.task_low_dim_state[2], distract_enc])
        gripper_node = np.concatenate([obs.task_low_dim_state[3], gripper_enc])

    nodes = torch.tensor(
        [target_node, distract_node, distract2_node, gripper_node],
        dtype=torch.float)

    # edges
    edge_index = torch.tensor(
        [[i, j] for i in range(node_num) for j in range(node_num) if i != j],
        dtype=torch.long)

    return nodes, edge_index


def make_graph_dataset(data_dir, relative_pos=True):
    """Constructs graph dataset from raw rlbench data."""
    # TODO(Justin): use torch.geometric.data.Dataset instead.
    dataset = []

    # get all episodes
    pattern = os.path.join(data_dir, "*/episodes/episode*")
    episode_dirs = glob.glob(pattern, recursive=True)

    for d in episode_dirs:
        # load data
        data_path = os.path.join(d, "low_dim_obs.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # convert data
        for t in range(len(data._observations) - 1):
            obs = data._observations[t]
            nodes, edge_index = obs_to_graph(obs, relative_pos=relative_pos)

            state_next = data._observations[t + 1].task_low_dim_state
            y = torch.tensor([state_next[3]], dtype=torch.float)

            graph_data = Data(x=nodes,
                              edge_index=edge_index.t().contiguous(),
                              y=y)
            dataset.append(graph_data)
    return dataset


class MLPAgent:

    def __init__(self, config, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = MLP(input_dim,
                         output_dim,
                         hidden_dims=config.hidden_dims,
                         act="relu",
                         output_act=None,
                         init_weights=True)

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

    def act(self, obs):
        # concatenate all graph nodes to single vector,
        # assume all graphs have same number of nodes and and in order
        # assumes nodes from same graph are contiguous in batch
        flat_x = obs.x.reshape(-1, self.input_dim)
        action = self.model(flat_x)
        return action


# -----------------------------------------------------------------------------------
#                   Main
# -----------------------------------------------------------------------------------


def train(config):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Manual seeds for reproducible results
    set_manual_seed(config.seed)
    print(config)

    print("----------------------------------------")
    print("Loading Model and Data")
    print("----------------------------------------\n")

    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Current device is set to: ', device)

    if (not os.path.exists(config.data_dir)):
        print('The data directory does not exist:', config.data_dir)
        return

    dataset = make_graph_dataset(config.data_dir)
    dataset_train, dataset_test = split_train_test(dataset)

    # Load the dataset (num_workers is async, set to 0 if using notebooks)
    loader = DataLoader(dataset_train, batch_size=config.batch_size)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size)

    # Build Model
    input_dim = 24
    output_dim = 3
    agent = MLPAgent(config, input_dim, output_dim)
    agent.to(device=device)

    optimizer = torch.optim.Adam(agent.model.parameters(), 1e-3)
    start_epoch = 0

    # Load a model if resuming training
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Resuming Training from Epoch: {start_epoch} at Loss:{loss}')

    # Create a log directory using the current timestamp
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    config.log_dir = os.path.join(
        config.log_dir,
        config.tag + '_' + "seed{}".format(config.seed) + '_' + current_time)

    os.makedirs(config.log_dir, exist_ok=True)
    print('Logs are being written to {}'.format(config.log_dir))
    print('Use TensorBoard to look at progress!')
    summary_writer = SummaryWriter(config.log_dir)

    # Dump training information in log folder (for future reference!)
    save_config(config, config.log_dir)
    save_command(config.log_dir)

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    pbar = tqdm(total=config.num_epochs)
    pbar.n = start_epoch
    pbar.refresh()
    for epoch in range(start_epoch, config.num_epochs):
        agent.train()
        loss_total = 0.0

        for i, data in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = agent.act(data)
            loss = torch.nn.functional.mse_loss(out, data.y)
            loss_total += loss.item()

            loss.backward()
            optimizer.step()

        loss_total /= len(loader)

        # logging
        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss_total:.5f}')
        summary_writer.add_scalar('loss', loss_total, epoch)
        summary_writer.flush()

        # checkpoint
        save_checkpoint(
            config.log_dir, config.model_name, {
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # evaluation
        if epoch > 0 and epoch % config.eval_interval == 0:
            agent.eval()
            loss_eval_total = 0.0

            for data in loader_test:
                data = data.to(device)
                with torch.no_grad():
                    out = agent.act(data)
                loss_eval = torch.nn.functional.mse_loss(out, data.y)
                loss_eval_total += loss_eval.item()

            loss_eval_total /= len(loader_test)
            summary_writer.add_scalar('loss_eval', loss_eval_total, epoch)
            summary_writer.flush()


def test_policy(config):
    """Evaluates trained policy."""
    # make env
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_WORLD_FRAME)
    # action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    # make & restore agent
    config_path = os.path.join(config.checkpoint_dir, "config.yaml")
    with open(config_path, "r") as f:
        # NOTE: hack!
        old_config = SN()
        old_config.__dict__.update(yaml.load(f))

    input_dim = 24
    output_dim = 3
    agent = MLPAgent(old_config, input_dim, output_dim)
    # NOTE: hack!
    checkpoint_path = os.path.join(config.checkpoint_dir, [
        f for f in os.listdir(config.checkpoint_dir) if '.pth' in f
    ][0])
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
            nodes, edge_index = obs_to_graph(obs)
            graph = Data(x=nodes, edge_index=edge_index.t().contiguous())
            with torch.no_grad():
                action = agent.act(graph)

            arm = action.squeeze().numpy()
            gripper = [1.0]  # Always open
            rotation = obs.gripper_pose[3:]
            full_action = np.concatenate([arm, rotation, gripper], axis=-1)

            obs, reward, done = task.step(full_action)
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
    if config.eval:
        test_policy(config)
    else:
        train(config)
