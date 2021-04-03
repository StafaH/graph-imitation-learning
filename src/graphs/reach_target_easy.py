#!/usr/bin/env python
"""Reach target task 

Examples 
    train with mlp::
        $ python src/graphs/reach_target_easy.py --tag feat_64x3 --seed 3 --hidden_dims 64 64 64 --data_dir data/reach_target/ --model_name mlp --num_epochs 1000

    test mlp::
        $ python src/graphs/reach_target_easy.py --eval --eval_batch_size 10 --max_episode_length 100 --checkpoint_dir logs/rt_mlp_feat_64x3/seed3_{...} --seed 6

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
from model.GCN import GCNModel
from utils import set_manual_seed, save_checkpoint, save_config, save_command
from utils import pose_quat_to_rpy, pose_rpy_to_quat

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------

# reach target task
task_name = "rt"

target_enc = np.array([1, 0, 0])
distract_enc = np.array([0, 1, 0])
gripper_enc = np.array([0, 0, 1])

num_node_features = 9
num_nodes = 4
input_dim = 24
# # for pose output
# output_dim = 6
# for joint velocities
output_dim = 7


def make_env():
    """Constructs the environment"""
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task = env.get_task(ReachTarget)
    return env, task


def obs_to_input(obs):
    """Construct input to mlp from raw rlbench obs."""
    features = []
    features.append(obs.joint_positions)  # (7,)
    features.append(obs.joint_velocities)  # (7,)
    features.append(obs.gripper_pose)  # (7,)

    target_pos = obs.task_low_dim_state[0][:3]
    features.append(target_pos)  # (3,)

    feature = np.concatenate(features)  # (24,)
    return feature


def load_data(data_dir):
    """Data before train-valid split."""
    # get all episodes
    pattern = os.path.join(data_dir, "**/episodes/episode*")
    episode_dirs = glob.glob(pattern, recursive=True)

    dataset = []
    for d in episode_dirs:
        # load data
        data_path = os.path.join(d, "low_dim_obs.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # convert data
        for t in range(len(data._observations) - 1):
            obs = data._observations[t]
            feature = obs_to_input(obs)

            y = torch.tensor([data._observations[t + 1].joint_velocities],
                             dtype=torch.float)

            dataset.append([feature, y])
    return dataset


class ReachTargetDataset(torch.utils.data.Dataset):

    def __init__(self, data_list):
        super().__init__()
        features = [data[0] for data in data_list]
        labels = [data[1] for data in data_list]
        self.features = np.vstack(features)
        self.labels = np.vstack(labels)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.features[idx])
        label = torch.FloatTensor(self.labels[idx])
        return feat, label


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

    def act(self, features):
        if len(features.shape) == 1:
            # for testing
            features = features.unsqueeze(0)
        action = self.model(features)
        return action


def make_agent(config):
    """Constructs agent based on config."""
    if config.model_name == "mlp":
        agent = MLPAgent(config, input_dim, output_dim)
    else:
        raise NotImplementedError
    return agent


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

    # Build Model
    agent = make_agent(config)
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
        config.log_dir, "_".join([task_name, config.model_name, config.tag]),
        "seed{}".format(config.seed) + '_' + current_time)

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

    # load dataset
    dataset = load_data(config.data_dir)
    dataset_train, dataset_test = split_train_test(dataset)

    dataset_train = ReachTargetDataset(dataset_train)
    loader = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=config.batch_size,
                                         shuffle=True)
    dataset_test = ReachTargetDataset(dataset_test)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size)

    loss_eval_best = None

    for epoch in range(start_epoch, config.num_epochs):
        agent.train()
        loss_total = 0.0

        for i, (features, labels) in enumerate(loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = agent.act(features)
            loss = torch.nn.functional.mse_loss(out, labels)
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
            config.log_dir, "checkpoint", {
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # evaluation
        agent.eval()
        loss_eval_total = 0.0

        for features, labels in loader_test:
            features = features.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = agent.act(features)
            loss_eval = torch.nn.functional.mse_loss(out, labels)
            loss_eval_total += loss_eval.item()

        loss_eval_total /= len(loader_test)
        summary_writer.add_scalar('loss_eval', loss_eval_total, epoch)
        summary_writer.flush()

        if loss_eval_best is None or loss_eval_total < loss_eval_best:
            loss_eval_best = loss_eval_total
            save_checkpoint(
                config.log_dir, "checkpoint_best", {
                    'epoch': epoch,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_total
                })


def test_policy(config):
    """Evaluates trained policy."""
    # make env
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    # make & restore agent
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
            feature = obs_to_input(obs)
            feature = torch.FloatTensor(feature)
            with torch.no_grad():
                action = agent.act(feature)

            vel_pred = action.squeeze().numpy()
            full_action = vel_pred

            gripper = [1.0]  # Always open
            full_action = np.concatenate([full_action, gripper], axis=-1)

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
    # env, task = make_env()

    # demos = task.get_demos(2, live_demos=True)  # -> List[List[Observation]]
    # import pdb
    # pdb.set_trace()
    # print()