#!/usr/bin/env python
"""Reach target task 
 
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
# # for pose output
# output_dim = 6
# for joint velocities
output_dim = 7


def make_env():
    """Constructs the environment"""
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)

    # action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_WORLD_FRAME)
    # action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode, obs_config=obs_config, headless=True)
    env.launch()

    task = env.get_task(ReachTarget)
    return env, task


def collect_data(config, task, relative_pos=False, joint_vel=True):
    """Generates live data for imitation learning."""
    demos = task.get_demos(config.episodes_per_update,
                           live_demos=True)  # -> List[List[Observation]]
    dataset = []

    # convert data
    for demo in demos:
        for t in range(len(demo) - 1):
            obs = demo[t]
            nodes, edge_index = obs_to_graph(obs, relative_pos=relative_pos)

            if joint_vel:
                y = torch.tensor([demo[t + 1].joint_velocities],
                                 dtype=torch.float)
            else:
                gripper_pose_next = demo[t + 1].gripper_pose
                gripper_pose_next = pose_quat_to_rpy(gripper_pose_next)
                y = torch.tensor([gripper_pose_next], dtype=torch.float)

            graph_data = Data(x=nodes,
                              edge_index=edge_index.t().contiguous(),
                              y=y)
            dataset.append(graph_data)
    return dataset


def obs_to_graph(obs, relative_pos=False):
    """"Converts raw obs from rlbench to graph attributes."""
    node_num = len(obs.task_low_dim_state)

    # nodes
    gripper_pose = obs.gripper_pose
    target_pose = obs.task_low_dim_state[0]
    distract_pose = obs.task_low_dim_state[1]
    distract2_pose = obs.task_low_dim_state[2]

    gripper_pose = pose_quat_to_rpy(gripper_pose)
    target_pose = pose_quat_to_rpy(target_pose)
    distract_pose = pose_quat_to_rpy(distract_pose)
    distract2_pose = pose_quat_to_rpy(distract2_pose)

    if relative_pos:
        # TODO: fix quaternion subtraction
        target_node = np.concatenate([target_pose - gripper_pose, target_enc])
        distract_node = np.concatenate(
            [distract_pose - gripper_pose, distract_enc])
        distract2_node = np.concatenate(
            [distract2_pose - gripper_pose, distract_enc])
        gripper_node = np.concatenate(
            [gripper_pose - gripper_pose, gripper_enc])
    else:
        target_node = np.concatenate([target_pose, target_enc])
        distract_node = np.concatenate([distract_pose, distract_enc])
        distract2_node = np.concatenate([distract2_pose, distract_enc])
        gripper_node = np.concatenate([gripper_pose, gripper_enc])

    nodes = torch.tensor(
        [target_node, distract_node, distract2_node, gripper_node],
        dtype=torch.float)

    # edges
    edge_index = torch.tensor(
        [[i, j] for i in range(node_num) for j in range(node_num) if i != j],
        dtype=torch.long)

    return nodes, edge_index


def make_graph_dataset(data_dir, relative_pos=False, joint_vel=True):
    """Constructs graph dataset from raw rlbench data."""
    # TODO(Justin): use torch.geometric.data.Dataset instead.
    dataset = []

    # get all episodes
    pattern = os.path.join(data_dir, "**/episodes/episode*")
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

            gripper_pose_next = data._observations[t + 1].gripper_pose
            gripper_pose_next = pose_quat_to_rpy(gripper_pose_next)
            if joint_vel:
                y = torch.tensor([data._observations[t + 1].joint_velocities],
                                 dtype=torch.float)
            else:
                y = torch.tensor([gripper_pose_next], dtype=torch.float)

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

    def act(self, graph):
        # concatenate all graph nodes to single vector,
        # assume all graphs have same number of nodes and and in order
        # assumes nodes from same graph are contiguous in batch
        flat_x = graph.x.reshape(-1, self.input_dim)
        action = self.model(flat_x)
        return action


class GraphAgent(MLPAgent):

    def __init__(self, config, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = GCNModel(input_dim,
                              output_dim,
                              config.hidden_dims,
                              act="relu",
                              output_act=None)

    def act(self, graph):
        if hasattr(graph, "batch"):
            # training
            batch = graph.batch
        else:
            # testing
            batch = torch.zeros(num_nodes).long()
        action = self.model(graph.x, graph.edge_index, batch)
        return action


def make_agent(config):
    """Constructs agent based on config."""
    if config.model_name == "mlp":
        input_dim = num_node_features * num_nodes
        agent = MLPAgent(config, input_dim, output_dim)
    elif config.model_name == "graph":
        input_dim = num_node_features
        agent = GraphAgent(config, input_dim, output_dim)
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

    # loss_eval_best = None
    env, task = make_env()

    for epoch in range(start_epoch, config.num_epochs):
        agent.train()

        print("Episode {}, Collecting new data...".format(epoch))
        dataset = collect_data(config, task)
        dataset_train, dataset_test = split_train_test(dataset)

        # Load the dataset (num_workers is async, set to 0 if using notebooks)
        loader = DataLoader(dataset_train, batch_size=config.batch_size)
        loader_test = DataLoader(dataset_test, batch_size=config.batch_size)

        for _ in range(config.sub_epochs):
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
            config.log_dir, "checkpoint", {
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # evaluation
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

        # if loss_eval_best is None or loss_eval_total < loss_eval_best:
        #     loss_eval_best = loss_eval_total
        #     save_checkpoint(
        #         config.log_dir, "checkpoint_best", {
        #             'epoch': epoch,
        #             'model_state_dict': agent.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': loss_total
        #         })


def test_policy(config):
    """Evaluates trained policy."""
    # make env
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_WORLD_FRAME)
    # action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
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
    checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint.pth")
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

            # # for pose output
            # pose_pred = action.squeeze().numpy()
            # full_action = pose_rpy_to_quat(pose_pred)

            # for joint velocities
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
