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
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from collections import deque
from copy import deepcopy

import imageio
from PIL import Image
from torchvision import transforms, utils

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from config import get_base_parser
from data import split_train_test
from model.mlp import MLP
from model.conv_mlp import ConvMLP, TransporterMLP
from model.GCN import GCNModel
from utils import set_manual_seed, save_checkpoint, save_config, save_command
from utils import pose_quat_to_rpy, pose_rpy_to_quat, save_video

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------

# reach target task
TASK_NAME = "rt"

TARGET_ENC = np.array([1, 0, 0])
DISTRACT_ENC = np.array([0, 1, 0])
GRIPPER_ENC = np.array([0, 0, 1])

NUM_NODE_FEATURES = 9
NUM_NODES = 4

# input_dim = 24
INPUT_DIM = (3, 128, 128)
# with distractors
# INPUT_DIM = 7 * 3 + 7 * 3

# # for pose output
OUTPUT_DIM = 7
# for joint velocities
# OUTPUT_DIM = 7

TRAIN_RATIO = 0.8

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------


def distance_metric(obs):
    """L2 distance cost."""
    gripper_pos = obs.gripper_pose[:3]
    target_pos = obs.task_low_dim_state[0][:3]
    cost = np.linalg.norm(np.asarray(gripper_pos) - np.asarray(target_pos))
    return cost


def make_env(headless=True, full_obs=False):
    """Constructs the environment"""
    obs_config = ObservationConfig()
    if full_obs:
        obs_config.set_all(True)
    else:
        obs_config.set_all_low_dim(True)

    action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
    # action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

    env = Environment(action_mode, obs_config=obs_config, headless=headless)
    env.launch()

    task = env.get_task(ReachTarget)
    return env, task


def delta_in_pose(pose1, pose2):
    # TODO: Double check this stuff
    # RLbench Pose = X, Y, Z, QX, QY, QZ, QW
    # Quaternions = QW, QX, QY, QZ

    q1x, q1y, q1z, q1w = pose1[3:]
    q2x, q2y, q2z, q2w = pose2[3:]

    q1 = Quaternion(q1w, q1x, q1y, q1z)
    q2 = Quaternion(q2w, q2x, q2y, q2z)

    delta_rot = q2 * q1.inverse

    # Normalize to be unit quaternion
    #delta_rot = delta_rot.unit
    qw, qx, qy, qz = list(delta_rot)

    x, y, z = pose2[:3] - pose1[:3]

    diff = [x, y, z] + [qx, qy, qz, qw]

    return np.array(diff)


# concise state
def obs_to_input(config, obs, use_relative_position=True):
    """Construct input to mlp from raw rlbench obs."""
    features = []
    # features.append(obs.joint_positions)  # (7,)
    # features.append(obs.joint_velocities)  # (7,)
    features.append(obs.gripper_pose)  # (7,) + 1

    features.append(delta_in_pose(obs.gripper_pose, obs.task_low_dim_state[0]))
    feature = np.concatenate(features)  # (14,)
    return feature


# def obs_to_input(obs, add_distractors=True, relative_pos=True):
#     """Construct input to mlp from raw rlbench obs."""
#     features = []
#     features.append(obs.joint_positions)  # (7,)
#     features.append(obs.joint_velocities)  # (7,)
#     features.append(obs.gripper_pose)  # (7,)

#     # target_pos = obs.task_low_dim_state[0][:3]
#     # features.append(target_pos)  # (3,)
#     target_pose = obs.task_low_dim_state[0]
#     if relative_pos:
#         target_pose -= obs.gripper_pose
#     features.append(target_pose)  # (7,)

#     if add_distractors:
#         distractor_pose = obs.task_low_dim_state[1]
#         distractor2_pose = obs.task_low_dim_state[2]
#         if relative_pos:
#             distractor_pose -= obs.gripper_pose
#             distractor2_pose -= obs.gripper_pose
#         features.append(distractor_pose)  # (7,)
#         features.append(distractor2_pose)  # (7,)

#     # minimum (24,)
#     # with distractors (42,)
#     feature = np.concatenate(features)
#     return feature

# -----------------------------------------------------------------------------------
#                   Data
# -----------------------------------------------------------------------------------


def make_image_transform():
    """"""
    return transforms.Compose([
        # transforms.Rescale(64),
        transforms.ToTensor()
    ])


def obs_to_vision_input(config, obs, transform=None):
    """Construct input to mlp from raw rlbench obs."""
    if transform is None:
        transform = transforms.ToTensor()
    img = obs.front_rgb
    img = transform(img)
    return img


# full data loading with images
def load_vision_data(config, data_dir):
    """Data before train-valid split."""
    # get all episodes
    pattern = os.path.join(data_dir, "**/episodes/episode*")
    episode_dirs = glob.glob(pattern, recursive=True)
    num_stack = getattr(config, "num_stack", 1)

    dataset = []
    for d in episode_dirs:
        # load data
        data_path = os.path.join(d, "low_dim_obs.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        buffer = deque(maxlen=num_stack)
        # convert data
        for t in range(len(data._observations) - 1):
            obs = data._observations[t]
            # input is image
            image_path = os.path.join(d, "front_rgb", "{}.png".format(t))
            # depth_path = os.path.join(d, "front_depth", "{}.png".format(t))

            # # repeat for first step or just add to input queue
            # if t == 0:
            #     [buffer.append(deepcopy(feature)) for _ in range(num_stack)]
            # else:
            #     buffer.append(deepcopy(feature))
            # # get stacked feature
            # feature_stack = np.concatenate(list(buffer))

            delta = delta_in_pose(data._observations[t].gripper_pose,
                                  data._observations[t + 1].gripper_pose)
            y = np.asarray(delta)

            # dataset.append([feature_stack, y])
            dataset.append([image_path, y])
    return dataset


class ReachTargetVisionDataset(torch.utils.data.Dataset):

    def __init__(self, data_list=None, transform=None):
        super().__init__()

        self.image_paths = [path for path, _ in data_list]
        self.labels = [y for _, y in data_list]
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image as tensor
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        # load label as tensor
        label = torch.FloatTensor(self.labels[idx])
        return image, label


# -----------------------------------------------------------------------------------
#                   Agent
# -----------------------------------------------------------------------------------


class ConvMLPAgent:

    def __init__(self, config, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = ConvMLP(input_dim,
                             output_dim,
                             hidden_dims=config.hidden_dims)

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

    def act(self, features):
        if len(features.shape) == 3:
            # for testing
            features = features.unsqueeze(0)
        action = self.model(features)
        # return action
        x, y, z, q1x, q1y, q1z, q1w = action[0]
        q1 = Quaternion(q1w, q1x, q1y, q1z)

        # Normalize to be unit quaternion
        q1 = q1.unit
        qw, qx, qy, qz = list(q1)

        return np.asarray([x, y, z, qx, qy, qz, qw])


class TransporterMLPAgent(ConvMLPAgent):

    def __init__(self, config, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = TransporterMLP(input_dim,
                                    output_dim,
                                    hidden_dims=config.hidden_dims)


def make_agent(config):
    """Constructs agent based on config."""
    if config.model_name == "conv_mlp":
        input_dim = INPUT_DIM
        # input_dim *= getattr(config, "num_stack", 1)
        output_dim = OUTPUT_DIM
        agent = ConvMLPAgent(config, input_dim, output_dim)
    elif config.model_name == "trans_mlp":
        input_dim = INPUT_DIM
        # input_dim *= getattr(config, "num_stack", 1)
        output_dim = OUTPUT_DIM
        agent = TransporterMLPAgent(config, input_dim, output_dim)
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
        config.log_dir, "_".join([TASK_NAME, config.model_name, config.tag]),
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
    dataset = load_vision_data(config, config.data_dir)
    dataset_train, dataset_test = split_train_test(dataset,
                                                   train_ratio=TRAIN_RATIO)
    dataset_train = ReachTargetVisionDataset(data_list=dataset_train)
    loader = torch.utils.data.DataLoader(dataset_train,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         num_workers=4)
    dataset_test = ReachTargetVisionDataset(data_list=dataset_test)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config.batch_size)

    # trianing loop
    loss_eval_best = None
    if config.eval_when_train:
        # make env
        env, task = make_env(headless=True, full_obs=True)

    for epoch in range(start_epoch, config.num_epochs):
        agent.train()
        loss_total = 0.0

        for i, (features, labels) in enumerate(loader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = agent.output(features)
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
                out = agent.output(features)
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

        # evaluation rollouts
        if config.eval_when_train and epoch > 0 and epoch % config.eval_when_train_freq == 0:
            results = collect_rollouts(
                config,
                task,
                agent,
                device=device,
                max_episode_length=config.max_episode_length,
                batch_size=config.eval_batch_size,
                render=config.render,
                vid_dir=config.log_dir)

            total_lengths = results["total_lengths"]
            total_rewards = results["total_rewards"]
            total_costs = results["total_costs"]

            summary_writer.add_scalar('loss_eval/total_lengths',
                                      np.mean(total_lengths), epoch)
            summary_writer.add_scalar('loss_eval/total_rewards',
                                      np.mean(total_rewards), epoch)
            summary_writer.add_scalar('loss_eval/total_costs',
                                      np.mean(total_costs), epoch)
            summary_writer.flush()

    # clean up
    if config.eval_when_train:
        env.shutdown()


# -----------------------------------------------------------------------------------
#                   Test
# -----------------------------------------------------------------------------------


def collect_rollouts(config,
                     task,
                     agent,
                     device="cpu",
                     max_episode_length=250,
                     batch_size=5,
                     render=False,
                     vid_dir="archived"):
    """Gets trajectories for evaluation."""
    # setup rendering
    if render:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        gym_cam = VisionSensor.create([640, 360])
        gym_cam.set_pose(cam_placeholder.get_pose())
        # self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
        gym_cam.set_render_mode(RenderMode.OPENGL3)

    # for stacked features
    num_stack = getattr(config, "num_stack", 1)

    # run trajectories
    total_lengths = []
    total_rewards = []
    total_costs = []

    for i in range(batch_size):
        descriptions, obs = task.reset()
        done = False
        length = 0
        total_r = 0
        total_cost = distance_metric(obs)
        frames = []
        buffer = deque(maxlen=num_stack)

        while not done and length < max_episode_length:
            feature = obs_to_vision_input(config, obs).to(device)

            # # for feature stack
            # if length == 0:
            #     [buffer.append(deepcopy(feature)) for _ in range(num_stack)]
            # else:
            #     buffer.append(deepcopy(feature))
            # feature = np.concatenate(list(buffer))

            # feature = torch.FloatTensor(feature).to(device)
            with torch.no_grad():
                action = agent.act(feature)

            # add gripper action
            gripper = [1.0]  # Always open
            full_action = np.concatenate([action, gripper], axis=-1)

            try:
                obs, reward, done = task.step(full_action)
                # print("step: {}, returns: {}".format(length, total_r))
                total_r += reward
                length += 1
                total_cost += distance_metric(obs)

                if render:
                    img = gym_cam.capture_rgb()
                    frames.append(img)
            except:
                break

        total_lengths.append(length)
        total_rewards.append(total_r)
        total_costs.append(total_cost)
        if render:
            vid_name = os.path.join(vid_dir, "run_{}.gif".format(i))
            frames = [f * 255 for f in frames]
            save_video(vid_name, frames, fps=20)

    # log and clean up
    total_lengths = np.asarray(total_lengths)
    total_rewards = np.asarray(total_rewards)
    total_costs = np.asarray(total_costs)

    results = {
        "total_lengths": total_lengths,
        "total_rewards": total_rewards,
        "total_costs": total_costs
    }
    return results


def test_policy(config):
    """Evaluates trained policy."""
    # make env
    env, task = make_env(headless=False, full_obs=True)

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
    # checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pth")
    checkpoint = torch.load(checkpoint_path)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()

    # evaluate
    results = collect_rollouts(old_config,
                               task,
                               agent,
                               max_episode_length=config.max_episode_length,
                               batch_size=config.eval_batch_size,
                               render=config.render,
                               vid_dir=config.checkpoint_dir)

    total_lengths = results["total_lengths"]
    total_rewards = results["total_rewards"]
    total_costs = results["total_costs"]

    print("Eval traj lengths: {} +/- {}".format(total_lengths.mean(),
                                                total_lengths.std()))
    print("Eval traj returns: {} +/- {}".format(total_rewards.mean(),
                                                total_rewards.std()))
    print("Eval traj costs: {} +/- {}".format(total_costs.mean(),
                                              total_costs.std()))

    print('Done')
    env.shutdown()


# -----------------------------------------------------------------------------------
#                   Execution
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_base_parser()
    config = parser.parse_args()

    # function to run
    func = train
    if config.eval:
        func = test_policy

    # run
    func(config)
