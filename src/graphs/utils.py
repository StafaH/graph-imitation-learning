import random
import numpy as np
import os
import sys
import torch
import yaml

# To create reproducible results, set all seeds across all RNG manually,
# https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
# when using additional workers, those also need to set their seeds.


def set_manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(log_dir, model_name, checkpoint_dict):
    file_path = os.path.join(log_dir, model_name + '.pth')
    torch.save(checkpoint_dict, file_path)


def save_config(config, output_dir):
    """Logs configs to file under directory."""
    config_dict = config.__dict__
    file_path = os.path.join(output_dir, "config.yaml")
    with open(file_path, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False)


def save_command(output_dir):
    """Logs current executing command to text file."""
    with open(os.path.join(output_dir, 'cmd.txt'), 'a') as file:
        file.write(" ".join(sys.argv) + "\n")


def euler_angle_to_quaternion(rpy):
    """Converts roll, pitch, yaw euler angles to quaternion.
    Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    pyrep uses quat differently: https://pyrep.readthedocs.io/en/latest/pyrep.objects.html#pyrep.objects.object.Object.get_quaternion
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = np.array([x, y, z, w]).squeeze()
    return quat


def quaternion_to_euler_angle(quat):
    """Converts quaternion to euler angles
    Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles 
    pyrep uses quat differently: https://pyrep.readthedocs.io/en/latest/pyrep.objects.html#pyrep.objects.object.Object.get_quaternion
    """
    # w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    roll = float(roll)

    # pitch (y-axis rotation)
    sinp = float(2 * (w * y - z * x))
    if sinp > 1:
        pitch = np.pi / 2.0
    elif sinp < -1:
        pitch = -np.pi / 2.0
    else:
        pitch = np.arcsin(sinp)
        pitch = float(pitch)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    yaw = float(yaw)

    rpy = np.array([roll, pitch, yaw]).squeeze()
    return rpy


def pose_quat_to_rpy(pose):
    """Replaces quaterion in pose to euler angles
    pose: (7,), angles are normalized by np.pi 
    """
    pos, quat = pose[:3], pose[3:]
    rpy = quaternion_to_euler_angle(quat) / np.pi
    new_pose = np.concatenate([pos, rpy])
    return new_pose  # (6,)


def pose_rpy_to_quat(pose):
    """Replaces euler angles in pose to quaternion
    pose: (6,), angles are scaled back up by np.pi
    """
    pos, rpy = pose[:3], pose[3:]
    quat = euler_angle_to_quaternion(rpy * np.pi)
    new_pos = np.concatenate([pos, quat])
    return new_pos  # (7,)
