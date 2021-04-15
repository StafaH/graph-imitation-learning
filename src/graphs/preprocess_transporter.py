#!/usr/bin/env python
"""train_graph.py: Train Graph Neural Network for Imitation Learning.

    Arguments:
           [-h] [--tag TAG] [--seed SEED] 
           [-r RESUME]
           [--data_dir DATA_DIR] 
           [--log_dir LOG_DIR] 
           [--eval]
           [--max_episode_length MAX_EPISODE_LENGTH]
           [--eval_interval EVAL_INTERVAL]
           [--eval_batch_size EVAL_BATCH_SIZE]
           [--checkpoint_dir CHECKPOINT_DIR]
           [--model_name MODEL_NAME] 
           [--num_epochs NUM_EPOCHS]
           [--batch_size BATCH_SIZE]
           [--hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...] ]
           
    Example: ```python src/graphs/train_graph.py --data_dir data/reach_target/ --model_name graph --batch_size 64 --num_epochs 1```

"""

# Imports
import sys
import os
import glob
import argparse

from PIL import Image
import numpy as np
import torch

from config import get_graph_parser
from data import delta_in_pose
from utils import set_manual_seed, save_checkpoint, save_config, save_command


# -----------------------------------------------------------------------------------
#                   Main
# -----------------------------------------------------------------------------------


def main(config):
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

    print("Current device is set to: ", device)

    transformer = transforms.Compose([
                                     # transforms.Resize((128, 128)), Already 128
                                     transforms.ToTensor()
                                     ])

    if(not os.path.exists(config.data_dir)):
        print("The data directory does not exist:", config.data_dir)
        return

    # Build Transporter Model architecture
    feature_encoder = FeatureExtractor(config.num_channels)
    key_net = KeyNet(config.num_channels, config.num_keypoints)
    refine_net = RefineNet(config.num_channels)

    transporter_model = Transporter(feature_encoder, key_net, refine_net)
    transporter_model.to(device=device)
    
    checkpoint = torch.load(args.resume)
    transporter_model.load_state_dict(checkpoint['model_state_dict'])
    transporter_model.eval()
    print(f'Transporter model loaded!')

    pattern = os.path.join(data_dir, "**/episode*/")
    episode_folders = glob.glob(pattern)

    print(f'Found {len(episode_folders)} episode folders found')
    
    # Construct new dataset from state data and transporter keypoints + features

    for f_path in episode_files:
        state_data = np.load(f_path)
        new_dataset = []
        for k in range(len(state_data) - 1):
            # nodes
            NUM_NODES = 4

            gripper_node = np.concatenate([state_data[k][3], GRIPPER_ENC])

            if use_relative_position:
                target_node = np.concatenate([
                    delta_in_pose(state_data[k][3], state_data[k][4]),
                    TARGET_ENC
                ])
                distract_node = np.concatenate([
                    delta_in_pose(state_data[k][3], state_data[k][5]),
                    DISTRACT_ENC
                ])
                distract2_node = np.concatenate([
                    delta_in_pose(state_data[k][3], state_data[k][6]),
                    DISTRACT_ENC
                ])
            else:
                target_node = np.concatenate([
                    state_data[k][3], state_data[k][4],
                    TARGET_ENC
                ])
                distract_node = np.concatenate([
                    state_data[k][3], state_data[k][5],
                    DISTRACT_ENC
                ])
                distract2_node = np.concatenate([
                    state_data[k][3], state_data[k][6],
                    DISTRACT_ENC
                ])

            # TODO: Transporter
            transporter_model.point_net(image)
            
            delta = delta_in_pose(state_data[k][3], state_data[k + 1][3])

            new_dataset.append(graph_data)
        
        print(f'Episode X processed!')


if __name__ == '__main__':
    parser = get_graph_parser()
    config = parser.parse_args()
    main(config)
