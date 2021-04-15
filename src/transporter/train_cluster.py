#!/usr/bin/env python

"""train.py: Train transporter network for learning unsupervised keypoints.

    Arguments:
        Required:

        Optional:

    Usage: train.py [-h] -d DATA -l LOG -k KEYPOINTS [-c CHANNELS] [-e EPOCHS] [-n NAME] [-b BATCH]
    Usage:
    Example:
"""

# Imports
import os
import argparse
import numpy as np

import torch
from torchvision import transforms
import pytorch_lightning as pl

from model.transporter import LitTransporter
from utils import set_manual_seed
from config import load_default_config
from data import TransporterDataset


def _init_fn(worker_id):
    np.random.seed(int(52))


def main(args):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Manual seeds for reproducible results
    set_manual_seed(52)
    config = load_default_config()
    print(config)
    print(args)

    print("----------------------------------------")
    print("Loading Model and Data")
    print("----------------------------------------\n")

    transformer = transforms.Compose([
                                     transforms.Resize((128, 128)),
                                     transforms.ToTensor()
                                     ])

    if(not os.path.exists(config.data_dir)):
        print("The data directory does not exist:", config.data_dir)
        return

    dataset = TransporterDataset(config.data_dir, transform=transformer)
    print("Images loaded from data directory: ", len(dataset))

    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                         pin_memory=True, num_workers=4,
                                         worker_init_fn=_init_fn, shuffle=False)

    # Build Transporter Model and train using Pytorch Lightning
    transporter_model = LitTransporter(config.num_channels, config.num_keypoints)

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    trainer = pl.Trainer(max_epochs=config.num_epochs, gpus=torch.cuda.device_count(),
                         num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES")),
                         accelerator='ddp')
    trainer.fit(transporter_model, loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unsupervised Keypoints Learning from video frames')
    parser.add_argument('-r', '--resume', type=str, default='', help='path to last checkpoint (default = None)')
    args = parser.parse_args()
    main(args)
