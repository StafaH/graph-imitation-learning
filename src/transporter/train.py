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
import sys
import os
import glob
import argparse
from datetime import datetime

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Sampler
#from torchsummary import summary

from model.transporter import FeatureExtractor, KeyNet, RefineNet, Transporter
from utils import set_manual_seed, save_checkpoint
from config import load_default_config
from data import RLBenchTransporterDataset


def _init_fn(worker_id):
    np.random.seed(int(53))


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

    dataset = RLBenchTransporterDataset(config.data_dir, transform=transformer)
    print("Images loaded from data directory: ", len(dataset))

    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                         pin_memory=True, num_workers=4, worker_init_fn=_init_fn,
                                         shuffle=True)

    # Build Transporter Model architecture
    feature_encoder = FeatureExtractor(config.num_channels)
    key_net = KeyNet(config.num_channels, config.num_keypoints)
    refine_net = RefineNet(config.num_channels)

    transporter_model = Transporter(feature_encoder, key_net, refine_net)
    transporter_model.to(device=device)

    # Optional: Print summary of model
    # summary(transporter_model, [(3, 128, 128), (3, 128, 128)])

    optimizer = torch.optim.Adam(transporter_model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(25000), gamma=0.95)

    # Gradient Scaling to prevent small magnitudes from "flusing" to zero from AMP
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    # Load a model if resuming training
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        transporter_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Resuming Training from Epoch: {start_epoch} at Loss:{loss}')

    # Create a log directory using the current timestamp
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    config.log_dir = os.path.join(config.log_dir, 'transporter' + '_' + current_time)

    os.makedirs(config.log_dir, exist_ok=True)
    print('Logs are being written to {}'.format(config.log_dir))
    print('Use TensorBoard to look at progress!')
    summary_writer = SummaryWriter(config.log_dir)

    # Dump training information in log folder (for future reference!)
    with open(config.log_dir + "/info.txt", "w") as text_file:
        print(f"Data directory: {config.data_dir}", file=text_file)
        print(f"Channels: {config.num_channels}", file=text_file)
        print(f"Keypoints: {config.num_keypoints}", file=text_file)
        print(f"Epochs: {config.num_epochs}", file=text_file)
        print(f"Batch Size: {config.batch_size}", file=text_file)

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    pbar = tqdm(total=config.num_epochs)
    pbar.n = start_epoch
    pbar.refresh()
    transporter_model.train()

    for epoch in range(start_epoch, config.num_epochs):
        for i, (source, target) in enumerate(loader):
            source = source.to(device)
            target = target.to(device)

            # Mixed Precision
            with torch.cuda.amp.autocast():
                reconstruction = transporter_model(source, target)
                loss = torch.nn.functional.mse_loss(reconstruction, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss:.5f}')

        save_checkpoint(config.log_dir, config.model_name, {'epoch': epoch,
                                                            'model_state_dict': transporter_model.state_dict(),
                                                            'optimizer_state_dict': optimizer.state_dict(),
                                                            'scaler': scaler.state_dict(),
                                                            'loss': loss})

        summary_writer.add_scalar('reconstruction_loss', loss, epoch)

        # Due to large file size of image grids, only save images every 100 epochs
        if epoch % 100 == 0:
            reconst_grid = torchvision.utils.make_grid(reconstruction)
            source_grid = torchvision.utils.make_grid(source)
            target_grid = torchvision.utils.make_grid(target)
            summary_writer.add_image('source', source_grid, epoch)
            summary_writer.add_image('target', target_grid, epoch)
            summary_writer.add_image('reconst_target', reconst_grid, epoch)

        summary_writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unsupervised Keypoints Learning from video frames')
    parser.add_argument('-r', '--resume', type=str, default='', help='path to last checkpoint (default = None)')
    args = parser.parse_args()
    main(args)
