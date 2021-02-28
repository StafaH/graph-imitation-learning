#!/usr/bin/env python

"""train.py: Train Graph Neural Network for Imitation Learning.

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
from torch.utils.data import Dataset
from torchsummary import summary

from model import GraphModel
from utils import set_manual_seed, save_checkpoint
from config import load_default_config


def _init_fn(worker_id):
    np.random.seed(int(53))


class GraphDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform

    def __len__(self):
        images = glob.glob('{}*.jpg'.format(self.root_dir))
        return(int(len(images)))

    def get_image(self, n):
        image = Image.open('{}{}.jpg'.format(self.root_dir, n))
        return image

    def __getitem__(self, idx):
        image = Image.open(f'{self.root_dir}{idx}.jpg')
        if self.transform is not None:
            image = self.transform(image)

        return image


def main(args):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Manual seeds for reproducible results
    set_manual_seed(53)
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
                                     transforms.Resize((128, 128)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])

    if(not os.path.exists(config.data_dir)):
        print("The data directory does not exist:", config.data_dir)
        return

    dataset = GraphDataset(config.data_dir, transform=transformer)
    print("Images loaded from data directory: ", len(dataset))

    # Load the dataset (num_workers is async, set to 0 if using notebooks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                         pin_memory=True, num_workers=4,
                                         worker_init_fn=_init_fn)

    # Build Model

    model = GraphModel()
    model.to(device=device)

    # Optional: Print summary of model
    # summary(transporter_model, [(3, 128, 128), (3, 128, 128)])

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(25000), gamma=0.95)
    start_epoch = 0

    # Load a model if resuming training
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    for epoch in range(start_epoch, config.num_epochs):
        for i, (source, target) in enumerate(loader):
            model.train()
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(source, target)
            loss = torch.nn.functional.mse_loss(reconstruction, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            # The sampler will cause enumerate(loader) to loop infinitely,
            # so we must break manually every epoch
            if i % len(dataset) == 0:
                break

        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss:.5f}')

        save_checkpoint(config.log_dir, config.model_name, {'epoch': epoch,
                                                            'model_state_dict': model.state_dict(),
                                                            'optimizer_state_dict': optimizer.state_dict(),
                                                            'loss': loss})

        summary_writer.add_scalar('reconstruction_loss', loss, epoch)

        # Due to large file size of image grids, only save images every 1000 epochs
        if epoch % 1000 == 0:
            reconst_grid = torchvision.utils.make_grid(reconstruction)
            MEAN = torch.tensor([0.5, 0.5, 0.5], device=device)
            STD = torch.tensor([0.5, 0.5, 0.5], device=device)
            source = source * STD[:, None, None] + MEAN[:, None, None]
            target = target * STD[:, None, None] + MEAN[:, None, None]
            source_grid = torchvision.utils.make_grid(source)
            target_grid = torchvision.utils.make_grid(target)
            summary_writer.add_image('source', source_grid, epoch)
            summary_writer.add_image('target', target_grid, epoch)
            summary_writer.add_image('reconst_target', reconst_grid, epoch)

        summary_writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Imitation Learning')
    parser.add_argument('-r', '--resume', type=str, default='', help='path to last checkpoint (default = None)')
    args = parser.parse_args()
    main(args)
