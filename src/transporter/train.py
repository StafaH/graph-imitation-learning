#!/usr/bin/env python

"""train.py: Train transporter network for learning unsupervised keypoints.

    Arguments:
        Required:

        Optional:

    Usage: train.py [-h] -d DATA -l LOG -k KEYPOINTS [-c CHANNELS] [-e EPOCHS] [-n NAME] [-b BATCH]
    Usage: 
    Example:
"""

#Imports 
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

sys.path.append('src/transporter/model/')
from model.transporter import *


class TransporterDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform

    def __len__(self):
        images = glob.glob('{}*.jpg'.format(self.root_dir))
        return(int(len(images)))

    def get_image(self, n):
        image = np.array(Image.open('{}{}l.jpg'.format(self.root_dir, n)))
        return image

    def __getitem__(self, idx):
        source_index, target_index = idx

        # Sample from the left and right stereocameras randomly
        lr = np.random.choice(['l', 'r'])
        source_image = np.array(Image.open(f'{self.root_dir}{source_index}{lr}.jpg'))
        target_image = np.array(Image.open(f'{self.root_dir}{target_index}{lr}.jpg'))
        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image


class TransporterDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            # Half the images have the same index (left + right)
            num_images = len(self.dataset) / 2

            # Randomly choose an image, and then choose a target frame within 20 frames
            # Original source must be all images - 20 otherwise the second random integer might 
            # pass the total number of images and index out of the array!
            source_index = np.random.randint(1, num_images - 20)
            target_index = source_index + np.random.randint(20)
            yield source_index, target_index

    def __len__(self):
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Train Unsupervised Keypoints Learning from video frames')
    parser.add_argument('-d', '--data', type=str, required=True, help='location of the dataset folder')
    parser.add_argument('-l', '--log', type=str, required=True, help='output folder for log information and trained model')
    parser.add_argument('-k', '--keypoints', type=int, required=True, help='number of keypoints')
    parser.add_argument('-c', '--channels', type=int, default=3, help='number of channels')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs to train - 1 epoch equals one loop through the entire dataset')
    parser.add_argument('-n', '--name', type=str, default='transporter_model', help='filename of trained model')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-s', '--saveload', type=str, default='', help='Load a previous model to resume training')
    args = parser.parse_args()

    data_dir = args.data
    log_dir = args.log
    num_channels = args.channels
    num_keypoints = args.keypoints
    num_epochs = args.epochs
    model_file_name = args.name
    batch_size = args.batch
    load_file = args.saveload

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Current device is set to: ", device)

    transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    if(not os.path.exists(data_dir)):
        print("The data directory does not exist:", data_dir)
        return

    dataset = TransporterDataset(data_dir, transform=transformer)
    print("Images loaded from data directory: ", len(dataset))
    
    sampler = TransporterDatasetSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=4)

    # Build Transporter Model architecture
    feature_encoder = FeatureEncoder(num_channels)
    pose_regressor = PoseRegressor(num_channels, num_keypoints)
    refine_net = RefineNet(num_channels)

    transporter_model = Transporter(feature_encoder, pose_regressor, refine_net)
    transporter_model.to(device=device)

    optimizer = torch.optim.Adam(transporter_model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(25000), gamma=0.95)

    # Load a model if resuming training
    if load_file != '':
        transporter_model.load_state_dict(torch.load(load_file))

    # Create a log directory using the current timestamp
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(log_dir, 'transporter' + '_' + current_time)

    os.makedirs(log_dir, exist_ok=True)
    print('Logs are being written to {}'.format(log_dir))
    print('Use TensorBoard to look at progress!')
    summary_writer = SummaryWriter(log_dir)

    # Dump training information in log folder (for future reference!)
    with open(log_dir + "/info.txt", "w") as text_file:
        print(f"Data directory: {data_dir}", file=text_file)
        print(f"Channels: {num_channels}", file=text_file)
        print(f"Keypoints: {num_keypoints}", file=text_file)
        print(f"Epochs: {num_epochs}", file=text_file)
        print(f"Batch Size: {batch_size}", file=text_file)

    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        for i, (source, target) in enumerate(loader):
            transporter_model.train()
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            reconstruction = transporter_model(source, target)
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

        torch.save(transporter_model.state_dict(), os.path.join(log_dir, model_file_name + '.pth'))
        summary_writer.add_scalar('reconstruction_loss', loss, epoch)

        # Due to large file size of image grids, only save images every 20 epochs
        if epoch % 20 == 0:
            reconst_grid = torchvision.utils.make_grid(reconstruction)
            source_grid = torchvision.utils.make_grid(source)
            target_grid = torchvision.utils.make_grid(target)
            summary_writer.add_image('source', source_grid, epoch)
            summary_writer.add_image('target', target_grid, epoch)
            summary_writer.add_image('reconst_target', reconst_grid, epoch)

        summary_writer.flush()


if __name__ == '__main__':
    main()