#!/usr/bin/env python
"""train.py: Train Graph Neural Network for Imitation Learning.

    Arguments:
        Required:

        Optional:

    Usage: train.py [-h] -d DATA -l LOG -k KEYPOINTS [-c CHANNELS] [-e EPOCHS] [-n NAME] [-b BATCH]
    Usage:
    Example:
"""
import sys
import os
import glob
import argparse
from datetime import datetime

from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data, DataLoader

from config import get_base_parser
from data import load_data_to_graph, split_train_test
from model.mlp import MLP
from utils import set_manual_seed, save_checkpoint, save_config, save_command

# -----------------------------------------------------------------------------------
#                   Funcs
# -----------------------------------------------------------------------------------

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

    print('Current device is set to: ', device)

    if (not os.path.exists(config.data_dir)):
        print('The data directory does not exist:', config.data_dir)
        return

    dataset = load_data_to_graph(config.data_dir)
    dataset_train, dataset_test = split_train_test(dataset)

    # Load the dataset (num_workers is async, set to 0 if using notebooks)
    loader = DataLoader(dataset_train, batch_size=config.batch_size)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size)

    # Build Model
    input_dim = dataset[0].num_node_features * dataset[0].num_nodes
    output_dim = 3
    model = MLP(input_dim,
                output_dim,
                hidden_dims=config.hidden_dims,
                act="relu",
                output_act=None,
                init_weights=True)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    start_epoch = 0

    # Load a model if resuming training
    if config.resume != '':
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
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

    model.train()
    pbar = tqdm(total=config.num_epochs)
    pbar.n = start_epoch
    pbar.refresh()
    for epoch in range(start_epoch, config.num_epochs):
        loss_total = 0.0

        for i, data in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            # concatenate all graph nodes to single vector,
            # assume all graphs have same number of nodes and and in order
            # assumes nodes from same graph are contiguous in batch
            flat_x = data.x.reshape(-1, input_dim)
            out = model(flat_x)
            loss = torch.nn.functional.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total
            })

        # evaluation
        if epoch > 0 and epoch % config.eval_interval == 0:
            loss_eval_total = 0.0

            for data in loader_test:
                data = data.to(device)
                flat_x = data.x.reshape(-1, input_dim)
                with torch.no_grad():
                    out = model(flat_x)
                loss_eval = torch.nn.functional.mse_loss(out, data.y)
                loss_eval_total += loss_eval.item()

            loss_eval_total /= len(loader_test)
            summary_writer.add_scalar('loss_eval', loss_eval_total, epoch)
            summary_writer.flush()


# -----------------------------------------------------------------------------------
#                   Configs
# -----------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = get_base_parser()
    config = parser.parse_args()
    main(config)
