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
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from model.mlp import MLP 
from data import ProcessStateToGraphData
from utils import set_manual_seed, save_checkpoint
from config import load_default_config


def _init_fn(worker_id):
    np.random.seed(int(53))


def main(args):
    print("----------------------------------------")
    print("Configuring")
    print("----------------------------------------\n")

    # Manual seeds for reproducible results
    set_manual_seed(args.seed)
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

    print('Current device is set to: ', device)

    if(not os.path.exists(config.data_dir)):
        print('The data directory does not exist:', config.data_dir)
        return

    dataset = []

    # One-hot encodings
    target_enc = np.asarray([1, 0, 0])
    distract_enc = np.asarray([0, 1, 0])
    gripper_enc = np.asarray([0, 0, 1])

    for i in range(10):
        for j in range(10):
            data = ProcessStateToGraphData(f'{config.data_dir}reach/state_data_{i}_{j}.npy')

            for k in range(len(data) - 1):
                target_node = np.concatenate((data[k][0], target_enc))
                distract_node = np.concatenate((data[k][1], distract_enc))
                distract2_node = np.concatenate((data[k][2], distract_enc))
                gripper_node = np.concatenate((data[k][3], gripper_enc))

                nodes = torch.tensor([target_node, distract_node, distract2_node, gripper_node], dtype=torch.float)
                edge_index = torch.tensor([[0, 1],
                                           [1, 0],
                                           [0, 2],
                                           [2, 0],
                                           [0, 3],
                                           [3, 0],
                                           [1, 2],
                                           [2, 1],
                                           [1, 3],
                                           [3, 1],
                                           [2, 3],
                                           [3, 2]], dtype=torch.long)
                y = torch.tensor([data[k + 1][3]], dtype=torch.float)
                graph_data = graph_data = Data(x=nodes, edge_index=edge_index.t().contiguous(), y=y)
                dataset.append(graph_data)
    
    #dataset = GraphDataset(config.data_dir, transform=transformer)
    #print("Images loaded from data directory: ", len(dataset))

    # Load the dataset (num_workers is async, set to 0 if using notebooks)
    loader = DataLoader(dataset, batch_size=config.batch_size)
                                         #pin_memory=True, num_workers=4,
                                         #worker_init_fn=_init_fn)

    # Build Model
    input_dim = dataset[0].num_node_features * dataset[0].num_nodes
    output_dim = 3 
    model = MLP(
        input_dim,
        output_dim,
        hidden_dims=args.hiddens,
        act="relu",
        output_act=None,
        init_weights=True
    )
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
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
    config.log_dir = os.path.join(
        config.log_dir, 
        args.tag + '_' + "seed{}".format(args.seed) + '_' + current_time
    )

    os.makedirs(config.log_dir, exist_ok=True)
    print('Logs are being written to {}'.format(config.log_dir))
    print('Use TensorBoard to look at progress!')
    summary_writer = SummaryWriter(config.log_dir)

    # Dump training information in log folder (for future reference!)
    with open(config.log_dir + "/info.txt", "w") as text_file:
        print(f"Data directory: {config.data_dir}", file=text_file)
        print(f"Epochs: {config.num_epochs}", file=text_file)
        print(f"Batch Size: {config.batch_size}", file=text_file)

    print("----------------------------------------")
    print("Training Transporter Network")
    print("----------------------------------------\n")

    model.train()
    pbar = tqdm(total=config.num_epochs)
    pbar.n = start_epoch
    pbar.refresh()
    for epoch in range(start_epoch, config.num_epochs):
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

        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} - Loss - {loss:.5f}')

        save_checkpoint(config.log_dir, config.model_name, {'epoch': epoch,
                                                            'model_state_dict': model.state_dict(),
                                                            'optimizer_state_dict': optimizer.state_dict(),
                                                            'loss': loss})

        summary_writer.add_scalar('loss', loss, epoch)
        summary_writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Imitation Learning')
    parser.add_argument('-r', '--resume', type=str, default='', help='path to last checkpoint (default = None)')
    parser.add_argument('--tag', type=str, default='exp', help='experiment id')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument("--hiddens", nargs='+', type=int, default=[64, 64], 
                            help="mlp hidden layer dimensions")
    args = parser.parse_args()
    main(args)
