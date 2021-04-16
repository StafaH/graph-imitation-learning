# Final Course Project for CSC2626: Imitation Learning for Robotics

## Graph Imitation Learning

The key codebase is in **src/graphs**, a list of available arguments can be found in **src/graphs/config.py**.

## Installation

We recommend using Anaconda to install all relevant packages:

1. Create a new environment (Python 3.8 or older only)

```bash
conda create -n graphs python=3.8.8
```

2. Activate the environment:

```bash
conda activate graphs
```

3. Install PyTorch from https://pytorch.org/

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c=conda-forge
```

4. Install PyTorch Geometric https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html (Make sure you are running the anaconda prompt as administrator)

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

```

5. Install the rest of the packages

```bash
pip install -r requirements.txt
```

## Training

<!-- [0]: joint position
[1]: joint velocity
[2]: Gripper open/close
[3]: Gripper pose -->

To train MLP baseline for Reach-Target, with imitation dataset in **data/reach_target**, use

```bash
python src/graphs/reach_target_mlp.py --data_dir data/reach_target  --model_name mlp --hidden_dims 64 64 64 --num_epochs 3000 --lr 0.0001 --eval_when_train --tag 64x3_lr --seed 9
```

To train MLP baseline for Pick-and-Lift, with imitation dataset in **data/pick_and_lift**, use

```bash
python src/graphs/pick_and_lift_mlp.py --data_dir data/pick_and_lift  --model_name mlp --hidden_dims 64 64 64 --num_epochs 3000 --lr 0.0001 --eval_when_train --tag 64x3_lr --seed 9
```

## Evaluation

To generate evaluation rollouts for a given checkpoint, for example, **rt_mlp_64x3_lr/seed6_Apr15_14-04-30**, use

```bash
python src/graphs/reach_target_mlp.py --eval --eval_batch_size 10 --max_episode_length 250 --checkpoint_dir logs/rt_mlp_64x3_lr/seed6_Apr15_14-04-30/ --seed 66 --render
```

## Visualization

1. To visualize the resulting model's loss in Tensorboard, run the following command, pointing to the `logs` directory where the experiments are saved:
   `tensorboard --logdir=logs`

2. Then, go to **localhost:6006** in your browser to view Tensorboard.
