# Final Course Project for CSC2626: Imitation Learning for Robotics

## Graph Imitation Learning

### Installation Procedure:

We recommend using Anaconda to install all relevant packages:

1. Create a new environment (Python 3.8 or older only)

``` conda create -n graphs python=3.8.8 ```

2. Activate the environment:

```conda activate graphs```

3. Install PyTorch from https://pytorch.org/

``` conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c=conda-forge ```

4. Install PyTorch Geometric https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html (Make sure you are running the anaconda prompt as administrator)

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

```

5. Install the rest of the packages

``` conda install numpy tensorboard```
