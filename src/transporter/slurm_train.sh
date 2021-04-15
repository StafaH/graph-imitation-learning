#!/bin/bash
#SBATCH --nodes 2
#SBATCH --gres=gpu:2

#SBATCH --tasks-per-node=2

#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision pytorch-lightning --no-index

srun python pytorch-ddp-test-pl.py  --batch_size 256