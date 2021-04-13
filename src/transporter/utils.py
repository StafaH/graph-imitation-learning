import random
import numpy as np
import torch
import os

# To create reproducible results, set all seeds across all RNG manually,
# https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
# when using additional workers, those also need to set their seeds.


def set_manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

    # TODO:(mustafa) This should be True for reproducilbility, but letting it benchmark increases
    # speed dramatically, so swap this for final code push
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def save_checkpoint(log_dir, model_name, checkpoint_dict):
    file_path = os.path.join(log_dir, model_name + '.pth')
    torch.save(checkpoint_dict, file_path)
