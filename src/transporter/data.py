import os
import glob
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, Sampler


class RLBenchTransporterDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        # self.episode_folders = [os.listdir(
        #     os.path.join(self.root, 'variation0/episodes/episode{}/front_rgb/'.format(i)))
        #     for i in range(1, num_episodes)]
        pattern = os.path.join(root, "**/episodes/episode*/front_rgb/*.png")
        self.image_files = glob.glob(pattern, recursive=True)
        self.length = len(self.image_files)

    def __len__(self):
        return self.length

    def get_image(self, n):
        image = Image.open(self.image_files[n])
        return image

    def __getitem__(self, idx):
        # Randomly choose a another frame, as all episodes have the same 
        # objects, we can choose anything
        target_index = idx - np.random.randint(1, self.length)

        source_image = Image.open(self.image_files[idx])
        target_image = Image.open(self.image_files[target_index])
        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image


# Old Implementation of using a random index sampler + dataset (created an infinite sampler
# so it was impossible to use in a cluster, and the dataset could not be easily indexed e.g. data[i])
class TransporterRandomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform

    def __len__(self):
        images = glob.glob('{}*.jpg'.format(self.root_dir))
        return(int(len(images)))

    def get_image(self, n):
        image = Image.open('{}{}l.jpg'.format(self.root_dir, n))
        return image

    def __getitem__(self, idx):
        source_index, target_index = idx

        # Sample from the left and right stereocameras randomly
        lr = np.random.choice(['l', 'r'])
        source_image = Image.open(f'{self.root_dir}{source_index}{lr}.jpg')
        target_image = Image.open(f'{self.root_dir}{target_index}{lr}.jpg')
        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image


class TransporterRandomSampler(Sampler):
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
