from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
import math
from tools import *


class SkylineDataset(Dataset):

    def __init__(self, root, seed=None, transform=None):
        self.root_path = root
        self.transform = transform
        self.seed = seed
        self.filenames = os.listdir(self.root_path)

        if seed is not None:
            random.seed(seed)
        print('CureDataset_init.')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file = read_data_row(os.path.join(self.root_path, filename), 1).strip().split(" ")
        line1 = np.array(list(map(int, file[0].split(','))))  # C*H*W = (1, length, 1)

        length = len(line1)
        if self.transform:
            line1 = Move()(line1)
            line1 = Rotate()(line1)
        line2 = np.array(list(map(int, file[1].split(','))))

        # Normalization
        line = np.hstack((line1, line2))
        line_min, line_max = line.min(), line.max()
        line = (line - line_min) / (line_max - line_min)

        line1 = line[:length].reshape(1, length, 1)
        line2 = line[length:].reshape(1, length, 1)


        label = np.array(list(map(int, file[2])))
        sample = {"line": [line1, line2], "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.filenames)


class Rotate(object):
    def __init__(self, angle=6):
        self.angle = angle

    def __call__(self, line):
        angle = random.randint(-1*self.angle, self.angle)
        pointx = len(line) // 2
        pointy = line[pointx]
        angle = float(angle) * 3.1415 / float(180)
        x = np.arange(len(line))
        y = (x - pointx) * math.sin(angle) + (line - pointy) * math.cos(angle) + pointy

        return y


class Move(object):
    def __init__(self, mving_step=100):
        self.mving_step = mving_step

    def __call__(self, line):
        delta = random.randint(-1*self.mving_step, self.mving_step)
        return line + delta


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        line, lable = sample['line'], sample['label']
        return {'line': [torch.from_numpy(line[0]).float(), torch.from_numpy(line[1]).float()],
                'label': torch.from_numpy(lable).float()}
