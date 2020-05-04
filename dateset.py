# -*- coding: utf-8 -*-
# author: Dabin Cheng time: 2020/3/5

from __future__ import print_function, division
import os
import cv2
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        self.means = np.array((r, g, b))

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        new_video_x = video_x - self.means
        return {'video_x': new_video_x, 'video_label': video_label}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(182, 242)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        h, w = video_x.shape[1], video_x[2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x = np.zeros((16, new_h, new_w, 3))
        for i in range(16):
            image = video_x[i, :, :, :]
            img = transform.resize(image, (new_h, new_w))
            new_video_x[i, :, :, :] = img

        return {'video_x': new_video_x, 'video_label': video_label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(128, 128)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        h, w = video_x.shape[1], video_x.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video_x = np.zeros((16, new_h, new_w, 3))
        for i in range(16):
            image = video_x[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video_x': new_video_x, 'video_label': video_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        video_label = [video_label]
        return {'video_x': torch.from_numpy(video_x), 'video_label': torch.FloatTensor(video_label)}


class UCF101(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, info_list, root_dir, transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    # get (16,240,320,3)
    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        video_label = self.landmarks_frame.iloc[idx, 1]
        video_x = self.get_single_video_x(video_path)
        sample = {'video_x': video_x, 'video_label': video_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path):

        video_x = np.zeros((16, 240, 320, 3))
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()

        if success:
            for i in range(16):
                video_x[i, :, :, :] = frame
                success, frame = cap.read()
        else:
            print('get single video erro !')
        return video_x


if __name__ == '__main__':
    # usage
    root_list = '/Users/dabincheng/downloads/UCF-101/'
    info_list = '/Users/dabincheng/downloads/ucfTrainTestlist/trainlist01.txt'
    myUCF101 = UCF101(info_list, root_list,
                      transform=transforms.Compose([ClipSubstractMean(), Rescale(), RandomCrop(), ToTensor(), ]))

    dataloader = DataLoader(myUCF101, batch_size=4, shuffle=True, num_workers=1)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     if i_batch == 0:
    #
    #         print(i_batch, sample_batched['video_x'].size(), sample_batched['video_label'].size())

#b,t,c,h,w
print(myUCF101.__len__())